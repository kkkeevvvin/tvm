/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <HAP_compute_res.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>

#include "tvm/runtime/hexagon/ops/conv2d.h"

// Current limitations:
// - N in NHWC must be 1
// - dilated convolutions are not supported
// - Bias is not accepted
// - Optional "relu" is not performed

// Packed arguments:
//   0: DLTensor activations (NHWC)
//   1: DLTensor weights (HWIO)
//   2: int offset_top
//   3: int offset_left
//   4: int stride_h
//   5: int stride_w
//   6: DLTensor output (NHWC)
extern "C" int conv2d_packed_fp16(TVMValue* args, int* type_codes, int num_args, TVMValue* out_val,
                                  int out_code, void* res_handle);

namespace tvm {
namespace runtime {
namespace hexagon {

/**
 * @brief Returns the pointer to the element within the given block
 * assuming fp16 type and speicific layout as mentioned in blockize_hwc_16b.
 * All the below params are explained with the same layout assumption
 *
 * @param block_out_y y-index of block
 * @param block_out_x x-index of block
 * @param block_out_c c-index of block
 * @param yi height-offset within the block
 * @param xio outer width offset within the block
 * @param ci channel offset within the block
 * @param xii inner width offset within the block
 * @param block base DLTensor
 *
 * @return The pointer to the element within the given block
 */
static inline uint16_t* getElementPtr(int block_out_y, int block_out_x, int block_out_c, int yi,
                                      int xio, int ci, int xii, const DLTensor& tensor) {
  auto block_ptr = nhwc_at(tensor, 0, block_out_y, block_out_x, block_out_c);
  auto block_offset = yi * 128 + xio * 64 + ci * 2 + xii;
  auto first_element_ptr = reinterpret_cast<uint16_t*>(block_ptr);
  return first_element_ptr + block_offset;
}

/**
 * @brief Compute 2 vectors with ones in the even and odd lanes
 *
 * Output vectors are:
 * vector 1     = [0xFFFF,0x0000,0xFFFFF,0x0000,...,0xFFFF,0x0000]
 * vector lanes = [   0  ,   2  ,   3   ,   4  ,...,   62 ,   63 ]
 *
 * vector 2     = [0x0000,0xFFFF,0x0000,0xFFFFF,...,0xFFFF,0x0000]
 * vector lanes = [   0  ,   2  ,   3   ,   4  ,...,   62 ,   63 ]
 *
 * @return Return the 2 vectors
 */
inline std::pair<HVX_Vector, HVX_Vector> getOddEvenOnes() {
  HVX_Vector v0 = Q6_V_vzero();
  HVX_Vector v1 = Q6_Vh_vsplat_R(0xFFFF);

  HVX_Vector v1e = Q6_Vh_vshuffe_VhVh(v0, v1);
  HVX_Vector v1o = Q6_V_vnot_V(v1e);
  return {v1e, v1o};
}

/**
 * @brief Return the input vector filled with the 2 channel elements(which is the 1st and 3rd
 * element) from base_ptr filled up 32 times to get 64 elements
 *
 * 1. It's generated by first creating 2 vectors "splatted" with the 2 required elements
 * 2. Then we andd it with vectors containing all ones (0xFFFF) in the even and odd lanes
 * 3. Finally those 2 vectors are OR'ed together
 *
 * @param base_ptr pointer to the first of the 2 channel elements to be filled
 *
 * @return input vector
 */
inline HVX_Vector getInputVector(uint16_t* base_ptr) {
  HVX_Vector v1 = Q6_Vh_vsplat_R(base_ptr[0]);
  HVX_Vector v2 = Q6_Vh_vsplat_R(base_ptr[2]);

  auto oddEvenOnes = getOddEvenOnes();
  auto v1e = oddEvenOnes.first;
  auto v1o = oddEvenOnes.second;

  HVX_Vector v_even_vals = Q6_V_vand_VV(v1, v1e);
  HVX_Vector v_odd_vals = Q6_V_vand_VV(v2, v1o);

  return Q6_V_vor_VV(v_even_vals, v_odd_vals);
}

/**
 * @brief Return the Output vector which contains the 32 output channels in the even lanes
 *
 * The output vector is commputed as:
 * 1. vector multiply(vmpy) of input and weights
 * 2. Rotate the vector right by 1 element and add with the first vector to add the 2 input channels
 * 3. Then convert the results back from qfloat16 to IEEE half-precision float
 * 4. The added values are in even lanes, so zero out the odd lanes by anding with ones in even
 * lanes and return
 *
 * @param act_vec Input activations vector
 * @param wgt_vec Weights vector
 *
 * @return output vector with 32 output channels even lanes
 */
inline HVX_Vector computeOuputVector(HVX_Vector act_vec, HVX_Vector wgt_vec) {
  HVX_Vector v_res = Q6_Vqf16_vmpy_VhfVhf(act_vec, wgt_vec);  // result is in qf16
  HVX_Vector v_rot = Q6_V_vror_VR(v_res, 2);
  HVX_Vector v_reduced = Q6_Vqf16_vadd_Vqf16Vqf16(v_res, v_rot);
  HVX_Vector v_hf = Q6_Vhf_equals_Vqf16(v_reduced);
  HVX_Vector v1e = getOddEvenOnes().first;
  HVX_Vector v_reduced_even_lanes = Q6_V_vand_VV(v_hf, v1e);
  return v_reduced_even_lanes;
}

static int round_down(int v, int base) { return v - (v % base); }

/**
 * @brief Compute the convolution of inputs from cr_act, and weights from
 * cr_filt to update the output to cr_out. The goal is to have an efficient
 * HVX implementation
 *
 * Assumptions:
 * -----------
 * - This implementation right now assumes that the dilation is 1
 * - there is zero padding or the input was already pre-padded.
 * - block specific spatial padding is only expected at the end and hence
 *   pad_top and pad_left are not yet used
 * - Relu activation is not used
 * - Bias add is not done
 *
 * @param cr_out blockized output tensor with zeros already filled in
 * @param cr_act blockized activations
 * @param cr_filt Chunkified weights as returned from output of prepare_hwio
 * @param out_shape Original output shape of the tensor before blockization
 * @param act_shape Original input shape
 * @param bias_flat Flat bias values and are not used right now
 *        TODO (quic-sanirudh) Add support for bias add
 * @param filt_shape Original filter shape
 * @param pad_shape Pad top and pad left shape
 * @param relu Whether to apply relu after convolution, not done right now
 *        TODO (quic-sanirudh) Add support for relu activation
 * @param zero_block A block filled with zeros
 *
 * @return
 */
void conv_layer_fp16_hvx(DLTensor& cr_out, const DLTensor& cr_act,  // NOLINT(*)
                         const DLTensor& cr_filt, const DLTensor& out_shape,
                         const DLTensor& act_shape, const DLTensor& bias_flat,
                         const DLTensor& filt_shape, const DLTensor& pad_shape, bool relu,
                         int stride_h, int stride_w, uintptr_t zero_block) {
  int64_t filt_height = filt_shape.shape[0];
  int64_t filt_width = filt_shape.shape[1];
  int64_t filt_idepth = filt_shape.shape[2];

  int pad_top = pad_shape.shape[0];
  int pad_left = pad_shape.shape[1];
  LOG_INFO << "filt_height=" << filt_height << ", filt_width=" << filt_width
           << ", filt_idepth=" << filt_idepth << ", pad_top=" << pad_top
           << ", pad_left=" << pad_left << "\n";

  ICHECK_LT(pad_top, 8) << "pad_top offset cannot be >= 8";
  ICHECK_LT(pad_left, 4) << "pad_left offset cannot be >= 4";

  int a_height = cr_act.shape[1];
  int a_width = cr_act.shape[2];
  int a_depth = cr_act.shape[3];

  int w_height = cr_filt.shape[0];
  int w_width = cr_filt.shape[1];

  int o_depth = cr_out.shape[3];
  int b_depth = bias_flat.shape[0];

  int o_height = cr_out.shape[1];
  int o_width = cr_out.shape[2];

  int out_height = out_shape.shape[1];
  int out_width = out_shape.shape[2];

  LOG_INFO << "a: 1x" << a_height << "x" << a_width << "x" << a_depth << ", w: " << w_height << "x"
           << w_width << "x" << static_cast<int>(cr_filt.shape[2]) << "x"
           << static_cast<int>(cr_filt.shape[3]) << ", o: 1x" << o_height << "x" << o_width << "x"
           << o_depth << ", b: " << b_depth << ", out_shape: " << out_height << "x" << out_width
           << "\n";

  ICHECK_EQ(a_depth, cr_filt.shape[2]) << "input depth should match weights input channels";
  ICHECK_EQ(o_depth, cr_filt.shape[3]) << "output depth should match the weights output channel";

  int rd = round_down(filt_width, 4);
  int wgt_chunk_thin_width = filt_width - rd;

  /*
   * Compute the output vector of either 1 or 2 elements along the width and max 32 elements along
   * the depth to constitue a maximum of 64 elements
   *
   * The weights are loaded directly in the order they're stored, which results
   * in 2 input channels and 32 output channels
   *
   * Weights vector illustration:
   * ------- ------ ------------
   * weights_vec = [0-0,0-1,1-0,1-1,2-0,2-1,3-0,3-1,4-0,4-1,...,31-0,31-1] -> This is the
   * vector representation of weights, where the elements are represented as
   * "out_channel-input_channel"
   *
   *
   * Same 2 input channels have to be multiplied across all output channels in the weights.
   *
   * Activations vector would thus be:
   * ----------- ------ ----- ---- --
   * act_vec = [i0,i1,i0,i1,i0,i1,...,i0,i1] - 2 elements of the input channels broadcasted 32 times
   * to fill 64 elements of the vector
   *
   *
   * Thus the computation is just a vmpy(act_vec,weights_vec) followed by a some rearrangement to
   * add every pair of 16b lanes in the vector to reduce along the input channels
   *
   * This result is added to the result of the next pair of input channels all the way until we
   * have reduced across the entire input channels.
   *
   * Then the same vector is added to the results of the following elements along the width and
   * height to finally get 32 elements representing 32 output channels.
   *
   * Since the output block also has the 8h2w32c2w format, the 32 elements of the next element
   * along the width is also added into the the same vector such that the first 32 channel elements
   * occupy the even lanes and the next 32 occupy the odd lanes to form a single 64-element vector
   * which is then stored
   */
  auto computeConv = [filt_height, filt_width, wgt_chunk_thin_width, filt_idepth, stride_h,
                      stride_w, &cr_out, &cr_act, &cr_filt](int out_act_y, int out_act_x, int out_c,
                                                            int h, int wo, bool skip_wi_1 = false) {
    auto out_element_ptr = getElementPtr(out_act_y, out_act_x, out_c, h, wo, 0, 0, cr_out);

    LOG_INFO << "out_act_y: " << out_act_y << ", out_act_x: " << out_act_x << ", out_c: " << out_c
             << ", h: " << h << ", wo: " << wo << " out_element_ptr: " << out_element_ptr;

    HVX_Vector* out_vector = reinterpret_cast<HVX_Vector*>(out_element_ptr);
    HVX_Vector existing_out_vec = *out_vector;

    for (int fh = 0; fh < filt_height; ++fh) {
      for (int fw = 0; fw < filt_width; ++fw) {
        int fch = fh / 8;
        int fcw = 0;
        if (fw >= wgt_chunk_thin_width) {
          fcw = (fw - wgt_chunk_thin_width) / 4 + 1;
        }
        int fx = (fw < wgt_chunk_thin_width) ? fw : ((fw - wgt_chunk_thin_width) % 4);
        int fy = fh % 8;
        for (int c = 0; c < round_up(filt_idepth, 2); c += 2) {
          int out_act_cc = c / 32;
          int ci = c % 32;
          auto wgt_chunk = hwio_at(cr_filt, fch, fcw, out_act_cc, out_c);

          // Find weight chunk offset ptr
          int max_x = (fcw == 0) ? wgt_chunk_thin_width : 4;

          int wi = 0;

          int out_width_idx = out_act_x * 4 + wo * 2 + wi;
          int act_width_access_idx = out_width_idx * stride_w + fw;
          int true_out_act_x = act_width_access_idx / 4;
          int true_wo = (act_width_access_idx % 4) / 2;
          int true_wi = act_width_access_idx % 2;

          int out_height_idx = out_act_y * 8 + h;
          int act_height_access_idx = out_height_idx * stride_h + fh;
          int true_out_act_y = act_height_access_idx / 8;
          int true_h = act_height_access_idx % 8;

          int act_channel_idx = out_act_cc * 32 + ci;

          auto act_element_ptr = getElementPtr(true_out_act_y, true_out_act_x, out_act_cc, true_h,
                                               true_wo, ci, true_wi, cr_act);
          HVX_Vector act_vec = getInputVector(act_element_ptr);

          auto wgt_chunk_offset = hwio_to_sm_16b(max_x, fy, fx, ci, 0);
          auto base_chunk_ptr = reinterpret_cast<uint16_t*>(wgt_chunk);
          auto chunk_ptr = base_chunk_ptr + wgt_chunk_offset;

          LOG_INFO << "act:  0x" << act_height_access_idx << "x" << act_width_access_idx << "x"
                   << act_channel_idx << ", wgt: " << fh << "x" << fw << "x" << act_channel_idx
                   << "x" << out_c * 32 << ", out: 0x" << out_height_idx << "x" << out_width_idx
                   << "x" << out_c * 32 << ", wgt_chunk_offset: " << wgt_chunk_offset;

          const HVX_Vector* weights_vec_ptr = reinterpret_cast<const HVX_Vector*>(chunk_ptr);
          HVX_Vector weights_vec = *weights_vec_ptr;

          HVX_Vector reduced_vec_even_elements = computeOuputVector(act_vec, weights_vec);

          if (!skip_wi_1) {
            wi = 1;

            out_width_idx = out_act_x * 4 + wo * 2 + wi;
            act_width_access_idx = out_width_idx * stride_w + fw;
            true_out_act_x = act_width_access_idx / 4;
            true_wo = (act_width_access_idx % 4) / 2;
            true_wi = act_width_access_idx % 2;

            act_element_ptr = getElementPtr(true_out_act_y, true_out_act_x, out_act_cc, true_h,
                                            true_wo, ci, true_wi, cr_act);
            act_vec = getInputVector(act_element_ptr);

            LOG_INFO << "act:  0x" << act_height_access_idx << "x" << act_width_access_idx << "x"
                     << act_channel_idx << ", wgt: " << fh << "x" << fw << "x" << act_channel_idx
                     << "x" << out_c * 32 << ", out: 0x" << out_height_idx << "x" << out_width_idx
                     << "x" << out_c * 32 << ", wgt_chunk_offset: " << wgt_chunk_offset;

            HVX_Vector reduced_vec_odd_elements = computeOuputVector(act_vec, weights_vec);
            reduced_vec_odd_elements = Q6_V_vror_VR(reduced_vec_odd_elements, -2);
            HVX_Vector out_final = Q6_V_vor_VV(reduced_vec_even_elements, reduced_vec_odd_elements);

            HVX_Vector out_vec_qf16 = Q6_Vqf16_vadd_VhfVhf(out_final, existing_out_vec);
            existing_out_vec = Q6_Vhf_equals_Vqf16(out_vec_qf16);
          } else {
            HVX_Vector out_vec_qf16 =
                Q6_Vqf16_vadd_VhfVhf(reduced_vec_even_elements, existing_out_vec);
            existing_out_vec = Q6_Vhf_equals_Vqf16(out_vec_qf16);
          }
        }
      }
    }
    *out_vector = existing_out_vec;
  };

  auto computeFullWidth = [&computeConv](int out_y, int out_x, int out_c, int h) {
    for (int wo = 0; wo < 2; ++wo) {
      computeConv(out_y, out_x, out_c, h, wo);
    }
  };

  auto computePartialWidth = [out_width, o_width, &computeConv](int out_y, int out_c, int h) {
    int out_x = o_width - 1;
    int wo = 0;
    for (; wo < (out_width % 4) / 2; ++wo) {
      computeConv(out_y, out_x, out_c, h, wo);
    }

    if (out_width % 2) {
      computeConv(out_y, out_x, out_c, h, wo, true /* skip_wi_1 */);
    }
  };

  for (int out_c = 0; out_c < cr_filt.shape[3]; ++out_c) {
    for (int out_act_y = 0; out_act_y < out_height / 8; ++out_act_y) {
      int out_y = out_act_y;
      for (int out_act_x = 0; out_act_x < out_width / 4; ++out_act_x) {
        int out_x = out_act_x;
        for (int h = 0; h < 8; ++h) {
          computeFullWidth(out_y, out_x, out_c, h);
        }
      }

      for (int h = 0; h < 8; ++h) {
        computePartialWidth(out_y, out_c, h);
      }
    }

    int out_y = o_height - 1;
    for (int h = 0; h < out_height % 8; ++h) {
      for (int out_act_x = 0; out_act_x < out_width / 4; ++out_act_x) {
        int out_x = out_act_x;
        computeFullWidth(out_y, out_x, out_c, h);
      }
      computePartialWidth(out_y, out_c, h);
    }
  }
}
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

int conv2d_packed_fp16(TVMValue* args, int* type_codes, int num_args, TVMValue* out_val,
                       int out_code, void* res_handle) {
  namespace hexagonrt = tvm::runtime::hexagon;
  ICHECK_EQ(num_args, 7) << "Unexpected number of arguments";
  ICHECK_EQ(type_codes[0], kTVMDLTensorHandle)
      << "First argument is expected to be the input tensor";  // Input activations
  ICHECK_EQ(type_codes[1], kTVMDLTensorHandle)
      << "Second argument is expected to be the weights tensor";  // Weights
  ICHECK_EQ(type_codes[2], kDLInt)
      << "Third argument is expected to be the pad_top offset";  // pad_top offset
  ICHECK_EQ(type_codes[3], kDLInt)
      << "Fourth argument is expected to be the pad_left offset";  // pad_left offset
  ICHECK_EQ(type_codes[4], kDLInt) << "Fifth argument is expected to be the stride_h";  // stride_h
  ICHECK_EQ(type_codes[5], kDLInt) << "Sixth argument is expected to be the stride_w";  // stride_w
  ICHECK_EQ(type_codes[6], kTVMDLTensorHandle)
      << "Seventh argument is expected to be the output tensor";  // output

  auto* act_flat = static_cast<DLTensor*>(args[0].v_handle);
  auto* wgt_flat = static_cast<DLTensor*>(args[1].v_handle);
  auto* out_flat = static_cast<DLTensor*>(args[6].v_handle);

  // Temporary assertion until multiple batches are supported
  ICHECK_EQ(act_flat->shape[0], 1) << "Input batch size more than 1 is not supported yet";

  // Temporary assertion until multiple batches are supported
  ICHECK_EQ(out_flat->shape[0], 1) << "Output batch size more than 1 is not supported yet";

  int pad_top = args[2].v_int64;
  int pad_left = args[3].v_int64;
  int stride_h = args[4].v_int64;
  int stride_w = args[5].v_int64;

  LOG_INFO << "act.shape=" << act_flat->shape[0] << "x" << act_flat->shape[1] << "x"
           << act_flat->shape[2] << "x" << act_flat->shape[3]
           << ", wgt.shape=" << wgt_flat->shape[0] << "x" << wgt_flat->shape[1] << "x"
           << wgt_flat->shape[2] << "x" << wgt_flat->shape[3] << ", pad_top=" << pad_top
           << ", pad_left=" << pad_left;

  auto* device_api = tvm::runtime::DeviceAPI::Get(hexagonrt::hexagon_device, false);
  ICHECK(device_api != nullptr);
  tvm::runtime::String vtcm_scope = "global.vtcm";

  auto act_vtcm = hexagonrt::prepare_nhwc(device_api, act_flat, /*copy_data=*/true);

  ICHECK_NE(wgt_flat->shape[0], 0) << "Weights height should not be zero";
  ICHECK_NE(wgt_flat->shape[1], 0) << "Weights width should not be zero";
  ICHECK_NE(wgt_flat->shape[2], 0) << "Weights input channels should not be zero";
  ICHECK_NE(wgt_flat->shape[3], 0) << "Weights output channels should not be zero";
  int num_wgt_chunks = hexagonrt::calculate_num_weight_chunks(wgt_flat->shape);
  LOG_INFO << "num_wgt_chunks: " << num_wgt_chunks;
  auto wgt_ptr_table =
      reinterpret_cast<void**>(__builtin_alloca(num_wgt_chunks * sizeof(uintptr_t)));
  auto wgt_vtcm = hexagonrt::prepare_hwio(device_api, wgt_flat, num_wgt_chunks, wgt_ptr_table);

  auto out_vtcm = hexagonrt::prepare_nhwc(device_api, out_flat, /*copy_data=*/false);

  // Prepare zero_block
  int64_t block_nbytes = 2048;
  void* zero_block = device_api->AllocDataSpace(hexagonrt::hexagon_device, 1, &block_nbytes,
                                                tvm::runtime::DataType::UInt(8), vtcm_scope);
  memset(zero_block, 0, 2048);

  // FIXME: Setting bias to zero_block: this works for up to 256 output channels.
  auto bias_flat =
      hexagonrt::SDLTensor<1>(zero_block, wgt_flat->dtype, zero_block, &wgt_flat->shape[3]);
  auto act_shape = hexagonrt::SDLTensor<4>(nullptr, act_flat->dtype, nullptr, act_flat->shape);
  auto filt_shape = hexagonrt::SDLTensor<4>(nullptr, wgt_flat->dtype, nullptr, wgt_flat->shape);
  auto pad_shape = hexagonrt::SDLTensor<2>(nullptr, act_flat->dtype, nullptr, {pad_top, pad_left});
  auto out_shape = hexagonrt::SDLTensor<4>(nullptr, out_flat->dtype, nullptr, out_flat->shape);
  bool relu = false;

  hexagonrt::conv_layer_fp16_hvx(out_vtcm, act_vtcm, wgt_vtcm, out_shape, act_shape, bias_flat,
                                 filt_shape, pad_shape, relu, stride_h, stride_w,
                                 hexagonrt::to_uint(zero_block));

  hexagonrt::deblockize_hwc_16b(out_flat->data, out_vtcm.data, out_flat->shape[1],
                                out_flat->shape[2], out_flat->shape[3]);

  device_api->FreeDataSpace(hexagonrt::hexagon_device, zero_block);
  hexagonrt::release(device_api, out_vtcm);
  hexagonrt::release(device_api, wgt_vtcm);
  hexagonrt::release(device_api, act_vtcm);

  return 0;
}