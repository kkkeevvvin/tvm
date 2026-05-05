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

/*!
 * \file src/relax/analysis/dnnfusion_mapping.cc
 */

#include "dnnfusion_mapping.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <unordered_map>

namespace tvm {
namespace relax {
namespace dnnfusion {

namespace {

// ---------------------------------------------------------------------------
// Table 2 (DNNFusion §3.1): explicit op-name -> MappingType.
//
// Mapping derived from the ONNX op classification in the paper, translated
// to the corresponding TVM Relax op registry names verified against
// /home/kevin/tvm-dnnf/src/relax/op/{tensor,nn,image,statistical,...}.
//
// Entries that have no direct relax equivalent (e.g. ONNX Reciprocal is
// emitted as relax.divide(1, x), DepthToSpace decomposes to reshape +
// permute_dims) are intentionally absent; the OpPatternKind fallback
// covers them.
// ---------------------------------------------------------------------------
const std::unordered_map<std::string, MappingType>& OpNameToMappingType() {
  static const std::unordered_map<std::string, MappingType> table = {
      // ===== One-to-One =====
      // Binary elemwise (RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL)
      {"relax.add", MappingType::kOneToOne},
      {"relax.subtract", MappingType::kOneToOne},
      {"relax.multiply", MappingType::kOneToOne},
      {"relax.divide", MappingType::kOneToOne},
      {"relax.floor_divide", MappingType::kOneToOne},
      {"relax.mod", MappingType::kOneToOne},
      {"relax.floor_mod", MappingType::kOneToOne},
      {"relax.power", MappingType::kOneToOne},
      {"relax.log_add_exp", MappingType::kOneToOne},
      {"relax.maximum", MappingType::kOneToOne},
      {"relax.minimum", MappingType::kOneToOne},
      {"relax.left_shift", MappingType::kOneToOne},
      {"relax.right_shift", MappingType::kOneToOne},
      {"relax.bitwise_and", MappingType::kOneToOne},
      {"relax.bitwise_or", MappingType::kOneToOne},
      {"relax.bitwise_xor", MappingType::kOneToOne},
      {"relax.logical_and", MappingType::kOneToOne},
      {"relax.logical_or", MappingType::kOneToOne},
      {"relax.logical_xor", MappingType::kOneToOne},
      // Comparison (RELAX_REGISTER_CMP_OP_AND_IMPL): paper's Greater family.
      {"relax.equal", MappingType::kOneToOne},
      {"relax.greater", MappingType::kOneToOne},
      {"relax.greater_equal", MappingType::kOneToOne},
      {"relax.less", MappingType::kOneToOne},
      {"relax.less_equal", MappingType::kOneToOne},
      {"relax.not_equal", MappingType::kOneToOne},
      // Unary arith / check (RELAX_REGISTER_UNARY_*_OP_AND_IMPL)
      {"relax.abs", MappingType::kOneToOne},
      {"relax.acos", MappingType::kOneToOne},
      {"relax.acosh", MappingType::kOneToOne},
      {"relax.asin", MappingType::kOneToOne},
      {"relax.asinh", MappingType::kOneToOne},
      {"relax.atan", MappingType::kOneToOne},
      {"relax.atanh", MappingType::kOneToOne},
      {"relax.bitwise_not", MappingType::kOneToOne},
      {"relax.ceil", MappingType::kOneToOne},
      {"relax.cos", MappingType::kOneToOne},
      {"relax.cosh", MappingType::kOneToOne},
      {"relax.erf", MappingType::kOneToOne},
      {"relax.exp", MappingType::kOneToOne},
      {"relax.floor", MappingType::kOneToOne},
      {"relax.isfinite", MappingType::kOneToOne},
      {"relax.isinf", MappingType::kOneToOne},
      {"relax.isnan", MappingType::kOneToOne},
      {"relax.log", MappingType::kOneToOne},
      {"relax.logical_not", MappingType::kOneToOne},
      {"relax.negative", MappingType::kOneToOne},
      {"relax.round", MappingType::kOneToOne},
      {"relax.rsqrt", MappingType::kOneToOne},
      {"relax.sigmoid", MappingType::kOneToOne},
      {"relax.sign", MappingType::kOneToOne},
      {"relax.sin", MappingType::kOneToOne},
      {"relax.sinh", MappingType::kOneToOne},
      {"relax.sqrt", MappingType::kOneToOne},
      {"relax.square", MappingType::kOneToOne},
      {"relax.tan", MappingType::kOneToOne},
      {"relax.tanh", MappingType::kOneToOne},
      {"relax.trunc", MappingType::kOneToOne},
      // Other paper-OneToOne with explicit relax registrations
      {"relax.astype", MappingType::kOneToOne},          // ONNX Cast
      {"relax.clip", MappingType::kOneToOne},
      {"relax.where", MappingType::kOneToOne},           // Ternary, but per-element
      {"relax.concat", MappingType::kOneToOne},          // Paper Table 2 lists Concat as OtO
      {"relax.strided_slice", MappingType::kOneToOne},   // ONNX Slice
      {"relax.dynamic_strided_slice", MappingType::kOneToOne},
      {"relax.split", MappingType::kOneToOne},
      {"relax.ewise_fma", MappingType::kOneToOne},
      // NN unary activations
      {"relax.nn.relu", MappingType::kOneToOne},
      {"relax.nn.gelu", MappingType::kOneToOne},
      {"relax.nn.gelu_tanh", MappingType::kOneToOne},
      {"relax.nn.selu", MappingType::kOneToOne},
      {"relax.nn.silu", MappingType::kOneToOne},
      {"relax.nn.softplus", MappingType::kOneToOne},
      {"relax.nn.leakyrelu", MappingType::kOneToOne},
      {"relax.nn.prelu", MappingType::kOneToOne},
      // BatchNorm: paper Table 2 lists BatchNormalization as OtO.  Relax's
      // batch_norm returns a tuple (out, moving_mean, moving_var) and is
      // typically decomposed by DecomposeOpsForInference before FuseOps;
      // when it survives it should still be classified as OtO.
      {"relax.nn.batch_norm", MappingType::kOneToOne},
      {"relax.nn.dropout", MappingType::kOneToOne},

      // ===== One-to-Many =====
      {"relax.broadcast_to", MappingType::kOneToMany},      // ONNX Expand
      {"relax.take", MappingType::kOneToMany},              // ONNX Gather
      {"relax.gather_elements", MappingType::kOneToMany},
      {"relax.gather_nd", MappingType::kOneToMany},
      {"relax.repeat", MappingType::kOneToMany},
      {"relax.tile", MappingType::kOneToMany},
      {"relax.image.resize2d", MappingType::kOneToMany},    // ONNX Resize / Upsample
      {"relax.image.grid_sample", MappingType::kOneToMany},

      // ===== Many-to-Many (incl. Many-to-One) =====
      {"relax.matmul", MappingType::kManyToMany},           // ONNX GEMM / MatMul
      {"relax.einsum", MappingType::kManyToMany},
      {"relax.outer", MappingType::kManyToMany},
      {"relax.cumsum", MappingType::kManyToMany},
      {"relax.cumprod", MappingType::kManyToMany},
      // Reductions (RELAX_REGISTER_STATISTICAL_OP_INTERFACE)
      {"relax.sum", MappingType::kManyToMany},              // ReduceSum
      {"relax.mean", MappingType::kManyToMany},             // ReduceMean
      {"relax.max", MappingType::kManyToMany},              // ReduceMax
      {"relax.min", MappingType::kManyToMany},              // ReduceMin
      {"relax.prod", MappingType::kManyToMany},             // ReduceProd
      {"relax.std", MappingType::kManyToMany},
      {"relax.variance", MappingType::kManyToMany},
      {"relax.median", MappingType::kManyToMany},
      {"relax.argmax", MappingType::kManyToMany},
      {"relax.argmin", MappingType::kManyToMany},
      // Norms: paper's MtM bucket includes InstanceNorm; LayerNorm /
      // GroupNorm / RMSNorm follow the same reduction-then-affine shape.
      {"relax.nn.layer_norm", MappingType::kManyToMany},
      {"relax.nn.group_norm", MappingType::kManyToMany},
      {"relax.nn.instance_norm", MappingType::kManyToMany},
      {"relax.nn.rms_norm", MappingType::kManyToMany},
      // Softmax: paper Table 2 lists it as MtM (reduction along axis).
      {"relax.nn.softmax", MappingType::kManyToMany},
      {"relax.nn.log_softmax", MappingType::kManyToMany},
      // Conv / Pool families
      {"relax.nn.conv1d", MappingType::kManyToMany},
      {"relax.nn.conv2d", MappingType::kManyToMany},
      {"relax.nn.conv3d", MappingType::kManyToMany},
      {"relax.nn.conv1d_transpose", MappingType::kManyToMany},
      {"relax.nn.conv2d_transpose", MappingType::kManyToMany},
      {"relax.nn.avg_pool1d", MappingType::kManyToMany},
      {"relax.nn.avg_pool2d", MappingType::kManyToMany},
      {"relax.nn.avg_pool3d", MappingType::kManyToMany},
      {"relax.nn.max_pool1d", MappingType::kManyToMany},
      {"relax.nn.max_pool2d", MappingType::kManyToMany},
      {"relax.nn.max_pool3d", MappingType::kManyToMany},
      {"relax.nn.adaptive_avg_pool1d", MappingType::kManyToMany},
      {"relax.nn.adaptive_avg_pool2d", MappingType::kManyToMany},
      {"relax.nn.adaptive_avg_pool3d", MappingType::kManyToMany},
      {"relax.nn.attention", MappingType::kManyToMany},
      {"relax.nn.attention_bias", MappingType::kManyToMany},
      {"relax.nn.attention_var_len", MappingType::kManyToMany},

      // ===== Reorganize =====
      {"relax.reshape", MappingType::kReorganize},
      {"relax.flatten", MappingType::kReorganize},
      {"relax.squeeze", MappingType::kReorganize},
      {"relax.expand_dims", MappingType::kReorganize},      // ONNX Unsqueeze
      {"relax.layout_transform", MappingType::kReorganize},

      // ===== Shuffle =====
      {"relax.permute_dims", MappingType::kShuffle},        // ONNX Transpose
      {"relax.flip", MappingType::kShuffle},
  };
  return table;
}

// Fallback: derive a reasonable MappingType from OpPatternKind for ops
// not explicitly listed in OpNameToMappingType().  The mapping reflects
// how each TVM pattern *behaves* at the loop level, not a strict
// correspondence with the paper's ONNX-based taxonomy.
MappingType MappingTypeFromPattern(OpPatternKind pattern) {
  switch (pattern) {
    case kElemWise:
      return MappingType::kOneToOne;
    case kBroadcast:
      // TVM kBroadcast is "in-order axis broadcast", which the paper would
      // describe as OneToMany when the broadcast actually expands.
      return MappingType::kOneToMany;
    case kInjective:
      // Index-permuting, locality-preserving in TVM's classification.
      return MappingType::kReorganize;
    case kCommReduce:
      return MappingType::kManyToMany;
    case kOutEWiseFusable:
      return MappingType::kManyToMany;
    case kTuple:
    case kOpaque:
    default:
      return MappingType::kUnknown;
  }
}

// ---------------------------------------------------------------------------
// Table 3 (DNNFusion §3.2): 5x5 fusion legality matrix.
//
// Indexed by [first][second] where indices match MappingType::values
// (kUnknown=0, kOneToOne=1, kOneToMany=2, kManyToMany=3,
// kReorganize=4, kShuffle=5).  Row/column 0 (Unknown) is always Red.
// ---------------------------------------------------------------------------
constexpr int kNumMappingTypes = 6;

struct TableEntry {
  FuseColor color;
  MappingType result;
};

// Verbatim from the paper's Table 3.  Each row is the "first" op, each
// column the "second" op.  Layout: { Unknown, OtO, OtM, MtM, Reorg, Shuffle }.
constexpr TableEntry kFuseTable[kNumMappingTypes][kNumMappingTypes] = {
    // first = Unknown  (anything involving Unknown is a fusion barrier)
    {
        {FuseColor::kRed, MappingType::kUnknown},
        {FuseColor::kRed, MappingType::kUnknown},
        {FuseColor::kRed, MappingType::kUnknown},
        {FuseColor::kRed, MappingType::kUnknown},
        {FuseColor::kRed, MappingType::kUnknown},
        {FuseColor::kRed, MappingType::kUnknown},
    },
    // first = OneToOne
    {
        {FuseColor::kRed,    MappingType::kUnknown},
        {FuseColor::kGreen,  MappingType::kOneToOne},     // OtO x OtO  -> OtO  (G)
        {FuseColor::kYellow, MappingType::kOneToMany},    // OtO x OtM  -> OtM  (Y)
        {FuseColor::kYellow, MappingType::kManyToMany},   // OtO x MtM  -> MtM  (Y)
        {FuseColor::kGreen,  MappingType::kReorganize},   // OtO x Reorg -> Reorg (G)
        {FuseColor::kGreen,  MappingType::kShuffle},      // OtO x Shuf -> Shuf (G)
    },
    // first = OneToMany
    {
        {FuseColor::kRed,    MappingType::kUnknown},
        {FuseColor::kGreen,  MappingType::kOneToMany},    // OtM x OtO  -> OtM  (G)
        {FuseColor::kYellow, MappingType::kOneToMany},    // OtM x OtM  -> OtM  (Y)
        {FuseColor::kRed,    MappingType::kUnknown},      // OtM x MtM  -> illegal
        {FuseColor::kYellow, MappingType::kOneToMany},    // OtM x Reorg -> OtM (Y)
        {FuseColor::kYellow, MappingType::kOneToMany},    // OtM x Shuf -> OtM  (Y)
    },
    // first = ManyToMany
    {
        {FuseColor::kRed,    MappingType::kUnknown},
        {FuseColor::kGreen,  MappingType::kManyToMany},   // MtM x OtO  -> MtM  (G)
        {FuseColor::kYellow, MappingType::kManyToMany},   // MtM x OtM  -> MtM  (Y)
        {FuseColor::kRed,    MappingType::kUnknown},      // MtM x MtM  -> illegal
        {FuseColor::kYellow, MappingType::kManyToMany},   // MtM x Reorg -> MtM (Y)
        {FuseColor::kYellow, MappingType::kManyToMany},   // MtM x Shuf -> MtM  (Y)
    },
    // first = Reorganize
    {
        {FuseColor::kRed,    MappingType::kUnknown},
        {FuseColor::kGreen,  MappingType::kReorganize},   // Reorg x OtO  -> Reorg (G)
        {FuseColor::kYellow, MappingType::kOneToMany},    // Reorg x OtM  -> OtM   (Y)
        {FuseColor::kYellow, MappingType::kManyToMany},   // Reorg x MtM  -> MtM   (Y)
        {FuseColor::kGreen,  MappingType::kReorganize},   // Reorg x Reorg -> Reorg (G)
        {FuseColor::kGreen,  MappingType::kReorganize},   // Reorg x Shuf -> Reorg (G)
    },
    // first = Shuffle
    {
        {FuseColor::kRed,    MappingType::kUnknown},
        {FuseColor::kGreen,  MappingType::kShuffle},      // Shuf x OtO  -> Shuf  (G)
        {FuseColor::kYellow, MappingType::kOneToMany},    // Shuf x OtM  -> OtM   (Y)
        {FuseColor::kYellow, MappingType::kManyToMany},   // Shuf x MtM  -> MtM   (Y)
        {FuseColor::kGreen,  MappingType::kReorganize},   // Shuf x Reorg -> Reorg (G)
        {FuseColor::kGreen,  MappingType::kShuffle},      // Shuf x Shuf -> Shuf  (G)
    },
};

}  // namespace

MappingType DeriveMappingType(const std::string& op_name, OpPatternKind pattern) {
  if (!op_name.empty()) {
    const auto& table = OpNameToMappingType();
    auto it = table.find(op_name);
    if (it != table.end()) return it->second;
  }
  return MappingTypeFromPattern(pattern);
}

FuseDecision FuseMappingCheck(MappingType first, MappingType second) {
  int i = static_cast<int>(first);
  int j = static_cast<int>(second);
  if (i < 0 || i >= kNumMappingTypes || j < 0 || j >= kNumMappingTypes) {
    return FuseDecision{FuseColor::kRed, MappingType::kUnknown};
  }
  const TableEntry& e = kFuseTable[i][j];
  return FuseDecision{e.color, e.result};
}

const char* MappingTypeName(MappingType t) {
  switch (t) {
    case MappingType::kOneToOne:   return "OneToOne";
    case MappingType::kOneToMany:  return "OneToMany";
    case MappingType::kManyToMany: return "ManyToMany";
    case MappingType::kReorganize: return "Reorganize";
    case MappingType::kShuffle:    return "Shuffle";
    case MappingType::kUnknown:    return "Unknown";
  }
  return "Unknown";
}

const char* FuseColorName(FuseColor c) {
  switch (c) {
    case FuseColor::kRed:    return "Red";
    case FuseColor::kYellow: return "Yellow";
    case FuseColor::kGreen:  return "Green";
  }
  return "Unknown";
}

// ---------------------------------------------------------------------------
// FFI bridge so Python tests can drive the classifier without compiling
// against libtvm headers.  Both functions take/return ints to keep the
// boundary simple.
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.DnnfusionDeriveMappingType",
           [](ffi::String op_name, int pattern) -> int {
             return static_cast<int>(DeriveMappingType(
                 std::string(op_name), static_cast<OpPatternKind>(pattern)));
           })
      .def("relax.analysis.DnnfusionFuseMappingCheck",
           [](int first, int second) -> ffi::Array<int64_t> {
             FuseDecision d = FuseMappingCheck(static_cast<MappingType>(first),
                                               static_cast<MappingType>(second));
             return ffi::Array<int64_t>{static_cast<int64_t>(d.color),
                                        static_cast<int64_t>(d.result)};
           })
      .def("relax.analysis.DnnfusionMappingTypeName",
           [](int t) -> ffi::String {
             return ffi::String(MappingTypeName(static_cast<MappingType>(t)));
           })
      .def("relax.analysis.DnnfusionFuseColorName",
           [](int c) -> ffi::String {
             return ffi::String(FuseColorName(static_cast<FuseColor>(c)));
           });
}

}  // namespace dnnfusion
}  // namespace relax
}  // namespace tvm
