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
 * \file src/relax/transform/annotate_irs.cc
 * \brief Annotate `irs_output_bytes` (intermediate result size, in bytes) on
 *        each TIR PrimFunc. Designed to run after LegalizeOps, so each PrimFunc
 *        is the lowered kernel of one Relax operator. Downstream passes
 *        (e.g. DNNFusion-style seed selection) consume this attribute via
 *        `mod[gv]->attrs["irs_output_bytes"]`.
 *
 *        Sentinel: a value of -1 means the output buffer was not in the
 *        canonical DPS position OR the shape contained a symbolic dim OR the
 *        dtype size could not be determined.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace relax {

static constexpr const char* kIRSOutputBytes = "irs_output_bytes";

/*! \brief Best-effort: find the output buffer of a DPS-style PrimFunc.
 *
 * Convention: the trailing entries of `params` map to output buffers via
 * `buffer_map`. We pick the last param that has a buffer entry. If that
 * buffer's shape is fully static and the dtype has a known size, return
 * the byte count; otherwise return -1.
 */
static int64_t ComputeOutputBytes(const tir::PrimFunc& f) {
  if (f->params.empty()) return -1;

  tir::Buffer out_buf;
  for (auto it = f->params.rbegin(); it != f->params.rend(); ++it) {
    auto buf_it = f->buffer_map.find(*it);
    if (buf_it != f->buffer_map.end()) {
      out_buf = (*buf_it).second;
      break;
    }
  }
  if (!out_buf.defined()) return -1;

  int64_t numel = 1;
  for (const PrimExpr& dim : out_buf->shape) {
    if (auto* imm = dim.as<IntImmNode>()) {
      numel *= imm->value;
    } else {
      return -1;
    }
  }

  DataType dt = out_buf->dtype;
  int dtype_bytes = dt.bytes() * dt.lanes();
  if (dtype_bytes <= 0) return -1;

  return numel * static_cast<int64_t>(dtype_bytes);
}

tir::PrimFunc AnnotateIRSAttr(tir::PrimFunc f) {
  if (f->attrs.defined() && f->attrs->dict.count(kIRSOutputBytes)) {
    return f;
  }
  int64_t bytes = ComputeOutputBytes(f);
  return WithAttr(std::move(f), kIRSOutputBytes, bytes);
}

namespace transform {

Pass AnnotateIRS() {
  auto pass_func = [=](tir::PrimFunc f, IRModule m, PassContext ctx) {
    return AnnotateIRSAttr(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "AnnotateIRS", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AnnotateIRS", AnnotateIRS);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
