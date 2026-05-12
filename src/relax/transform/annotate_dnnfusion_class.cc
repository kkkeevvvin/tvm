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
 * \file src/relax/transform/annotate_dnnfusion_class.cc
 * \brief PrimFuncPass that annotates each TIR PrimFunc with a "dnnf_op_class"
 *        Integer attribute, derived from the existing "op_pattern" attribute
 *        via the DNNFusion fallback heuristic.
 *
 * Hard ordering dependency: AnnotateTIROpPattern must run first so the
 * "op_pattern" attribute is populated. See the implementation plan
 * (MLC-oto/tvm/oto/01_op_classification_plan.md §5.3) and the wiki concept
 * page (docs/wiki/concepts/dnnfusion_taxonomy.md) for design rationale.
 *
 * The pass is opt-in — it is not added to any default Relax pipeline.
 * Drivers must invoke it explicitly:
 *   LegalizeOps -> AnnotateTIROpPattern -> AnnotateDnnfusionClass.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dnnfusion_op_class.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace relax {

tir::PrimFunc AnnotateDnnfusionClassImpl(tir::PrimFunc f) {
  // Skip-if-already-set: use GetAttr(...).defined() rather than
  // HasNonzeroAttr because kOneToOne == 0; HasNonzeroAttr would falsely
  // classify a user's pre-set kOneToOne as "missing" and overwrite it.
  if (f->attrs.GetAttr<Integer>(kDnnfOpClassPrimFuncAttr).defined()) {
    return f;
  }
  DnnfOpClass cls = InferDnnfClassFromTir(f);
  return WithAttr(std::move(f), kDnnfOpClassPrimFuncAttr,
                  Integer(static_cast<int>(cls)));
}

namespace transform {

Pass AnnotateDnnfusionClass() {
  auto pass_func = [=](tir::PrimFunc f, IRModule m, PassContext ctx) {
    return AnnotateDnnfusionClassImpl(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(
      pass_func, /*opt_level=*/0, "AnnotateDnnfusionClass", /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AnnotateDnnfusionClass",
                        AnnotateDnnfusionClass);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
