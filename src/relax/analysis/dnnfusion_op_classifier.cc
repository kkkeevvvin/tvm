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
 * \file src/relax/analysis/dnnfusion_op_classifier.cc
 * \brief DNNFusion op classifier (paper Table 2, see
 *        docs/wiki/concepts/dnnfusion_taxonomy.md).
 *
 * Lookup priority (first hit wins):
 *   1. Op::GetAttrMap<Integer>("TDnnfOpClass") on directly-named Relax ops
 *   2. PrimFunc attr "dnnf_op_class" on call_tir target functions
 *   3. OpPatternKind heuristic on PrimFunc "op_pattern" attribute
 *   4. kUnknown
 */
#include "./dnnfusion_op_classifier.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

namespace {

DnnfClassification Make(DnnfOpClass cls, DnnfClassSource src, std::string reason) {
  return DnnfClassification{cls, src, std::move(reason)};
}

}  // namespace

std::string DnnfOpClassToString(DnnfOpClass cls) {
  switch (cls) {
    case DnnfOpClass::kOneToOne: return "OneToOne";
    case DnnfOpClass::kOneToMany: return "OneToMany";
    case DnnfOpClass::kManyToMany: return "ManyToMany";
    case DnnfOpClass::kReorganize: return "Reorganize";
    case DnnfOpClass::kShuffle: return "Shuffle";
    case DnnfOpClass::kUnknown: return "Unknown";
  }
  return "Unknown";
}

std::string DnnfClassSourceToString(DnnfClassSource src) {
  switch (src) {
    case DnnfClassSource::kAttrTable: return "AttrTable";
    case DnnfClassSource::kTirAnnotation: return "TirAnnotation";
    case DnnfClassSource::kPatternHeuristic: return "PatternHeuristic";
    case DnnfClassSource::kFallback: return "Fallback";
  }
  return "Fallback";
}

DnnfOpClass DnnfClassFromOpPattern(OpPatternKind pattern) {
  // Heuristic fallback table — used only when the per-op TDnnfOpClass and
  // the PrimFunc "dnnf_op_class" annotation both miss. See
  // docs/wiki/concepts/dnnfusion_taxonomy.md §"與 TVM v0.23.0 OpPatternKind
  // 的對應".
  //
  // kBroadcast → kOneToMany (NOT kOneToOne). TVM's own definition of
  // kBroadcast — "out[i, ax1, j, ax2] = input[i, j]" — is structurally
  // a One-to-Many fan-out. A same-shape elementwise op tagged kBroadcast
  // is genuinely OtO, but we do not have shape info here; the safer
  // default for fusion-legality decisions is OtM (same-shape ops should
  // be precisely covered by the per-op TDnnfOpClass attribute, which
  // takes priority over this heuristic).
  switch (pattern) {
    case kElemWise:          return DnnfOpClass::kOneToOne;
    case kBroadcast:         return DnnfOpClass::kOneToMany;
    case kInjective:         return DnnfOpClass::kReorganize;
    case kCommReduce:        return DnnfOpClass::kManyToMany;
    case kOutEWiseFusable:   return DnnfOpClass::kManyToMany;
    case kTuple:             return DnnfOpClass::kUnknown;
    case kOpaque:            return DnnfOpClass::kUnknown;
  }
  return DnnfOpClass::kUnknown;
}

DnnfOpClass InferDnnfClassFromTir(const tir::PrimFunc& func) {
  // Single responsibility: read the op_pattern attribute (assumed populated
  // by AnnotateTIROpPattern) and run the heuristic. Any per-op precision
  // belongs on the Relax Op via Op::GetAttrMap<Integer>("TDnnfOpClass"),
  // not here — see plan §5.3 ("Single responsibility — heuristic only").
  if (auto p = func->attrs.GetAttr<Integer>("op_pattern")) {
    int v = p.value()->value;
    return DnnfClassFromOpPattern(static_cast<OpPatternKind>(v));
  }
  return DnnfOpClass::kUnknown;
}

DnnfClassification ClassifyDnnfOp(const Call& call) {
  static const auto& attr_map = Op::GetAttrMap<Integer>(kDnnfOpClassAttr);

  if (const auto* op_node = call->op.as<OpNode>()) {
    // call_tir: classifier alone cannot resolve the PrimFunc (no IRModule
    // in scope). Caller (fusion pass) should resolve the GlobalVar and
    // call InferDnnfClassFromTir, or read "dnnf_op_class" off the PrimFunc.
    if (op_node->name == "relax.call_tir") {
      return Make(DnnfOpClass::kUnknown, DnnfClassSource::kFallback,
                  "call_tir target — caller must resolve PrimFunc and use "
                  "InferDnnfClassFromTir or read \"dnnf_op_class\" attr");
    }
    // Direct Relax op — paper Table 2 hit when registered.
    Op op = ffi::GetRef<Op>(op_node);
    if (attr_map.count(op)) {
      int v = attr_map[op]->value;
      return Make(static_cast<DnnfOpClass>(v), DnnfClassSource::kAttrTable,
                  "registered TDnnfOpClass on " + op_node->name);
    }
    return Make(DnnfOpClass::kUnknown, DnnfClassSource::kFallback,
                "named Relax op without TDnnfOpClass: " + op_node->name);
  }

  // Non-Op calls: closures, packed funcs, etc.
  return Make(DnnfOpClass::kUnknown, DnnfClassSource::kFallback,
              "non-Op call expression");
}

// FFI globals for Python access.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.ClassifyDnnfOp",
           [](const Call& call) -> ffi::Array<ffi::Any> {
             DnnfClassification r = ClassifyDnnfOp(call);
             return {Integer(static_cast<int>(r.cls)),
                     Integer(static_cast<int>(r.source)),
                     ffi::String(r.reason)};
           })
      .def("relax.analysis.InferDnnfClassFromTir",
           [](const tir::PrimFunc& f) -> Integer {
             return Integer(static_cast<int>(InferDnnfClassFromTir(f)));
           })
      .def("relax.analysis.DnnfClassFromOpPattern",
           [](int p) -> Integer {
             return Integer(static_cast<int>(
                 DnnfClassFromOpPattern(static_cast<OpPatternKind>(p))));
           });
}

}  // namespace relax
}  // namespace tvm
