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
 * \file tvm/relax/dnnfusion_op_class.h
 * \brief DNNFusion (Niu et al., PLDI '21) operator classification.
 *
 * Five-class mapping-type taxonomy that runs in parallel to OpPatternKind.
 * The classification rides on Op-level attributes (name "TDnnfOpClass",
 * Integer value) for directly-named Relax ops, and on TIR PrimFunc
 * attributes (name "dnnf_op_class", Integer value) for call_tir targets.
 *
 * See docs/wiki/concepts/dnnfusion_taxonomy.md for the canonical taxonomy,
 * Table 2 per-op assignments, and Table 3 fusion result matrix.
 */
#ifndef TVM_RELAX_DNNFUSION_OP_CLASS_H_
#define TVM_RELAX_DNNFUSION_OP_CLASS_H_

#include <string>

namespace tvm {
namespace relax {

/*! \brief DNNFusion mapping-type class for an operator (paper Table 2). */
enum class DnnfOpClass : int {
  /*! \brief 1-to-1 index mapping with arithmetic op F. Add, Relu, Sigmoid... */
  kOneToOne = 0,
  /*! \brief Single input element fanned out to many outputs. Expand, Gather... */
  kOneToMany = 1,
  /*! \brief Cross-input aggregation + cross-axis traversal. Conv, GEMM, Softmax, Reduce... */
  kManyToMany = 2,
  /*! \brief Pure index reorganization, no F applied. Reshape, Squeeze, Flatten... */
  kReorganize = 3,
  /*! \brief Axis permutation. Transpose, DepthToSpace... */
  kShuffle = 4,
  /*! \brief Op not covered by paper Table 2 / cannot be classified. */
  kUnknown = 5,
};

/*!
 * \brief Op-level attribute key carrying DnnfOpClass for named Relax ops.
 *
 * Registered via TVM_REGISTER_OP("relax.X").set_attr<Integer>(kDnnfOpClassAttr, ...)
 * Read via Op::GetAttrMap<Integer>(kDnnfOpClassAttr).
 */
constexpr const char* kDnnfOpClassAttr = "TDnnfOpClass";

/*!
 * \brief PrimFunc-level attribute key carrying DnnfOpClass for call_tir targets.
 *
 * Written by AnnotateDnnfusionClass pass; read by ClassifyDnnfOp consumers.
 */
constexpr const char* kDnnfOpClassPrimFuncAttr = "dnnf_op_class";

/*!
 * \brief Provenance of a DnnfClassification result.
 *
 * Lets downstream consumers (fusion policy, profiler, debugger) tell whether
 * a class came from the paper Table 2 (high-confidence, exact match) or from
 * the OpPatternKind heuristic fallback (best-effort).
 */
enum class DnnfClassSource : int {
  /*! \brief Hit on Op::GetAttrMap<Integer>("TDnnfOpClass") — paper Table 2. */
  kAttrTable = 0,
  /*! \brief Read from a PrimFunc's "dnnf_op_class" attribute (annotation pass). */
  kTirAnnotation = 1,
  /*! \brief Derived via OpPatternKind heuristic — fallback. */
  kPatternHeuristic = 2,
  /*! \brief No information available — kUnknown. */
  kFallback = 3,
};

/*! \brief Result of classifying a single Relax expression. */
struct DnnfClassification {
  DnnfOpClass cls{DnnfOpClass::kUnknown};
  DnnfClassSource source{DnnfClassSource::kFallback};
  std::string reason;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DNNFUSION_OP_CLASS_H_
