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
 * \file src/relax/analysis/dnnfusion_mapping.h
 * \brief DNNFusion (Niu et al., PLDI '21) operator mapping types and
 *        Table 3 fusion legality matrix.
 *
 * This module is the classifier half of the DNNFusion port.  It does NOT
 * decide fusion — it only answers "what kind of operator is this" and "if
 * I fuse a left-of-type-A op with a right-of-type-B op, is it legal /
 * profitable / requires-profile?".  The fusion plan generator that calls
 * into these helpers lives in dnnfusion_planner.{h,cc}.
 */

#ifndef TVM_RELAX_ANALYSIS_DNNFUSION_MAPPING_H_
#define TVM_RELAX_ANALYSIS_DNNFUSION_MAPPING_H_

#include <tvm/relax/op_attr_types.h>

#include <string>

namespace tvm {
namespace relax {
namespace dnnfusion {

/*!
 * \brief Five-way operator classification per DNNFusion §3.1, Table 2.
 *
 * Defined by the index relationship between input and output elements:
 *   OneToOne   : y[d] = F(x[f(d)]) with bijective f                 (Add, Relu, ...)
 *   OneToMany  : single input element fans out to many outputs       (Expand, Gather, ...)
 *   ManyToMany : multi-input reduces/contracts to multi-output       (Conv, GEMM, Reduce, ...)
 *   Reorganize : 1-1 index permutation that preserves access locality (Reshape, Squeeze, ...)
 *   Shuffle    : 1-1 index permutation that breaks locality          (Transpose, ...)
 *
 * Note: Many-to-One in the paper is folded into ManyToMany (footnote: "we
 * do not consider it separately here").  Unknown is reserved for opaque
 * nodes (tuple, var-only, unrecognized opaque ops); it is treated as a
 * fusion barrier (red) in Table 3 lookups.
 */
enum class MappingType : int {
  kUnknown = 0,
  kOneToOne = 1,
  kOneToMany = 2,
  kManyToMany = 3,
  kReorganize = 4,
  kShuffle = 5,
};

/*!
 * \brief Legality / profitability of a fusion candidate, per DNNFusion §3.2.
 *
 *   kRed    : illegal or known-unprofitable; do not fuse.
 *   kYellow : legal; profitability requires profile lookup (§4.3.2).
 *   kGreen  : legal and profitable; fuse without further check.
 */
enum class FuseColor : int {
  kRed = 0,
  kYellow = 1,
  kGreen = 2,
};

/*!
 * \brief Result of a Table 3 lookup.
 * \param color  Whether this fusion is legal/profitable.
 * \param result MappingType of the fused op (only meaningful when color != kRed).
 */
struct FuseDecision {
  FuseColor color;
  MappingType result;
};

/*!
 * \brief Classify a relax operator by its registered name.
 *
 * Resolution order:
 *   1. Explicit op-name table (Table 2 from the paper, mapped to relax names).
 *   2. Fallback heuristic from OpPatternKind:
 *        kElemWise        -> OneToOne
 *        kBroadcast       -> OneToMany
 *        kInjective       -> Reorganize
 *        kCommReduce      -> ManyToMany
 *        kOutEWiseFusable -> ManyToMany
 *        kTuple / kOpaque -> Unknown
 *
 * \param op_name Registered op name, e.g. "relax.add", "relax.nn.conv2d".
 *                Empty string forces fallback path.
 * \param pattern OpPatternKind of the same op (used only when op_name has
 *                no explicit entry).
 */
MappingType DeriveMappingType(const std::string& op_name, OpPatternKind pattern);

/*!
 * \brief Look up DNNFusion Table 3 for a candidate (first → second) fusion.
 *
 * \param first  MappingType of the op being absorbed (or the seed).
 * \param second MappingType of its successor / predecessor.
 * \return Color + resulting fused MappingType.  Unknown on either side is
 *         always kRed.
 */
FuseDecision FuseMappingCheck(MappingType first, MappingType second);

/*! \brief Printable name for debugging / tracing. */
const char* MappingTypeName(MappingType t);

/*! \brief Printable name for debugging / tracing. */
const char* FuseColorName(FuseColor c);

}  // namespace dnnfusion
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNFUSION_MAPPING_H_
