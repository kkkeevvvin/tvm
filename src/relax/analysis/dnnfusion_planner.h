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
 * \file src/relax/analysis/dnnfusion_planner.h
 * \brief Helpers for the DNNFusion (Niu et al., PLDI '21) §4.3 fusion plan
 *        generator.  The Listing 1 traversal itself lives inside
 *        GraphPartitioner::RunFuseDnnfusion to keep union-find access
 *        local; this header only exposes the side-utilities that don't
 *        need to touch private partitioner state.
 */

#ifndef TVM_RELAX_ANALYSIS_DNNFUSION_PLANNER_H_
#define TVM_RELAX_ANALYSIS_DNNFUSION_PLANNER_H_

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <cstdint>
#include <string>
#include <unordered_map>

#include "dnnfusion_mapping.h"

namespace tvm {
namespace relax {
namespace dnnfusion {

/*!
 * \brief Yellow-cell handling policy for Table 3 fuse_depend cases.
 *
 * The paper §4.3.2 Step 2.3 calls for a profile-based oracle to decide
 * whether each yellow cell should fuse.  We don't have a profile DB on
 * dev/claude, so this enum picks one of three offline approximations.
 */
enum class YellowPolicy : int {
  // Treat all yellow cells as red (never fuse).  Safe; in practice gives
  // less fusion than TVM's baseline FuseOps because TVM's RunFuse already
  // accepts a few of these cells.
  kConservative = 0,
  // Treat all yellow cells as green (always fuse when legal).
  // Documented to break in three independent ways across vision +
  // transformer; useful only as stress test.
  kAggressive = 1,
  // Allow only the yellow cells that TVM's existing OpPatternKind cascade
  // also accepts, i.e. the subset where DNNFusion adds nothing risky on
  // top of the baseline:
  //   OtO  x MtM  : kElemWise -> kCommReduce / kOutEWiseFusable (safe)
  //   MtM  x OtO  is already Green in Table 3, no change
  //   MtM  x OtM  : kOutEWiseFusable -> kBroadcast (TVM's conv epilogue)
  //   OtM  x Reorg: kBroadcast -> kInjective (TVM allows broadcast into
  //                 injective epilogue)
  // All other yellow cells stay red.  Goal: reproduce TVM's fusion legality
  // while keeping DNNFusion's IRS-min OtO seed selection.
  kTvmCompat = 2,
  // Cost-proxy oracle approximating paper §4.3.2 Step 2.3 without a
  // profile DB.  Per-cell heuristics:
  //   MtM x OtM (conv -> broadcast):
  //       allow if broadcast factor IRS(consumer)/IRS(producer) <= 8
  //   All other yellow cells:
  //       deny (same as conservative)
  // This is meant to be the closest stand-in for the paper's profile-based
  // selection that we can ship without runtime measurements.  See
  // experiments/dnnfusion_port/design_yellow_oracle.md for rationale.
  kAuto = 3,
};

/*!
 * \brief Compile-time options for the DNNFusion plan generator.
 */
struct PlannerOptions {
  YellowPolicy yellow_policy = YellowPolicy::kConservative;

  // Used when yellow_policy == kAuto.  Allow MtM x OtM merges only when
  // the consumer's IRS / producer's IRS <= this factor.  Default 8 means
  // "broadcast at most 8x per element"; higher values pull more
  // broadcast-after-conv chains into the conv group at the cost of
  // potential register pressure.
  double mtm_otm_max_broadcast_factor = 8.0;

  // Convenience accessors that match the original boolean API.
  bool is_aggressive() const { return yellow_policy == YellowPolicy::kAggressive; }
  bool is_conservative() const { return yellow_policy == YellowPolicy::kConservative; }
  bool is_tvm_compat() const { return yellow_policy == YellowPolicy::kTvmCompat; }
  bool is_auto() const { return yellow_policy == YellowPolicy::kAuto; }
};

/*!
 * \brief Decide whether a specific yellow (producer, consumer) pair is
 * allowed under the kTvmCompat policy.  Returns true to allow (treat as
 * green), false to deny (treat as red).  Unknown / non-yellow inputs:
 * undefined behavior; only call after confirming the pair is yellow.
 */
bool TvmCompatAllowsYellow(MappingType producer, MappingType consumer);

/*!
 * \brief Walk an IRModule and collect a map from Var (binding lhs) to the
 * op name of the bound CallNode, if any.  Used by the planner to look up
 * MappingType via the explicit Table 2 op-name table.
 *
 * For non-CallNode bindings (TupleGetItem, Tuple, etc.) the map has no
 * entry and the planner falls back to OpPatternKind-based classification.
 *
 * For relax.call_tir / relax.call_tir_inplace bindings, the immediate op
 * is the call_tir wrapper; the wrapped PrimFunc's name is recorded under
 * the "tir." prefix (e.g. "tir.add" for a legalized relax.add).  This
 * gives the classifier a chance to recognize a legalized-but-still
 * paper-classified op via a "tir."-prefixed alias added in the future.
 */
std::unordered_map<const Object*, std::string> CollectVarToOpName(const IRModule& mod);

/*!
 * \brief Compute the IRS (intermediate result set) size in bytes for a
 * graph node, derived from the StructInfo of the bound Var or Constant.
 *
 * For tuple-typed values, the size is summed over all tensor fields.
 * Returns -1 if the StructInfo is unset, dynamic-shaped, or a non-tensor
 * type (Shape / Prim / Object) that has no meaningful byte count — the
 * caller should treat -1 as "sort to back of seed candidates".
 */
int64_t ComputeIrsBytes(const Object* ref);

/*!
 * \brief Resolve a graph node to a MappingType using the same fallback
 * chain as the standalone DeriveMappingType helper, but driven by the
 * (optional) side-map from CollectVarToOpName.
 *
 * \param ref     The graph node's owning Object* (Var or Constant).
 * \param pattern OpPatternKind from the partitioner's IndexedForwardGraph.
 * \param var_to_op_name Output of CollectVarToOpName for this IRModule.
 */
MappingType DeriveNodeMappingType(
    const Object* ref, OpPatternKind pattern,
    const std::unordered_map<const Object*, std::string>& var_to_op_name);

}  // namespace dnnfusion
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNFUSION_PLANNER_H_
