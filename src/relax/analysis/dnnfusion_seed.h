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
 * \file src/relax/analysis/dnnfusion_seed.h
 * \brief DNNFusion-style seed selection helpers for the Relax FuseOps
 *        partitioner outer loop.
 *
 * This is the exp2 narrow drop-in: only the partitioner's iteration order is
 * touched. The fusion decision (post-dominator + CheckPath + OpPatternKind
 * cascade) is unchanged. See exp2/plan.md for the scope contract.
 *
 * Mapping-type / IRS implementations are ported from exp1 phs1
 * (src/relax/transform/fuse_ops_dnnfusion.cc), restricted to what the
 * seed-ordering logic needs: no 5x5 mapping matrix, no Phase1AllowFusion.
 */
#ifndef TVM_RELAX_ANALYSIS_DNNFUSION_SEED_H_
#define TVM_RELAX_ANALYSIS_DNNFUSION_SEED_H_

#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>

#include <cstdint>
#include <string>
#include <vector>

#include "./graph_partitioner.h"

namespace tvm {
namespace relax {

/*! \brief DNNFusion paper's mapping types (Table 3). */
enum class MappingType : int {
  kOtO = 0,      // One-to-One   (relu, sigmoid, scalar add, ...)
  kOtM = 1,      // One-to-Many  (broadcast / expand)
  kMtM = 2,      // Many-to-Many (conv, matmul, reduce, pooling)
  kReorg = 3,    // Reorganize   (transpose, reshape, concat, ...)
  kShuffle = 4,  // Data-dependent indexing (gather, scatter, embedding, take)
  kBreak = 5,    // Cannot fuse / unknown
};

const char* MappingTypeToString(MappingType m);

/*!
 * \brief Substring whitelist for op names that should be classified as Shuffle.
 *        After LegalizeOps the high-level relax op identity is encoded in the
 *        lowered TIR PrimFunc's name (e.g. "take", "gather_nd").
 */
bool IsShuffleByCalleeName(const std::string& name);

/*!
 * \brief Derive the DNNFusion mapping type from TVM's existing OpPatternKind.
 *        \p callee_name is consulted only to disambiguate kOpaque -> Shuffle.
 *        Pass an empty string when the callee name is unknown; in that case
 *        kOpaque collapses to kBreak.
 */
MappingType DeriveMappingType(OpPatternKind pat, const std::string& callee_name);

/*!
 * \brief Compute IRS (intermediate-result size) in bytes for a StructInfo.
 *        Returns INT64_MAX for dynamic shape / unknown dtype / overflow so the
 *        owning binding sorts to the back of the seed-selection min-heap.
 *        For TupleStructInfo, sums fields; any dynamic field poisons the whole
 *        tuple to INT64_MAX.
 */
int64_t ComputeIRSBytes(const StructInfo& sinfo);

/*!
 * \brief Compute the partitioner outer-loop iteration order under the
 *        DNNFusion "OtO seeds first, smallest IRS first" policy.
 *
 *        Returns indices into \p graph.post_dfs_order, sorted by the key
 *        (is_OtO_seed_cand DESC, irs_bytes ASC, original_nid ASC). The third
 *        component preserves topological order as the final tiebreak so the
 *        ordering is fully deterministic and stable across runs.
 *
 *        is_OtO_seed_cand is true iff DeriveMappingType yields kOtO and the
 *        node's IRS is statically known (i.e. != INT64_MAX). A kOtO node with
 *        unknown IRS is *not* a seed candidate -- it sorts after all true
 *        seeds, with the topological tiebreak.
 *
 *        The classification uses only the per-node OpPatternKind (already set
 *        by GraphCreator); callee names are not threaded through here, so
 *        kOpaque -> kBreak. This is intentional: the seed-selection ranking
 *        only cares about who is OtO, and Shuffle ops are not seeds anyway.
 */
std::vector<size_t> ComputeSeedOrder(const IndexedForwardGraph& graph);

// ---------------------------------------------------------------------------
// FuseOps trace (exp2 Step 4)
// ---------------------------------------------------------------------------

/*!
 * \brief One row of the FuseOps fusion trace, recorded each time CommitFuse()
 *        runs to completion. Used to diff the topological vs dnnfusion_seed
 *        iteration orders against the same IRModule.
 */
struct FuseTraceRow {
  int phase;
  size_t src_nid;
  size_t sink_nid;
  uint32_t group_size_after;
};

/*! \brief Clear the per-thread FuseOps trace. */
void ResetFuseTrace();

/*! \brief Push a row into the per-thread FuseOps trace. */
void RecordFuseTrace(int phase, size_t src_nid, size_t sink_nid, uint32_t group_size_after);

/*! \brief Read out the per-thread FuseOps trace (as captured so far). */
const std::vector<FuseTraceRow>& GetFuseTrace();

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ANALYSIS_DNNFUSION_SEED_H_
