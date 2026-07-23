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
 * \file src/relax/analysis/dnnf_partitioner.h
 * \brief DNNFusion's (Niu et al., PLDI'21) mathematical-property-driven
 *        seed-and-expand fusion algorithm, split out of GraphPartitioner's
 *        classic post-dominator-tree algorithm (see graph_partitioner.h).
 */

#ifndef TVM_RELAX_ANALYSIS_DNNF_PARTITIONER_H_
#define TVM_RELAX_ANALYSIS_DNNF_PARTITIONER_H_

#include <tvm/ir/module.h>
#include <tvm/relax/op_attr_types.h>

#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/arena.h"
#include "./graph_partitioner.h"

namespace tvm {
namespace relax {

/*!
 * \brief Human-readable name of a DNNFusion Table 2 MappingType, for debug logs.
 * \param m The mapping type.
 * \return The enumerator name, or "kUnknown" for out-of-range values.
 */
inline const char* MappingTypeName(MappingType m) {
  switch (m) {
    case kOneToOne: return "kOneToOne";
    case kOneToMany: return "kOneToMany";
    case kManyToMany: return "kManyToMany";
    case kReorganize: return "kReorganize";
    case kShuffle: return "kShuffle";
    case kMappingOpaque: return "kMappingOpaque";
    default: return "kUnknown";
  }
}

/*!
 * \brief The DNNFusion-style fusion relation between a producer/consumer op pair.
 *
 * Classifies an ordered (src -> sink) MappingType pair -- DNNFusion (Niu et al.,
 * PLDI'21) Table 3, \S3.2 -- into one of three kinds that drive the bidirectional
 * RunDNNFuse merge:
 *  - kFuseBreak: the pair is illegal or known-unprofitable (Table 3's red cells).
 *    kMappingOpaque on either side always breaks, since it means the op fell
 *    outside DNNFusion's Table 2 classification.
 *  - kFuseThrough: the pair is legal and profitable with no further analysis
 *    needed (Table 3's green cells).
 *  - kFuseDepend: legal, but profitability requires profiling (Table 3's yellow
 *    cells); no decision on its own, defer to the surrounding planner.
 *
 * Defined inline here so the classifier is reachable from unit tests and the
 * dnnf_partitioner translation unit alike.
 */
class DNNFuseRelation {
 public:
  enum Kind { kFuseThrough, kFuseBreak, kFuseDepend };
  /*!
   * \brief Classify the fusion relation between a source and a sink mapping type.
   * \param src The MappingType of the source (producer) op.
   * \param sink The MappingType of the sink (consumer) op.
   * \return The classified relation.
   */
  static DNNFuseRelation Classify(MappingType src, MappingType sink) {
    if (src == kMappingOpaque || sink == kMappingOpaque) {
      return DNNFuseRelation(kFuseBreak, kMappingOpaque);
    }
    // A Table 3 cell carries both facets of the paper's encoding: the cell's
    // color (legality: green/yellow/red -> through/depend/break) and the
    // cell's value ("the mapping type of the operator after fusion", Table 3
    // caption). Red cells have no fused operator; they carry kMappingOpaque as
    // an unreachable placeholder -- IsBreak() rejects before anyone reads it.
    struct Cell {
      Kind kind;
      MappingType fused;
    };
    // DNNFusion Table 3: rows are the producer (first op), columns are the
    // consumer (second op), matching this function's (src, sink) order.
    // Indexed directly by MappingType's enum value (0..4); kMappingOpaque is
    // handled above and never reaches this table.
    static constexpr Cell kTable[5][5] = {
        // sink=O2O                  sink=O2M                   sink=M2M
        // sink=Reorg                sink=Shuffle
        /* src=O2O    */
        {{kFuseThrough, kOneToOne},
         {kFuseThrough, kOneToMany},
         {kFuseThrough, kManyToMany},
         {kFuseThrough, kReorganize},
         {kFuseThrough, kShuffle}},
        /* src=O2M    */
        {{kFuseThrough, kOneToMany},
         {kFuseDepend, kOneToMany},
         {kFuseBreak, kMappingOpaque},
         {kFuseDepend, kOneToMany},
         {kFuseDepend, kOneToMany}},
        /* src=M2M    */
        {{kFuseThrough, kManyToMany},
         {kFuseDepend, kManyToMany},
         {kFuseBreak, kMappingOpaque},
         {kFuseDepend, kManyToMany},
         {kFuseDepend, kManyToMany}},
        /* src=Reorg  */
        {{kFuseThrough, kReorganize},
         {kFuseDepend, kOneToMany},
         {kFuseDepend, kManyToMany},
         {kFuseThrough, kReorganize},
         {kFuseThrough, kReorganize}},
        /* src=Shuffle*/
        {{kFuseThrough, kShuffle},
         {kFuseDepend, kOneToMany},
         {kFuseDepend, kManyToMany},
         {kFuseThrough, kReorganize},
         {kFuseThrough, kShuffle}},
    };
    Cell cell = kTable[static_cast<int>(src)][static_cast<int>(sink)];
    return DNNFuseRelation(cell.kind, cell.fused);
  }

  bool IsThrough() const { return kind_ == kFuseThrough; }
  bool IsBreak() const { return kind_ == kFuseBreak; }
  bool IsDepend() const { return kind_ == kFuseDepend; }

  /*!
   * \brief Table 3 cell value: the mapping type of the operator after fusion.
   * \note Only meaningful when !IsBreak(); break cells hold kMappingOpaque.
   */
  MappingType FusedType() const { return fused_; }

  const char* Name() const {
    switch (kind_) {
      case kFuseThrough: return "fuse_through";
      case kFuseBreak: return "fuse_break";
      case kFuseDepend: return "fuse_depend";
    }
    return "unknown";
  }

 private:
  DNNFuseRelation(Kind kind, MappingType fused) : kind_(kind), fused_(fused) {}
  Kind kind_;
  MappingType fused_;
};

/*!
 * \brief Find the One-to-One node with the smallest output among unfused ops.
 *
 * The RunDNNFuse seed selector (DNNFusion Listing 1 Step 1): considers only
 * nodes whose mapping_type is kOneToOne and whose output_size is known
 * (non-negative), and returns the one with the smallest output_size. Ties are
 * broken by the smaller node index to keep the selection deterministic.
 *
 * Defined in src/relax/analysis/dnnf_partitioner.cc; declared here so it is
 * reachable from unit tests.
 *
 * \param unfused_ops The set of candidate nodes that have not been fused yet.
 * \return The matching node with the minimum output size, or nullptr if no
 *         eligible One-to-One node exists.
 */
IndexedForwardGraph::Node* FindMinOtO(
    const std::unordered_set<IndexedForwardGraph::Node*>& unfused_ops);

/*!
 * \brief Partition a graph using DNNFusion's (Niu et al., PLDI'21) seed-and-
 *        expand fusion algorithm: mathematical-property-driven fusion, an
 *        alternative to GraphPartitioner's post-dominator-tree algorithm.
 *
 * The dedicated entry point for relax.transform.DNNFuseOps, independent of
 * GraphPartitioner's classic Partition()/RunFuse() path. Reuses
 * GraphPartitioner::Group as its union-find node so the two algorithms'
 * output is interchangeable downstream (e.g. in OperatorFusor).
 */
class DNNFGraphPartitioner {
 public:
  explicit DNNFGraphPartitioner(IRModule mod, support::Arena* arena)
      : mod_(std::move(mod)), arena_(arena) {}

  /*!
   * \brief Partition a graph using the DNNFusion-style mathematical-property
   *        fusion algorithm: InitGroups followed by the seed-and-expand walk
   *        over mapping_type / DNNFuseRelation.
   * \param profile_tune_trials Number of profiling trials for the cost-oracle
   *        profiling of kFuseDepend candidates; 0 disables profiling (today's
   *        only wired value -- the profiling itself isn't implemented yet).
   * \return group assignments of each node.
   */
  std::vector<GraphPartitioner::Group*> Partition(const IndexedForwardGraph& graph,
                                                   int64_t profile_tune_trials = 0);

 private:
  /*! \brief The IRModule the graph being partitioned was created from. */
  IRModule mod_;
  /*! \brief The internal arena for temporary space. */
  support::Arena* arena_;
  /*! \brief The internal groups. */
  std::vector<GraphPartitioner::Group*> groups_;
  /*!
   * \brief Number of profiling trials for Partition's cost-oracle profiling
   *        of kFuseDepend candidates; 0 disables profiling. Set via the
   *        profile_tune_trials argument to Partition().
   */
  int64_t profile_tune_trials_{0};

  // Initialize the groups. Duplicated from GraphPartitioner::InitGroups
  // (graph_partitioner.cc) rather than shared via a free function -- see the
  // split's design notes for the tradeoff.
  void InitGroups(const IndexedForwardGraph& graph);

  // A fusion block: the set of nodes merged so far. Ordered by IFG index
  // (== post-DFS topological order) so iteration is always producer-before-
  // consumer, which is what TimeFusedBlock / FuseProfit need. The set dedups
  // on insert (a diamond in the IFG can reach the same node twice); membership
  // lookup is never used, so an ordered set fits the access pattern better than
  // an unordered_set that would have to be copied out and re-sorted per probe.
  struct NodeByIndex {
    bool operator()(const IndexedForwardGraph::Node* a,
                    const IndexedForwardGraph::Node* b) const {
      return a->index < b->index;
    }
  };
  using Block = std::set<IndexedForwardGraph::Node*, NodeByIndex>;

  /*!
   * \brief Execute the DNNFusion-based fusion algorithm.
   *
   * The body of Partition() (driven from Python via
   * relax.transform.DNNFuseOps()), an alternative to GraphPartitioner's
   * default 3-phase RunFuse pipeline. Instead of the dominator-tree-based
   * grouping, it works directly off the cached pattern / mapping_type /
   * output_size on each IndexedForwardGraph::Node, seeding from the smallest
   * One-to-One operator and expanding into successors / predecessors (via
   * DNNFuseRelation::Classify on mapping_type).
   *
   * \param graph The indexed forward graph to fuse over.
   */
  void RunDNNFuse(const IndexedForwardGraph& graph);
  /*!
   * \brief Check that fusing across the direct edge src -> sink keeps the
   *        group partition convex (its quotient graph acyclic).
   *
   * Merging the two groups is illegal when some src -> sink path runs through
   * nodes outside both groups: the merged group would then both feed and
   * consume the excluded path (e.g. a residual skip edge fused around its own
   * branch), and OperatorFuser cannot serialize such a group -- it emits the
   * fused call at the group's last binding while the excluded path still
   * reads a var whose binding moved inside, leaving the var undefined.
   * Implemented as a forward DFS from every edge that leaves src's group into
   * an outside node; reaching sink's group means such a path exists. The
   * reverse direction needs no check: the partition is acyclic before the
   * merge and the edge src -> sink exists, so a sink -> src path cannot.
   *
   * \param graph The indexed forward graph being partitioned.
   * \param src The source (producer) node of the candidate edge.
   * \param sink The sink (consumer) node of the candidate edge.
   * \return true if the merge keeps the partition convex (fusion is legal).
   */
  bool CheckEdgeConvexity(const IndexedForwardGraph& graph, IndexedForwardGraph::Node* src,
                          IndexedForwardGraph::Node* sink);
  /*!
   * \brief Merge src's group into sink's group across the direct edge src -> sink,
   *        and set the surviving root's mapping_type to the fused type derived
   *        from Table 3 (relation.FusedType()).
   *
   * The DNNFuse replacement for GraphPartitioner's CommitFuse/MergeFromTo: the
   * same union-find bookkeeping (num_nodes / args_num / parent), plus the
   * fused-type derivation MergeFromTo never did. MergeFromTo's anchor_ref
   * transfer and CombinePattern maintenance are deliberately dropped -- after
   * a DNNF partition nothing reads Group::pattern/anchor_ref, and Table 3's
   * red cells (M2M x M2M) are the anchor-collision rule now.
   *
   * \param src The source (producer) node.
   * \param sink The sink (consumer) node.
   * \param relation The Table 3 classification of (src, sink), already checked
   *        legal by the caller; legality is not re-checked here.
   * \note sink must be an immediate neighbour of src (DNNFuse only ever fuses
   *       across one edge at a time); enforced with an ICHECK.
   */
  void CommitFuseEdge(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                      const DNNFuseRelation& relation);
  /*!
   * \brief Recursively fuse forward (successor) neighbours into sp's group.
   *
   * Implements RunDNNFuse's forward expansion (DNNFusion Listing 1
   * Step 2.1-2.3), walking Node::outputs along the direct edge sp -> successor.
   * The successor is merged into sp's group based on DNNFuseRelation::Classify of
   * the two groups' root mapping types -- the legality gate:
   *   - kFuseBreak: reject the fusion outright.
   *   - kFuseDepend: profit-gated via FuseProfit; fuse only when the profiled
   *     fused kernel beats running the block and the candidate separately.
   *   - kFuseThrough: fuse, subject to the CheckEdgeConvexity structural gate
   *     (skip edges whose merge would leave a sp -> successor path outside the
   *     group; they may become fusable later once the path joins either side).
   * On success, CommitFuseEdge(sp, successor, relation) unions the two groups
   * across the edge and stamps the surviving root with Table 3's fused type,
   * the successor is added to block, and expansion recurses into its successors.
   *
   * \param graph The indexed forward graph being partitioned.
   * \param sp The seed node whose group is being extended.
   * \param successor The forward neighbour considered for fusion.
   * \param block The accumulating set of nodes fused into the seed's block.
   */
  void FuseSuccessor(const IndexedForwardGraph& graph, IndexedForwardGraph::Node* sp,
                     IndexedForwardGraph::Node* successor, Block* block);
  /*!
   * \brief Recursively fuse backward (predecessor) neighbours into sp's group.
   *
   * The backward mirror of FuseSuccessor.
   *
   * Implements RunDNNFuse's backward expansion (DNNFusion Listing 1 Step 2.1-2.3),
   * walking Node::inputs instead of Node::outputs, along the direct edge
   * predecessor -> sp.
   * The predecessor is merged into sp's group based on DNNFuseRelation::Classify of
   * the two groups' root mapping types -- the legality gate:
   *   - kFuseBreak: reject the fusion outright.
   *   - kFuseDepend: profit-gated via FuseProfit; fuse only when the profiled
   *     fused kernel beats running the block and the candidate separately.
   *   - kFuseThrough: fuse, subject to the CheckEdgeConvexity structural gate
   *     oriented along the direct edge predecessor -> sp.
   * On success, CommitFuseEdge(predecessor, sp, relation) unions the two groups
   * across the edge and stamps the surviving root with Table 3's fused type, the
   * predecessor is added to block, and expansion recurses into its predecessors.
   *
   * \param graph The indexed forward graph being partitioned.
   * \param sp The seed node whose group is being extended.
   * \param predecessor The backward neighbour considered for fusion.
   * \param block The accumulating set of nodes fused into the seed's block.
   */
  void FusePredecessor(const IndexedForwardGraph& graph, IndexedForwardGraph::Node* sp,
                       IndexedForwardGraph::Node* predecessor, Block* block);
  // Prototype hook for the ambiguous kFuseDepend case. Returns false until the
  // profiling oracle is implemented.
  bool FuseProfit(const Block& block, IndexedForwardGraph::Node* candidate);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ANALYSIS_DNNF_PARTITIONER_H_
