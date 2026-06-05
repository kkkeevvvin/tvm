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
 * \file src/relax/analysis/graph_partitioner.h
 * \brief The helper function for op fusion.
 */

#ifndef TVM_RELAX_ANALYSIS_GRAPH_PARTITIONER_H_
#define TVM_RELAX_ANALYSIS_GRAPH_PARTITIONER_H_

#include <tvm/ir/module.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/arena.h"

namespace tvm {
namespace relax {

using support::LinkedList;
using support::LinkNode;

/*!
 * \brief Compute the size in bytes of a value with the given struct info.
 *
 * Tensors contribute (product of static shape dims) * dtype bytes; tuples are
 * the sum of their fields. Returns -1 (the "unknown / dynamic / opaque"
 * sentinel) when any tensor has a non-static shape, a non-integer dim, or the
 * struct info is neither a tensor nor a tuple of computable fields.
 *
 * Defined in src/relax/transform/fuse_ops.cc; declared here so it is reachable
 * from unit tests and other analysis code.
 *
 * \param sinfo The struct info to measure.
 * \return The total byte size, or -1 if it cannot be determined statically.
 */
int64_t StructInfoBytes(const StructInfo& sinfo);

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
 * graph_partitioner translation unit alike.
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
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragment and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class IndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The DNNFusion Table 2 mapping type of the node, used by RunDNNFuse. */
    MappingType mapping_type{kMappingOpaque};
    /*! \brief Output size in bytes; -1 if unknown / dynamic / opaque sinfo. */
    int64_t output_size{-1};
    /*!
     * \brief Weak reference to the call_tir GlobalVar this node invokes; nullptr for
     *  non-call_tir nodes. Non-owning (like `ref`) so Node stays arena-safe -- the
     *  GlobalVarNode is owned by the IRModule, which outlives partitioning.
     */
    const GlobalVarNode* gvar{nullptr};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
    /*! \brief The inputs of the node.  */
    LinkedList<Edge> inputs;
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() const {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      std::string bytes_str =
          node->output_size < 0 ? "?" : std::to_string(node->output_size);
      os << "node[" << i << "], " << ffi::GetRef<ObjectRef>(node->ref);
      if (node->gvar != nullptr) {
        os << " gvar=" << node->gvar->name_hint;
      }
      os << " mapping=" << MappingTypeName(node->mapping_type)
         << " bytes=" << bytes_str << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << "\nIndexedForwardGraph: " << post_dfs_order.size() << " nodes\n"
              << os.str();
  }
};

/*!
 * \brief Find the One-to-One node with the smallest output among unfused ops.
 *
 * The RunDNNFuse seed selector (DNNFusion Listing 1 Step 1): considers only
 * nodes whose mapping_type is kOneToOne and whose output_size is known
 * (non-negative), and returns the one with the smallest output_size. Ties are
 * broken by the smaller node index to keep the selection deterministic.
 *
 * Defined in src/relax/analysis/graph_partitioner.cc; declared here so it is
 * reachable from unit tests.
 *
 * \param unfused_ops The set of candidate nodes that have not been fused yet.
 * \return The matching node with the minimum output size, or nullptr if no
 *         eligible One-to-One node exists.
 */
IndexedForwardGraph::Node* FindMinOtO(
    const std::unordered_set<IndexedForwardGraph::Node*>& unfused_ops);

/*!
 * \brief Dominator tree that represent domination or
 *  post domination relation of the node.
 */
class DominatorTree {
 public:
  /*!
   * \brief A node in the dominator tree.
   */
  struct Node {
    /*! \brief The node in the tree */
    IndexedForwardGraph::Node* gnode{nullptr};
    /*! \brief parent of the tree */
    Node* parent{nullptr};
    /*! \brief current depth*/
    int depth{0};
    /*! \brief aggregated pattern to parent */
    OpPatternKind pattern{kOpaque};
  };
  // index -> node.
  std::vector<Node*> nodes;
  /*!
   * \brief compute a post dominator relation for a given dataflow graph.
   * \param arena The arena used for node allocation.
   * \param graph The graph to be analyzed.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA (Least Common Ancestor)
   */
  static DominatorTree PostDom(support::Arena* arena, const IndexedForwardGraph& graph);

 private:
  // Combine pattern together.
  inline static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common ancestor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of the two.
   */
  static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern);
  /*!
   * \brief Find the least common ancestor of a list of nodes.
   * \param nodes the nodes.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of all nodes.
   */
  Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes,
                            OpPatternKind* edge_pattern);

  /*!
   * \brief Convert the Node from an IndexedForwardGraph Node into DomaintorTree Node.
   * \param arena The Arena.
   * \param gnode An IndexedForwardGraph Node.
   * \return The DominatorTree Node.
   */
  Node* GetNode(support::Arena* arena, IndexedForwardGraph::Node* gnode);
};

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class GraphPartitioner {
 public:
  explicit GraphPartitioner(IRModule mod, support::Arena* arena, int opt_level,
                            size_t max_fuse_depth, size_t max_function_args)
      : mod_(std::move(mod)),
        arena_(arena),
        opt_level_(opt_level),
        max_fuse_depth_(max_fuse_depth),
        max_function_args_(max_function_args) {}
  /*!
   * \brief Group as a union find data structure.
   */
  struct Group {
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief The DNNFusion Table 2 mapping type of the group, used by RunDNNFuse. */
    MappingType mapping_type;
    /*! \brief reference to the root node. */
    const tvm::Object* root_ref{nullptr};
    /*!
     * \brief Reference to the anchor node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Object* anchor_ref{nullptr};
    /*!
     * \brief The number of nodes belonging to this group
     */
    uint32_t num_nodes{1};
    /*!
     * \brief The number of function arguments belonging to this group
     */
    size_t args_num{0};

    /*! \brief Optional attributes to annotate the grouped function. */
    ffi::Map<ffi::String, Any> attrs;
    /*!
     * \brief Find the group root, perform path compression
     * \return The root type node.
     */
    Group* FindRoot();
  };
  /*!
   * \brief Partition a graph.
   * \return group assignments of each node.
   */
  std::vector<Group*> Partition(const IndexedForwardGraph& graph);

 private:
  /*! \brief The IRModule the graph being partitioned was created from. */
  IRModule mod_;
  /*! \brief The internal arena for temporary space. */
  support::Arena* arena_;
  /*! \brief optimization level for fuse operation. */
  int opt_level_;
  /*! \brief The maximum number of operations in one fused function */
  size_t max_fuse_depth_;
  /*! \brief The maximum number of arguments in one fused function */
  size_t max_function_args_;
  /*! \brief The internal groups. */
  std::vector<Group*> groups_;
  /*! \brief internal field used for deduplication */
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  /*! \brief The map with nodes which were postponed for fusing. */
  std::unordered_multimap<const IndexedForwardGraph::Node*, IndexedForwardGraph::Node*>
      postponed_fusing_map_;
  /*!
   * \brief Fusing of this node should be postponed till all child nodes are evaluated.
   *        It is used to calculate the number of arguments which will be passed to this node in
   *        the generated function.
   */
  const IndexedForwardGraph::Node* postpone_node_{nullptr};
  // Internal implementation of CheckPath
  template <typename F>
  bool CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond);

  /*!
   * \brief Check all the node and edge pattern
   *  between src and sink satisfies fcond.
   *
   * src is not checked.
   *
   * \param src The source node.
   * \param sink The termination node.
   * \param fcond The condition to be checked.
   * \tparam F the condition function, with signature
   * \note sink must be a post-dominator of src.
   */
  template <typename F>
  bool CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond);

  /*!
   * \brief Merge the child group to the parent.
   * \param child The child group.
   * \param parent The parent group.
   */
  void MergeFromTo(Group* child, Group* parent);

  // Internal implementation of CommitFuse
  void CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, Group* target);

  /*!
   * \brief Commit fusion operation.
   * \param src The source node.
   * \param sink The termination node.
   * \note sink must be a post-dominator of src.
   */
  void CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink);

  size_t CountNodesUptoSink_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink);
  // Calculate the number of arguments for the node.
  size_t CountArgs_(IndexedForwardGraph::Node* src, const IndexedForwardGraph& graph,
                    bool update_postpone = true);
  // Count the actual limit of arguments for a generated function.
  // max_function_args_ specifies the number of maximum function arguments. But
  // usually, output tensors are also passed to the function as arguments.
  // Additionally, in the case of dynamic shape, it is necessary to take into
  // account the number of parameters which specifies the sizes of the dynamic
  // dimensions.
  // This function computes the maximum number of arguments by the following formula:
  // limit = max_function_args_ - output_args_count
  size_t CountArgsLimit_(const IndexedForwardGraph::Node* child);

  // Count the number of nodes in a fused subgraph if child is additionally fused.
  // dom_parent is already known to be a part of the subgraph.
  // For a diamond structure, there can be multiple paths connecting child and dom_parent.
  // All intermediate nodes between child and dom_parent are taken into account.
  // Since dom_parent can itself be an intermediate node in the subgraph, calling FindRoot()
  // is important for correct calculation.
  size_t CountFusedNodesWithNewChild(IndexedForwardGraph::Node* child,
                                     IndexedForwardGraph::Node* dom_parent);
  // Count the number of arguments in a fused subgraph. This function also takes into account the
  // number of the child's output node argument. It helps to stop fusing before the node when the
  // limit will be exceeded.
  size_t CountFusedArgs(const IndexedForwardGraph& graph, IndexedForwardGraph::Node* child);

  // Initialize the groups.
  void InitGroups(const IndexedForwardGraph& graph);

  // execute the fusion algorithm.
  void RunFuse(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree, int phase);

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
   * Alternative fusion path selected when opt_level_ == 6 (driven from Python
   * via relax.transform.FuseOps(fuse_opt_level=6)), replacing the default
   * 3-phase RunFuse pipeline. Instead of the dominator-tree-based grouping, it
   * works directly off the cached pattern / mapping_type / output_size on each
   * IndexedForwardGraph::Node, seeding from the smallest One-to-One operator
   * and expanding into successors / predecessors (via DNNFuseRelation::Classify
   * on mapping_type).
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
   * The DNNFuse replacement for CommitFuse/MergeFromTo: the same union-find
   * bookkeeping (num_nodes / args_num / parent), plus the fused-type
   * derivation MergeFromTo never did. MergeFromTo's anchor_ref transfer and
   * CombinePattern maintenance are deliberately dropped -- after a DNNF
   * partition nothing reads Group::pattern/anchor_ref, and Table 3's red
   * cells (M2M x M2M) are the anchor-collision rule now.
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

  // Log the PrimFunc backing a node, resolved through node->gvar + mod_.
  void DumpNodePrimFunc(const char* label, const IndexedForwardGraph::Node* node);
  // GPU-schedule + build a node's PrimFunc on cuda and return the average
  // wall-clock latency (microseconds) over device executions; -1 on failure.
  double TimeNode(const IndexedForwardGraph::Node* node);
  // Build the merged PrimFunc fusing every node in `nodes` (FuseTIR over a
  // kPrimitive module) and time it on cuda; -1 if any op lacks a call_tir
  // binding or the fused kernel cannot be built. `nodes` must be in
  // producer-before-consumer order (sorted by index). The block counterpart of
  // TimeNode.
  double TimeFusedBlock(const std::vector<const IndexedForwardGraph::Node*>& nodes);
  // Profile the already-fused `block` and the `candidate` op, logging the
  // group-level fused-vs-separate comparison. Returns true iff merging the
  // candidate into the block (one kernel) is faster than the block kernel plus
  // the candidate run separately (false if anything could not be timed). `block`
  // excludes `candidate`. Used by FuseSuccessor / FusePredecessor to gate the
  // ambiguous kFuseDepend case.
  bool FuseProfit(const Block& block, IndexedForwardGraph::Node* candidate);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ANALYSIS_GRAPH_PARTITIONER_H_
