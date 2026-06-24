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

#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../support/arena.h"

namespace tvm {
namespace relax {

using support::LinkedList;
using support::LinkNode;

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
    /*! \brief Output size in bytes; -1 if unknown / dynamic / opaque sinfo. */
    int64_t output_size{-1};
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
    auto pattern_name = [](OpPatternKind p) -> const char* {
      switch (p) {
        case kElemWise: return "kElemWise";
        case kBroadcast: return "kBroadcast";
        case kInjective: return "kInjective";
        case kCommReduce: return "kCommReduce";
        case kOutEWiseFusable: return "kOutEWiseFusable";
        case kTuple: return "kTuple";
        case kOpaque: return "kOpaque";
        default: return "kUnknown";
      }
    };
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      std::string bytes_str =
          node->output_size < 0 ? "?" : std::to_string(node->output_size);
      os << "node[" << i << "], " << ffi::GetRef<ObjectRef>(node->ref)
         << " pattern=" << pattern_name(node->pattern)
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
  explicit GraphPartitioner(support::Arena* arena, int opt_level, size_t max_fuse_depth,
                            size_t max_function_args)
      : arena_(arena),
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
  /*! \brief The graph currently being processed by RunDNNFuse (for convexity queries). */
  const IndexedForwardGraph* dnnf_graph_{nullptr};
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

  /*!
   * \brief Execute the DNNFusion-based fusion algorithm.
   *
   * Alternative fusion path selected when opt_level_ == 6 (driven from Python
   * via relax.transform.FuseOps(fuse_opt_level=6)), replacing the default
   * 3-phase RunFuse pipeline. Instead of the dominator-tree-based grouping, it
   * works directly off the cached pattern / output_size on each
   * IndexedForwardGraph::Node, seeding from element-wise ops and expanding into
   * successors / predecessors.
   *
   * \param graph The indexed forward graph to fuse over.
   */
  void RunDNNFuse(const IndexedForwardGraph& graph);
  /*!
   * \brief Test whether adding \p extra to the group rooted at \p root keeps it convex.
   *
   * The DNNFuse path commits fusions as direct pairwise group unions instead of the
   * post-dominator-based CommitFuse, so it must check convexity explicitly: a group is
   * codegen-valid only if no node outside it lies on a path between two of its nodes.
   * Reachability is evaluated over dnnf_graph_ (set by RunDNNFuse).
   *
   * \param extra The node proposed to join the group.
   * \param root The current root of the seed's group.
   * \return true if the merged node set has no external bypass node, false otherwise.
   */
  bool KeepsBlockConvex(IndexedForwardGraph::Node* extra, Group* root);
  /*!
   * \brief Recursively fuse forward (successor) neighbours into sp's group.
   *
   * Implements RunDNNFuse's forward expansion (DNNFusion Listing 1
   * Step 2.1-2.3), walking Node::outputs along the data path sp -> successor.
   * The successor is merged into sp's group based on DNNFuseRelation::Classify of
   * the two groups' root patterns:
   *   - kFuseBreak: reject the fusion outright.
   *   - kFuseDepend: profit-gated; bail out until the profiler is implemented.
   *   - kFuseThrough: fuse.
   * On success, sp's group is unioned directly into successor's group (a pairwise
   * MergeFromTo, no post-dominator requirement), the successor is added to block,
   * and expansion recurses into its successors.
   *
   * \param sp The seed node whose group is being extended.
   * \param successor The forward neighbour considered for fusion.
   * \param block The accumulating set of nodes fused into the seed's block.
   */
  void FuseSuccessor(IndexedForwardGraph::Node* sp, IndexedForwardGraph::Node* successor,
                     std::unordered_set<IndexedForwardGraph::Node*>* block);
  /*!
   * \brief Recursively fuse backward (predecessor) neighbours into sp's group.
   *
   * The backward mirror of FuseSuccessor. 
   *
   * Implements RunDNNFuse's backward expansion (DNNFusion Listing 1 Step 2.1-2.3),
   * walking Node::inputs instead of Node::outputs, along the data path 
   * predecessor -> sp.
   * The predecessor is merged into sp's group based on DNNFuseRelation::Classify of
   * the two groups' root patternsx:
   *   - kFuseBreak: reject the fusion outright.
   *   - kFuseDepend: profit-gated; bail out until the profiler is implemented.
   *   - kFuseThrough: fuse.
   * On success, predecessor's group is unioned directly into sp's group (a pairwise
   * MergeFromTo, no post-dominator requirement), the predecessor is added to block,
   * and expansion recurses into its predecessors.
   *
   * \param sp The seed node whose group is being extended.
   * \param predecessor The backward neighbour considered for fusion.
   * \param block The accumulating set of nodes fused into the seed's block.
   */
  void FusePredecessor(IndexedForwardGraph::Node* sp, IndexedForwardGraph::Node* predecessor,
                       std::unordered_set<IndexedForwardGraph::Node*>* block);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ANALYSIS_GRAPH_PARTITIONER_H_
