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

#include "./graph_partitioner.h"

#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

IndexedForwardGraph::Node* FindMinOtO(
    const std::unordered_set<IndexedForwardGraph::Node*>& unfused_ops) {
  IndexedForwardGraph::Node* min_node = nullptr;
  for (IndexedForwardGraph::Node* node : unfused_ops) {
    if (node->mapping_type != kOneToOne) continue;
    if (node->output_size < 0) continue;
    if (min_node == nullptr || node->output_size < min_node->output_size ||
        (node->output_size == min_node->output_size && node->index < min_node->index)) {
      min_node = node;
    }
  }
  return min_node;
}

DominatorTree DominatorTree::PostDom(support::Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(Node* lhs, Node* rhs,
                                                        OpPatternKind* edge_pattern) {
  while (lhs != rhs) {
    if (lhs == nullptr) return nullptr;
    if (rhs == nullptr) return nullptr;
    if (lhs->depth < rhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      rhs = rhs->parent;
    } else if (rhs->depth < lhs->depth) {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      lhs = lhs->parent;
    } else {
      edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
      edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

DominatorTree::Node* DominatorTree::LeastCommonAncestor(
    const LinkedList<IndexedForwardGraph::Edge>& input_nodes, OpPatternKind* edge_pattern) {
  auto link = input_nodes.head;
  if (link == nullptr) {
    return nullptr;
  }
  auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
    size_t oindex = edge.node->index;
    ICHECK_LT(oindex, nodes.size());
    Node* onode = nodes[oindex];
    ICHECK(onode != nullptr);
    return onode;
  };
  Node* parent = get_node(link->value);
  *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  link = link->next;
  for (; link != nullptr; link = link->next) {
    parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
  }
  return parent;
}

DominatorTree::Node* DominatorTree::GetNode(support::Arena* arena,
                                            IndexedForwardGraph::Node* gnode) {
  Node* tnode = arena->make<Node>();
  tnode->gnode = gnode;
  if (gnode->extern_ref) {
    tnode->depth = 1;
    tnode->parent = nullptr;
    tnode->pattern = kOpaque;
  } else {
    // find the LCAs of all outputs.
    OpPatternKind pattern = kElemWise;
    Node* parent = LeastCommonAncestor(gnode->outputs, &pattern);
    tnode->depth = parent ? parent->depth + 1 : 1;
    tnode->parent = parent;
    tnode->pattern = pattern;
  }
  return tnode;
}

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  if (opt_level_ == 6) {
    this->RunDNNFuse(graph);
    return std::move(groups_);
  }
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);
  // run fusion algorithm.
  for (int phase = 0; phase < 3; ++phase) {
    this->RunFuse(graph, post_dom_tree, phase);
  }
  return std::move(groups_);
}

GraphPartitioner::Group* GraphPartitioner::Group::FindRoot() {
  // fast path
  if (this->parent == nullptr) return this;
  // slow path with path compression.
  Group* root = this;
  while (root->parent != nullptr) {
    root = root->parent;
  }
  for (Group* p = this; p != root;) {
    Group* parent = p->parent;
    p->parent = root;
    p = parent;
  }
  return root;
}

template <typename F>
bool GraphPartitioner::CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                  F fcond) {
  if (visited_.count(src)) return true;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  gnode = gnode->FindRoot();
  if (!fcond(gnode->pattern, src == sink)) return false;
  if (src == sink) return true;
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    if (!CheckPath_(link->value.node, sink, fcond)) return false;
  }
  return true;
}

template <typename F>
bool GraphPartitioner::CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                 F fcond) {
  ICHECK(!src->extern_ref);
  visited_.clear();
  ICHECK(src != sink);
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    if (!CheckPath_(link->value.node, sink, fcond)) return false;
  }
  return true;
}

OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
  if (lhs > kBroadcast && rhs > kBroadcast) {
    LOG(FATAL) << "Cannot merge two complex group together";
  }
  if (lhs > rhs) return lhs;
  return rhs;
}

void GraphPartitioner::MergeFromTo(Group* child, Group* parent) {
  child = child->FindRoot();
  parent = parent->FindRoot();
  if (child == parent) return;
  // update the number of nodes of the parent group
  parent->num_nodes += child->num_nodes;
  parent->args_num += child->args_num;
  child->parent = parent;
  // update anchor ref and pattern
  if (child->anchor_ref != nullptr) {
    ICHECK(parent->anchor_ref == nullptr);
    parent->anchor_ref = child->anchor_ref;
    parent->pattern = CombinePattern(child->pattern, parent->pattern);
  }
}

void GraphPartitioner::CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                                   Group* target) {
  if (postpone_node_ != nullptr) {
    postponed_fusing_map_.insert({postpone_node_, src});
    return;
  }
  if (src == sink) return;
  if (visited_.count(src)) return;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  // merge the current group to the parent if possible.
  MergeFromTo(gnode, target);
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    CommitFuse_(link->value.node, sink, target);
  }
}

void GraphPartitioner::CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
  Group* target = groups_[sink->index];
  visited_.clear();
  ICHECK(src != sink);
  CommitFuse_(src, sink, target);
}

size_t GraphPartitioner::CountNodesUptoSink_(IndexedForwardGraph::Node* src,
                                             IndexedForwardGraph::Node* sink) {
  if (src == sink || visited_.count(src)) return 0;
  visited_.insert(src);
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  auto sum = gnode->num_nodes;
  for (auto link = src->outputs.head; link != nullptr; link = link->next) {
    sum += CountNodesUptoSink_(link->value.node, sink);
  }
  return sum;
}

size_t GraphPartitioner::CountFusedNodesWithNewChild(IndexedForwardGraph::Node* child,
                                                     IndexedForwardGraph::Node* dom_parent) {
  Group* target = groups_[dom_parent->index];
  visited_.clear();
  ICHECK(child != dom_parent);
  return target->FindRoot()->num_nodes + CountNodesUptoSink_(child, dom_parent);
}

size_t GraphPartitioner::CountArgs_(IndexedForwardGraph::Node* src,
                                    const IndexedForwardGraph& graph, bool update_postpone) {
  std::unordered_set<Group*> visited_groups;
  Group* gnode = groups_[src->index];
  ICHECK(gnode != nullptr);
  auto sum = gnode->args_num;
  visited_groups.insert(gnode->FindRoot());
  auto calc_args_number = [this, src, &graph, &visited_groups,
                           update_postpone](const Expr& arg) -> size_t {
    if (arg.as<VarNode>()) return 0;
    auto* node = graph.node_map.at(arg.get());
    Group* prev_group = groups_[node->index]->FindRoot();
    if (visited_groups.count(prev_group) == 0) {
      visited_groups.insert(prev_group);
      if (prev_group->args_num > 0) {
        // Get the number of arguments from the group
        return prev_group->args_num;
      } else if (update_postpone) {
        // Update pointer to the node which should be postponed for deferred fusing
        postpone_node_ = src;
      } else {
        // Calculate the number of arguments for the node which wasn't processed before
        return CountArgs_(node, graph, update_postpone);
      }
    }
    return 0;
  };
  if (auto call_node = ffi::GetRef<ObjectRef>(src->ref).as<CallNode>()) {
    for (auto& it : call_node->args) {
      sum += calc_args_number(it);
    }
  } else if (auto tuple_node = ffi::GetRef<ObjectRef>(src->ref).as<TupleNode>()) {
    for (auto& it : tuple_node->fields) {
      sum += calc_args_number(it);
    }
  }
  return sum;
}

size_t GraphPartitioner::CountArgsLimit_(const IndexedForwardGraph::Node* child) {
  auto* outputs_list = child->outputs.head;
  size_t output_args = 0;
  while (outputs_list != nullptr) {
    output_args++;
    outputs_list = outputs_list->next;
  }
  return (max_function_args_ > output_args) ? max_function_args_ - output_args : 0;
}

size_t GraphPartitioner::CountFusedArgs(const IndexedForwardGraph& graph,
                                        IndexedForwardGraph::Node* child) {
  size_t args_num = 0;
  auto* outputs_list = child->outputs.head;
  while (outputs_list != nullptr) {
    args_num = std::max(args_num, CountArgs_(outputs_list->value.node, graph));
    outputs_list = outputs_list->next;
  }
  return args_num;
}

void GraphPartitioner::InitGroups(const IndexedForwardGraph& graph) {
  auto args_counter = [](const tvm::Object* obj) {
    size_t args_num = 0;
    if (auto call_node = ffi::GetRef<ObjectRef>(obj).as<CallNode>()) {
      for (auto& it : call_node->args) {
        if (it.as<VarNode>() || it.as<TupleGetItemNode>()) {
          args_num++;
        }
      }
    } else if (auto tuple_node = ffi::GetRef<ObjectRef>(obj).as<TupleNode>()) {
      for (auto& it : tuple_node->fields) {
        if (it.as<VarNode>() || it.as<TupleGetItemNode>()) {
          args_num++;
        }
      }
    } else if (ffi::GetRef<ObjectRef>(obj).as<VarNode>()) {
      args_num++;
    }
    return args_num;
  };

  groups_.resize(graph.post_dfs_order.size());
  for (size_t nid = 0; nid < groups_.size(); ++nid) {
    const auto* graph_node = graph.post_dfs_order[nid];
    auto* group_node = arena_->make<Group>();
    group_node->pattern = graph_node->pattern;
    group_node->mapping_type = graph_node->mapping_type;
    group_node->root_ref = graph_node->ref;
    // set anchor ref if necessary.
    if (group_node->pattern == kOutEWiseFusable) {
      group_node->anchor_ref = graph_node->ref;
    }
    group_node->args_num = args_counter(graph_node->ref);
    groups_[nid] = group_node;
  }
}

void GraphPartitioner::RunFuse(const IndexedForwardGraph& graph,    //
                               const DominatorTree& post_dom_tree,  //
                               int phase) {
  for (size_t nid = 0; nid < groups_.size(); ++nid) {
    // the group of current node has been specified already.
    auto* graph_node = graph.post_dfs_order[nid];
    auto* dom_node = post_dom_tree.nodes[nid];
    Group* group_node = groups_[nid];
    ICHECK(group_node != nullptr);
    postpone_node_ = nullptr;
    // Check if the fusing of some inputs was postponed
    if (postponed_fusing_map_.count(graph_node)) {
      auto range = postponed_fusing_map_.equal_range(graph_node);
      for (auto it = range.first; it != range.second; ++it) {
        // If the number of arguments is less than the limit then the input can be fused
        if (CountArgs_(graph_node, graph, false) <= CountArgsLimit_(graph_node)) {
          auto* src = it->second;
          auto* snode = post_dom_tree.nodes[src->index]->parent->gnode;
          if (groups_[snode->index]->anchor_ref != nullptr) continue;
          CommitFuse(src, snode);
        }
      }
      postponed_fusing_map_.erase(graph_node);
    }
    // no actions for opaque nodes
    if (group_node->pattern == kOpaque) continue;
    // no actions needed if the current node have no dominator
    if (dom_node->parent == nullptr) continue;
    ICHECK(!graph_node->extern_ref);
    size_t dom_parent_gindex = dom_node->parent->gnode->index;

    // refuse the fusion if too many ops are going to be fused together
    if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
      continue;
    // Refuse the fusion if too many arguments are going to be in the fused function
    if (max_function_args_ > 0) {
      auto limit = CountArgsLimit_(graph_node);
      if (limit > 0) {
        if (CountFusedArgs(graph, graph_node) > limit) {
          continue;
        }
      }
    }

    if (phase == 2) {
      // Fuse injective ops into intermediate tuples, if any
      if (group_node->pattern > kInjective) continue;
      Group* dom_parent_group = groups_[dom_parent_gindex];
      Group* dom_root_group = dom_parent_group->FindRoot();
      // If dom node group has a tuple as its root, we do not fuse tuple fields into it
      if (dom_root_group->pattern == kTuple) continue;
      if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
        // Now we know the tuple has been fused into subsequent injective ops
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        // dom_root_group can also be tuple, as in inception layers
        // CheckPath is needed to avoid fusing two intermediate tuples
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
      continue;
    }

    // Skip if current node is already fused to the parent.
    if (groups_[dom_parent_gindex] != nullptr &&
        group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
      continue;
    }
    // Do not fuse into tuple for now
    if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
    // Try to fuse current node to its post-dominator.
    if (group_node->pattern == kOutEWiseFusable) {
      if (phase != 0) continue;
      // Path for OutEWiseFusable: conv2d
      // Check if the dominator relation is elemwise.
      if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
        ICHECK(dom_node->parent->gnode != nullptr);
        // The fuse can be executed if all the intermediate ops are still broadcast.
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
    } else if (group_node->pattern <= kBroadcast) {
      // Pre-condition: can only be fused to parent which is injective or reduction.
      if (dom_node->parent != nullptr &&
          (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
        // Check if all the intermediate ops are still broadcast.
        // The final terminal node can already be fused to a OutEWiseFusable group.
        auto fcond = [](OpPatternKind kind, bool is_sink) {
          if (!is_sink) {
            // Elemwise, broadcast, and injective ops on the parallel branches
            // are allowed be fused to the elemwise/broadcast anchor.
            return kind <= kInjective;
          } else {
            return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
                    kind == kOutEWiseFusable);
          }
        };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      }
    } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
      // defer injective fusion to second phase.
      // so conv2d always finishes fusing.
      if (phase != 1) continue;
      // Check if all path are injective.
      auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
      if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
        CommitFuse(graph_node, dom_node->parent->gnode);
      }
    } else {
      // do nothing.
      ICHECK(group_node->pattern == kCommReduce);
    }
  }
}

bool GraphPartitioner::CheckEdgeConvexity(const IndexedForwardGraph& graph,
                                          IndexedForwardGraph::Node* src,
                                          IndexedForwardGraph::Node* sink) {
  Group* src_root = groups_[src->index]->FindRoot();
  Group* sink_root = groups_[sink->index]->FindRoot();
  if (src_root == sink_root) return true;
  const size_t num_nodes = graph.post_dfs_order.size();
  // Membership masks of the two groups about to merge.
  std::vector<bool> in_src(num_nodes, false);
  std::vector<bool> in_sink(num_nodes, false);
  for (size_t i = 0; i < num_nodes; ++i) {
    Group* root = groups_[i]->FindRoot();
    if (root == src_root) in_src[i] = true;
    if (root == sink_root) in_sink[i] = true;
  }
  // DFS forward from every edge that leaves src's group into an outside node.
  // Reaching sink's group means some src->sink path runs through nodes the
  // merged group would exclude, i.e. the merge would make the partition
  // quotient cyclic. (The reverse direction needs no check: the partition is
  // acyclic so far and the direct edge src->sink already exists, so a
  // sink->src path cannot.)
  std::vector<bool> visited(num_nodes, false);
  std::vector<IndexedForwardGraph::Node*> stack;
  for (size_t i = 0; i < num_nodes; ++i) {
    if (!in_src[i]) continue;
    for (auto* link = graph.post_dfs_order[i]->outputs.head; link != nullptr; link = link->next) {
      IndexedForwardGraph::Node* out = link->value.node;
      if (!in_src[out->index] && !in_sink[out->index]) stack.push_back(out);
    }
  }
  while (!stack.empty()) {
    IndexedForwardGraph::Node* node = stack.back();
    stack.pop_back();
    if (visited[node->index]) continue;
    visited[node->index] = true;
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      IndexedForwardGraph::Node* out = link->value.node;
      if (in_sink[out->index]) return false;
      if (!in_src[out->index] && !visited[out->index]) stack.push_back(out);
    }
  }
  return true;
}

void GraphPartitioner::CommitFuseEdge(IndexedForwardGraph::Node* src,
                                      IndexedForwardGraph::Node* sink,
                                      const DNNFuseRelation& relation) {
  // Direct-edge precondition: sink must be an immediate neighbour of src.
  bool sink_is_neighbour = false;
  for (auto* link = src->outputs.head; link != nullptr; link = link->next) {
    if (link->value.node == sink) {
      sink_is_neighbour = true;
      break;
    }
  }
  ICHECK(sink_is_neighbour) << "CommitFuseEdge requires a direct edge node[" << src->index
                            << "] -> node[" << sink->index << "]";
  Group* src_root = groups_[src->index]->FindRoot();
  Group* sink_root = groups_[sink->index]->FindRoot();
  if (src_root == sink_root) return;
  sink_root->num_nodes += src_root->num_nodes;
  sink_root->args_num += src_root->args_num;
  src_root->parent = sink_root;
  // Table 3's cell value: the mapping type of the operator after fusion.
  sink_root->mapping_type = relation.FusedType();
}

void GraphPartitioner::FuseSuccessor(const IndexedForwardGraph& graph,
                                     IndexedForwardGraph::Node* sp,
                                     IndexedForwardGraph::Node* successor, Block* block) {
  LOG(INFO) << "  successor of node[" << sp->index << "]:"
            << " node[" << successor->index << "] " << ffi::GetRef<ObjectRef>(successor->ref)
            << " (mapping=" << MappingTypeName(successor->mapping_type)
            << ", bytes=" << successor->output_size << ")";
  // check the mapping relationship
  MappingType sp_map = groups_[sp->index]->FindRoot()->mapping_type;
  MappingType succ_map = groups_[successor->index]->FindRoot()->mapping_type;
  DNNFuseRelation relation = DNNFuseRelation::Classify(sp_map, succ_map);
  LOG(INFO) << "    relation = " << relation.Name();
  // return if successor can not be fused
  if (relation.IsBreak()) return;
  // Structural gate: skip merges that would leave a src->sink path outside the
  // group (e.g. a residual skip edge fused around its own branch). The edge may
  // become fusable later, once the branch nodes have joined either group.
  if (!CheckEdgeConvexity(graph, sp, successor)) {
    LOG(INFO) << "    skipped: merge would break group convexity";
    return;
  }
  // kFuseDepend fusion is profit-gated: only merge when the profiler says the
  // fused kernel beats running the block and the candidate separately.
  if (relation.IsDepend()) {
    bool profitable = FuseProfit(*block, successor);
    if (!profitable) {
      LOG(INFO) << "    kFuseDepend: not profitable";
      return;
    }
    LOG(INFO) << "    kFuseDepend: profitable";
  }
  CommitFuseEdge(sp, successor, relation);
  block->insert(successor);
  // Recurse into the fused successor to extend the chain past one hop.
  for (auto* link = successor->outputs.head; link != nullptr; link = link->next) {
    FuseSuccessor(graph, successor, link->value.node, block);
  }
}

void GraphPartitioner::FusePredecessor(const IndexedForwardGraph& graph,
                                       IndexedForwardGraph::Node* sp,
                                       IndexedForwardGraph::Node* predecessor, Block* block) {
  LOG(INFO) << "  predecessor of node[" << sp->index << "]:"
            << " node[" << predecessor->index << "] " << ffi::GetRef<ObjectRef>(predecessor->ref)
            << " (mapping=" << MappingTypeName(predecessor->mapping_type)
            << ", bytes=" << predecessor->output_size << ")";
  // check the mapping relationship
  MappingType sp_map = groups_[sp->index]->FindRoot()->mapping_type;
  MappingType pred_map = groups_[predecessor->index]->FindRoot()->mapping_type;
  DNNFuseRelation relation = DNNFuseRelation::Classify(pred_map, sp_map);
  LOG(INFO) << "    relation = " << relation.Name();
  // return if predecessor can not be fused
  if (relation.IsBreak()) return;
  // Structural gate: mirror of FuseSuccessor's convexity check, oriented along
  // the direct edge predecessor -> sp.
  if (!CheckEdgeConvexity(graph, predecessor, sp)) {
    LOG(INFO) << "    skipped: merge would break group convexity";
    return;
  }
  // kFuseDepend fusion is profit-gated: only merge when the profiler says the
  // fused kernel beats running the block and the candidate separately.
  if (relation.IsDepend()) {
    bool profitable = FuseProfit(*block, predecessor);
    if (!profitable) {
      LOG(INFO) << "    kFuseDepend: not profitable";
      return;
    }
    LOG(INFO) << "    kFuseDepend: profitable";
  }
  CommitFuseEdge(predecessor, sp, relation);
  block->insert(predecessor);
  // Recurse into the fused predecessor to extend the chain past one hop.
  for (auto* link = predecessor->inputs.head; link != nullptr; link = link->next) {
    FusePredecessor(graph, predecessor, link->value.node, block);
  }
}

void GraphPartitioner::RunDNNFuse(const IndexedForwardGraph& graph) {
  graph.DebugDump();
  // unfused_ops = all_operaters
  std::unordered_set<IndexedForwardGraph::Node*> unfused_ops(
      graph.post_dfs_order.begin(), graph.post_dfs_order.end());
  LOG(INFO) << "unfused_ops: " << unfused_ops.size() << " nodes";
  IndexedForwardGraph::Node* seed = nullptr;
  // generate seed
  while ((seed = FindMinOtO(unfused_ops)) != nullptr) {
    // block = [ seed ]
    Block block{seed};
    LOG(INFO) << "\nkOneToOne op with min output_size:"
              << " node[" << seed->index << "], " << ffi::GetRef<ObjectRef>(seed->ref)
              << " bytes=" << seed->output_size;
    // head to successor
    for (auto* link = seed->outputs.head; link != nullptr; link = link->next) {
      FuseSuccessor(graph, seed, link->value.node, &block);
    }
    // head to predecessor
    for (auto* link = seed->inputs.head; link != nullptr; link = link->next) {
      FusePredecessor(graph, seed, link->value.node, &block);
    }
    // unfused_ops = unfused_ops - block
    for (IndexedForwardGraph::Node* op : block) unfused_ops.erase(op);
  }
}

bool GraphPartitioner::FuseProfit(const Block& block, IndexedForwardGraph::Node* candidate) {
  LOG(INFO) << "    kFuseDepend: FuseProfit prototype disabled for block_size=" << block.size()
            << " candidate=node[" << candidate->index << "]";
  return false;
}

}  // namespace relax
}  // namespace tvm
