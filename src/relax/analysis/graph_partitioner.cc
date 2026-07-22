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

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
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

void GraphPartitioner::DumpNodePrimFunc(const char* label,
                                        const IndexedForwardGraph::Node* node) {
  if (node->gvar == nullptr) {
    LOG(INFO) << "    " << label << " node[" << node->index << "]: no PrimFunc";
    return;
  }
  GlobalVar gvar = ffi::GetRef<GlobalVar>(node->gvar);
  LOG(INFO) << "    " << label << " node[" << node->index << "] PrimFunc " << node->gvar->name_hint
            << ":\n"
            << Downcast<tir::PrimFunc>(mod_->Lookup(gvar));
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

namespace {

// Sample count for the in-pass cost oracle: TimePrimFunc averages over this
// many timed device runs (after one untimed warmup).
constexpr int kProfileRuns = 100;
constexpr int kProfileRepeats = 3;

// MetaSchedule trials used to schedule each profiled PrimFunc. The cost oracle
// tunes each kernel briefly (rather than the DefaultGPUSchedule heuristic) so
// the timed kernel reflects a tuned schedule.
constexpr int kProfileTuneTrials = 4;

// When false, skip MetaSchedule tuning entirely and schedule every profiled
// kernel with the DefaultGPUSchedule heuristic (used for fast smoke tests). Set
// true to tune each profiled kernel with kProfileTuneTrials MS trials.
constexpr bool kUseMetaSchedule = true;

// Schedule `func` and build it on `target`, returning the callable device
// kernel; nullopt if scheduling or build fails. An unscheduled PrimFunc has no
// thread bindings, so it must be scheduled before tir.build. Scheduling reads
// Target::Current(), so build stays under the same target scope that drives the
// host/device split. The PrimFunc is MetaSchedule-tuned (kProfileTuneTrials
// trials) via the `relax.dnnf.MetaScheduleSchedulePrimFunc` Python helper, with
// a DefaultGPUSchedule fallback when MS is unavailable or finds no schedule.
// `record_name` names the tuning-record subdir under $DNNF_PROFILER_WORKDIR.
ffi::Optional<ffi::Function> BuildPrimFuncGPU(const tir::PrimFunc& func, const Target& target,
                                              const std::string& record_name) {
  try {
    tvm::With<Target> target_scope(target);
    tir::PrimFunc named = WithAttr(func, tvm::attr::kGlobalSymbol, ffi::String("tir_function"));
    GlobalVar gv("tir_function");
    ffi::Map<GlobalVar, BaseFunc> funcs;
    funcs.Set(gv, named);
    IRModule m(funcs);

    tir::PrimFunc scheduled;
    ffi::Optional<tir::PrimFunc> tuned;
    if (kUseMetaSchedule) {
      if (auto ms_schedule =
              ffi::Function::GetGlobal("relax.dnnf.MetaScheduleSchedulePrimFunc")) {
        ffi::Any ret = (*ms_schedule)(named, target, kProfileTuneTrials, ffi::String(record_name));
        tuned = ret.try_cast<tir::PrimFunc>();
      }
    }
    if (tuned) {
      scheduled = tuned.value();
    } else {
      // No tuned schedule (MS disabled for smoke test, MS not imported, or no
      // valid record in the trial budget): fall back to DefaultGPUSchedule.
      LOG(INFO) << "  MetaSchedule disabled/unavailable; DefaultGPUSchedule fallback";
      m = tir::transform::DefaultGPUSchedule()(m);
      scheduled = Downcast<tir::PrimFunc>(m->Lookup(gv));
    }

    const auto build = tvm::ffi::Function::GetGlobalRequired("tir.build");
    ffi::Module rt_module = build(scheduled, target).cast<ffi::Module>();
    return rt_module->GetFunction("tir_function").value();
  } catch (const tvm::Error& err) {
    LOG(INFO) << "  build failed: " << err.what();
    return std::nullopt;
  }
}

// Materialize one device tensor per buffer param of `func` (inputs + outputs, in
// param order): allocate on `cpu_dev`, random-fill float32 buffers with values in
// [-1, 1], then copy to `dev` (a GPU kernel can't be fed host pointers, and we
// can't write GPU memory directly; for a CPU `dev` this CopyTo is just a host
// copy). nullopt if any param is not a buffer or has a dynamic shape.
ffi::Optional<std::vector<runtime::Tensor>> MakeRandomDeviceArgs(const tir::PrimFunc& func,
                                                                 DLDevice dev,
                                                                 DLDevice cpu_dev) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<runtime::Tensor> args;
  for (const tir::Var& param : func->params) {
    ffi::Optional<tir::Buffer> opt_buf = func->buffer_map.Get(param);
    if (!opt_buf) {
      LOG(INFO) << "  param " << param << " is not a buffer; skip profiling";
      return std::nullopt;
    }
    tir::Buffer buf = opt_buf.value();
    std::vector<int64_t> shape;
    for (const PrimExpr& dim : buf->shape) {
      const auto* imm = dim.as<IntImmNode>();
      if (imm == nullptr) {
        LOG(INFO) << "  param " << param << " has a dynamic shape; skip profiling";
        return std::nullopt;
      }
      shape.push_back(imm->value);
    }
    runtime::Tensor host_t = runtime::Tensor::Empty(ffi::Shape(shape), buf->dtype, cpu_dev);
    if (buf->dtype == DataType::Float(32)) {
      int64_t numel = 1;
      for (int64_t d : shape) numel *= d;
      float* p = static_cast<float*>(host_t->data);
      for (int64_t i = 0; i < numel; ++i) p[i] = dist(rng);
    }
    args.push_back(host_t.CopyTo(dev));  // CopyTo triggers a TVMSynchronize
  }
  return args;
}

// Run `kernel` on `args` once untimed (warmup, to absorb cold-cache / first-
// dispatch PTX-JIT cost), then time batches of kProfileRuns launches with the
// device timer (e.g. cudaEvent elapsed time on CUDA; a host-clock timer on CPU).
double TimeKernel(const ffi::Function& kernel, const std::vector<runtime::Tensor>& args,
                  DLDevice dev) {
  // Pack args once (AnyView is non-owning, so `args` must outlive the calls).
  std::vector<ffi::AnyView> packed(args.size());
  for (size_t i = 0; i < args.size(); ++i) packed[i] = args[i];

  runtime::DeviceAPI* dev_api = runtime::DeviceAPI::Get(dev);

  {
    ffi::Any ret;
    kernel.CallPacked(ffi::PackedArgs(packed.data(), packed.size()), &ret);
    dev_api->StreamSync(dev, nullptr);
  }

  double total_us = 0.0;
  for (int repeat = 0; repeat < kProfileRepeats; ++repeat) {
    runtime::Timer timer = runtime::Timer::Start(dev);
    for (int r = 0; r < kProfileRuns; ++r) {
      ffi::Any ret;
      kernel.CallPacked(ffi::PackedArgs(packed.data(), packed.size()), &ret);
    }
    timer->Stop();
    total_us += static_cast<double>(timer->SyncAndGetElapsedNanos()) / 1000.0 / kProfileRuns;
  }
  return total_us / kProfileRepeats;
}

// Build `func` on the ambient Target::Current() (BuildPrimFuncGPU), feed it
// random inputs on that target's device (MakeRandomDeviceArgs), and time it
// (TimeKernel); -1 if the build fails or any param cannot be materialized.
// Requires a Target to be in scope -- callers wrap the surrounding FuseOps()
// call in `with target:` -- so both the build and the timing device follow
// whatever target is actually driving the pass, rather than a value hardcoded
// here (profiling a CPU target's PrimFunc with CUDA-resident args would trip
// the device_type assert TVM's calling convention bakes into every arg).
double TimePrimFunc(const tir::PrimFunc& func, const std::string& record_name) {
  Target target = Target::Current();
  ICHECK(target.defined()) << "TimePrimFunc requires a Target to be in scope "
                            << "(wrap the FuseOps() call in `with target:`)";
  DLDevice dev = {static_cast<DLDeviceType>(target->GetTargetDeviceType()), 0};
  DLDevice cpu_dev = {kDLCPU, 0};

  LOG(INFO) << "  PrimFunc to build:\n" << func;
  ffi::Optional<ffi::Function> kernel = BuildPrimFuncGPU(func, target, record_name);
  if (!kernel) return -1.0;
  ffi::Optional<std::vector<runtime::Tensor>> args = MakeRandomDeviceArgs(func, dev, cpu_dev);
  if (!args) return -1.0;
  return TimeKernel(kernel.value(), args.value(), dev);
}

// Find the call_tir Call bound to `var` in any relax function of `mod`.
ffi::Optional<Call> FindCallTIR(const IRModule& mod, const Object* var) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  for (const auto& kv : mod->functions) {
    const auto* func = kv.second.as<FunctionNode>();
    if (func == nullptr) continue;
    for (const BindingBlock& block : func->body->blocks) {
      for (const Binding& binding : block->bindings) {
        const auto* vb = binding.as<VarBindingNode>();
        if (vb == nullptr || vb->var.get() != var) continue;
        const auto* call = vb->value.as<CallNode>();
        if (call != nullptr && call->op.same_as(call_tir_op)) {
          return ffi::GetRef<Call>(call);
        }
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

// Construct the inner `fused_block` relax Function over `nodes` (given in
// producer-before-consumer order, i.e. sorted by IndexedForwardGraph index): a
// kPrimitive dataflow block chaining every node's call_tir. Each callee PrimFunc
// is registered in `bb`; an input that is the output of another node in the set
// is wired var-to-var (the internal edges), while every other input -- including
// constants -- becomes a tensor param, so the kernel is self-contained. Each
// node output not consumed inside the set becomes a function output (a single
// var, or a Tuple if several). Returns nullopt if any node lacks a call_tir
// binding.
ffi::Optional<Function> MakeFusedBlockFunc(
    BlockBuilder bb, const IRModule& mod,
    const std::vector<const IndexedForwardGraph::Node*>& nodes) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");

  // Resolve every node's call_tir up front so we can bail before mutating `bb`.
  std::vector<Call> calls;
  calls.reserve(nodes.size());
  for (const auto* n : nodes) {
    ffi::Optional<Call> c = FindCallTIR(mod, n->ref);
    if (!c) return std::nullopt;
    calls.push_back(c.value());
  }

  ffi::Array<Var> params;
  int pidx = 0;
  std::unordered_map<const Object*, Var> produced;  // node->ref -> emitted var
  std::unordered_set<const Object*> consumed;        // outputs used within the set

  bb->BeginDataflowBlock();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Call& call = calls[i];
    auto pf = Downcast<tir::PrimFunc>(mod->Lookup(Downcast<GlobalVar>(call->args[0])));
    GlobalVar callee = bb->AddFunction(pf, "k" + std::to_string(i));
    ffi::Array<Expr> args;
    for (const Expr& a : Downcast<Tuple>(call->args[1])->fields) {
      auto it = produced.find(a.get());
      if (it != produced.end()) {
        consumed.insert(a.get());  // internal edge: another node's output feeds this op
        args.push_back(it->second);
      } else {
        Var param("p" + std::to_string(pidx++), GetStructInfo(a));
        params.push_back(param);
        args.push_back(param);
      }
    }
    Call inner(call_tir_op, {callee, Tuple(args)}, Attrs(), call->sinfo_args);
    produced[nodes[i]->ref] = bb->Emit(inner);
  }

  // Any node whose output is not consumed by another node in the set escapes.
  // The highest-index node is never consumed internally, so there is >= 1.
  ffi::Array<Expr> outs;
  for (const auto* n : nodes) {
    if (!consumed.count(n->ref)) outs.push_back(produced[n->ref]);
  }
  Expr out_expr;
  if (outs.size() == 1) {
    out_expr = outs[0];
  } else {
    out_expr = Tuple(outs);
  }
  Var out = bb->EmitOutput(out_expr);
  BindingBlock blk = bb->EndBlock();

  Expr body = bb->Normalize(out);
  body = bb->Normalize(SeqExpr({blk}, body));
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set(attr::kPrimitive, true);
  return Function(params, body, /*ret_struct_info=*/std::nullopt, /*is_pure=*/true, DictAttrs(attrs));
}

// Append a public `main` to `bb` that forwards fresh params straight to
// `callee_gv`. FuseTIR ends with DeadCodeElimination, which would otherwise reap
// the merged kernel since it has no caller in this standalone module.
void AppendMainCaller(BlockBuilder bb, const GlobalVar& callee_gv, const ffi::Array<Var>& params) {
  ffi::Array<Var> main_params;
  ffi::Array<Expr> main_args;
  for (const Var& p : params) {
    Var mp(p->name_hint() + "_in", GetStructInfo(p));
    main_params.push_back(mp);
    main_args.push_back(mp);
  }
  bb->BeginDataflowBlock();
  Var main_out = bb->EmitOutput(Call(callee_gv, main_args));
  BindingBlock main_blk = bb->EndBlock();
  Expr main_body = bb->Normalize(SeqExpr({main_blk}, bb->Normalize(main_out)));
  ffi::Map<ffi::String, ffi::Any> main_attrs;
  main_attrs.Set(tvm::attr::kGlobalSymbol, ffi::String("main"));
  Function main_fn(main_params, main_body, /*ret_struct_info=*/std::nullopt, /*is_pure=*/true,
                   DictAttrs(main_attrs));
  bb->AddFunction(main_fn, "main");
}

// Run FuseTIR over bb's context module and return `target_gv`'s merged PrimFunc;
// nullopt if FuseTIR throws or the result is not a PrimFunc.
ffi::Optional<tir::PrimFunc> FuseTIRAndExtract(BlockBuilder bb, const GlobalVar& target_gv) {
  IRModule scratch = bb->GetContextIRModule();
  try {
    scratch = transform::FuseTIR()(scratch);
  } catch (const tvm::Error& err) {
    LOG(INFO) << "  FuseTIR failed: " << err.what();
    return std::nullopt;
  }
  ffi::Optional<BaseFunc> merged =
      scratch->functions.Get(scratch->GetGlobalVar(target_gv->name_hint));
  if (merged) {
    if (const auto* pf = merged.value().as<tir::PrimFuncNode>()) {
      return ffi::GetRef<tir::PrimFunc>(pf);
    }
  }
  LOG(INFO) << "  fused result is not a PrimFunc";
  return std::nullopt;
}

// Build a kPrimitive module fusing all of `nodes` (producer-before-consumer
// order), run FuseTIR, and return the merged PrimFunc. Internal edges are wired
// var-to-var; every other input -- including constant args -- becomes a tensor
// param, so the merged kernel is self-contained. Returns nullopt if any node
// lacks a call_tir binding or FuseTIR cannot produce a single PrimFunc.
ffi::Optional<tir::PrimFunc> BuildFusedBlock(
    const IRModule& mod, const std::vector<const IndexedForwardGraph::Node*>& nodes) {
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  ffi::Optional<Function> fused = MakeFusedBlockFunc(bb, mod, nodes);
  if (!fused) return std::nullopt;
  GlobalVar fused_gv = bb->AddFunction(fused.value(), "fused_block");
  AppendMainCaller(bb, fused_gv, fused.value()->params);
  return FuseTIRAndExtract(bb, fused_gv);
}

// Join the nodes' kernel names with '_' into a filesystem-safe label for
// tuning-record directory names ("matmul_add"). If `marked` is given, its name
// is bracketed in place ("matmul_[relu]_add").
std::string JoinNodeNames(const std::vector<const IndexedForwardGraph::Node*>& nodes,
                          const IndexedForwardGraph::Node* marked = nullptr) {
  std::ostringstream names;
  for (size_t i = 0; i < nodes.size(); ++i) {
    ffi::String name = nodes[i]->gvar ? nodes[i]->gvar->name_hint : ffi::String("unknown");
    names << (i ? "_" : "");
    if (nodes[i] == marked) {
      names << "[" << name << "]";
    } else {
      names << name;
    }
  }
  return names.str();
}

}  // namespace

double GraphPartitioner::TimeNode(const IndexedForwardGraph::Node* node,
                                  const std::string& record_name) {
  if (node->gvar == nullptr) {
    LOG(INFO) << "  node has no PrimFunc; skip profiling";
    return -1.0;
  }
  LOG(INFO) << "  TimeNode: " << node->gvar->name_hint;
  return TimePrimFunc(
      Downcast<tir::PrimFunc>(mod_->Lookup(ffi::GetRef<GlobalVar>(node->gvar))), record_name);
}

double GraphPartitioner::TimeFusedBlock(
    const std::vector<const IndexedForwardGraph::Node*>& nodes, const std::string& record_name) {
  // Build the merged kernel fusing every node in `nodes` (FuseTIR over a
  // kPrimitive module) and time it; -1.0 if any op lacks a call_tir binding or
  // the fused PrimFunc cannot be built. The block counterpart of TimeNode.
  ffi::Optional<tir::PrimFunc> fused = BuildFusedBlock(mod_, nodes);
  if (!fused) return -1.0;
  std::ostringstream names;
  for (size_t i = 0; i < nodes.size(); ++i) {
    names << (i ? " + " : "") << (nodes[i]->gvar ? nodes[i]->gvar->name_hint : ffi::String("?"));
  }
  LOG(INFO) << "  TimeFusedBlock: " << names.str();
  return TimePrimFunc(fused.value(), record_name);
}

bool GraphPartitioner::FuseProfit(const Block& block, IndexedForwardGraph::Node* candidate) {
  // Group-level profit (DNNFusion §4.3.2): fuse only if `block` + `candidate` as
  // one kernel beats them run apart. `block` excludes `candidate` and is index-
  // ordered, so it and `if_fuse` iterate producer-before-consumer for FuseTIR.
  Block if_fuse = block;
  if_fuse.insert(candidate);
  std::vector<const IndexedForwardGraph::Node*> block_nodes(block.begin(), block.end());
  std::vector<const IndexedForwardGraph::Node*> if_fuse_nodes(if_fuse.begin(), if_fuse.end());

  // Time the block alone, the candidate alone, and the block+candidate fused
  // (any -1.0 = a PrimFunc lookup / build / timing failure). Each FuseProfit call
  // keeps its three timing targets' tuning records under one per-call subdir of
  // $DNNF_PROFILER_WORKDIR: task_<index>__<names>/{block,candidate,fused}--<name>,
  // where <names> lists block + candidate in producer-before-consumer order with
  // the candidate bracketed in place ("conv_[relu]_add").
  std::string cand_name(candidate->gvar ? std::string(candidate->gvar->name_hint) : "unknown");
  std::string block_name = JoinNodeNames(block_nodes);
  std::string task_dir = "task_" + std::to_string(fuse_profit_count_++) + "__" +
                         JoinNodeNames(if_fuse_nodes, candidate) + "/";
  double block_latency = TimeFusedBlock(block_nodes, task_dir + "block." + block_name);
  double cand_latency = TimeNode(candidate, task_dir + "candi." + cand_name);
  double if_fuse_latency =
      TimeFusedBlock(if_fuse_nodes, task_dir + "fused." + JoinNodeNames(if_fuse_nodes));
  double not_fuse_latency = block_latency + cand_latency;
  if (block_latency < 0.0 || cand_latency < 0.0 || if_fuse_latency < 0.0) {
    LOG(INFO) << "    profile: timing failed (block=" << block_latency << " cand=" << cand_latency
              << " if_fuse=" << if_fuse_latency << ") - skip";
    return false;
  }

  bool profitable = if_fuse_latency < not_fuse_latency;
  LOG(INFO) << "    profile: if_fuse=" << if_fuse_latency << "us vs not_fuse="
            << not_fuse_latency << "us (block=" << block_latency
            << " cand=" << cand_latency << ") -> " << (profitable ? "fuse" : "skip");
  return profitable;
}

}  // namespace relax
}  // namespace tvm
