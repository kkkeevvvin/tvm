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

#include <limits>
#include <set>
#include <vector>

namespace tvm {
namespace relax {

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

// elmsd v1: seed-driven fusion replacing the post-dominator + 3-phase RunFuse loop.
// See MLC-elmsd/log/20260507_elmsd_v1_design.md.
//
// Algorithm (DNNFusion §4.3.2 Listing 1, with TVM OpPattern as the fusibility check):
//   while seed = pick min-output-size kElemWise op from unfused:
//     fuse seed's outputs recursively (depth-first)
//     fuse seed's inputs  recursively (depth-first)
//     remove fused block from unfused
//
// Note: RunFuse / CommitFuse / CheckPath / DominatorTree are no longer called from
// Partition, but are kept in the source for now (will be removed in a follow-up
// cleanup once the new algorithm is validated).
std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);

  const size_t N = graph.post_dfs_order.size();

  // Build reverse adjacency once for predecessor walks (IFG only stores forward edges).
  std::vector<std::vector<IndexedForwardGraph::Node*>> preds(N);
  for (auto* node : graph.post_dfs_order) {
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      preds[link->value.node->index].push_back(node);
    }
  }

  // Initialize unfused set with all node indices. std::set keeps post-DFS order
  // ascending, giving deterministic tie-break when multiple seeds share min IRS.
  std::set<size_t> unfused;
  for (size_t i = 0; i < N; ++i) unfused.insert(i);

  while (auto* seed = this->ElmsdGenerateSeed(unfused, graph)) {
    // Forward (successor) walk
    visited_.clear();
    visited_.insert(seed);
    this->ElmsdFuseDirection(seed, seed, graph, /*forward=*/true, preds);
    // Backward (predecessor) walk
    visited_.clear();
    visited_.insert(seed);
    this->ElmsdFuseDirection(seed, seed, graph, /*forward=*/false, preds);
    // Remove every node now in seed's group from unfused. Always erase seed itself
    // so the loop makes progress even if the block did not grow.
    Group* root = groups_[seed->index]->FindRoot();
    for (size_t i = 0; i < N; ++i) {
      if (groups_[i]->FindRoot() == root) unfused.erase(i);
    }
  }

  return std::move(groups_);
}

IndexedForwardGraph::Node* GraphPartitioner::ElmsdGenerateSeed(
    const std::set<size_t>& unfused, const IndexedForwardGraph& graph) {
  IndexedForwardGraph::Node* best = nullptr;
  int64_t best_size = std::numeric_limits<int64_t>::max();
  for (size_t idx : unfused) {
    auto* n = graph.post_dfs_order[idx];
    if (n->pattern != kElemWise) continue;
    // extern_ref is allowed: a kElemWise output binding (e.g. the last relu) is a
    // valid seed; fusing toward its predecessors mirrors TVM phase 0's behavior of
    // letting an extern_ref node serve as the dom-parent sink.
    if (n->output_size <= 0) continue;  // unknown / non-tensor; skip.
    if (n->output_size < best_size) {
      best = n;
      best_size = n->output_size;
    }
  }
  return best;
}

void GraphPartitioner::ElmsdFuseDirection(
    IndexedForwardGraph::Node* seed, IndexedForwardGraph::Node* current,
    const IndexedForwardGraph& graph, bool forward,
    const std::vector<std::vector<IndexedForwardGraph::Node*>>& preds) {
  // Collect neighbors in chosen direction (forward=outputs, backward=preds).
  std::vector<IndexedForwardGraph::Node*> neighbors;
  if (forward) {
    for (auto* link = current->outputs.head; link != nullptr; link = link->next) {
      neighbors.push_back(link->value.node);
    }
  } else {
    neighbors = preds[current->index];
  }

  for (auto* nbr : neighbors) {
    if (visited_.count(nbr)) continue;
    visited_.insert(nbr);
    // Note: extern_ref is NOT auto-skipped. Param nbrs (extern_ref + kOpaque) are
    // rejected by ElmsdIsFusible's kOpaque check. Output bindings (extern_ref +
    // kElemWise / kBroadcast / etc.) are allowed to be absorbed.

    Group* seed_root = groups_[seed->index]->FindRoot();
    Group* nbr_root = groups_[nbr->index]->FindRoot();

    // Skip neighbors that already belong to a different non-singleton block (i.e.,
    // a previously processed seed's block). DNNFusion's outer loop iterates with
    // unfused_ops -= block; the equivalent here is: don't reach across into
    // committed blocks. When seed has merged with nbr earlier in this same call,
    // nbr is in visited_ and was skipped above.
    if (nbr_root != seed_root && nbr_root->num_nodes > 1) continue;

    if (!this->ElmsdIsFusible(seed, nbr)) continue;

    // Cycle prevention for forward walk: a multi-input successor would create a
    // cross-group cycle if any of its other inputs sits in a different group that
    // (transitively) depends on seed's block. The conservative rule is to demand
    // that every input of `nbr` is either in seed's block already or an extern
    // param (kOpaque). This trades some fusion opportunity for safety; matching
    // TVM's CheckPath-style multi-path absorption is left to a follow-up branch.
    if (forward) {
      bool has_external_input = false;
      for (auto* p : preds[nbr->index]) {
        Group* p_root = groups_[p->index]->FindRoot();
        if (p_root == seed_root) continue;                 // already in block
        if (p->extern_ref && p->pattern == kOpaque) continue;  // param
        has_external_input = true;
        break;
      }
      if (has_external_input) continue;
    }

    // Honor max_fuse_depth_ limit (nbr is currently a singleton if we got here,
    // so adding it costs 1 node).
    if (seed_root->num_nodes + nbr_root->num_nodes > max_fuse_depth_) continue;

    this->ElmsdMergeIntoSeedGroup(nbr_root, seed_root);
    this->ElmsdFuseDirection(seed, nbr, graph, forward, preds);
  }
}

bool GraphPartitioner::ElmsdIsFusible(IndexedForwardGraph::Node* seed_node,
                                      IndexedForwardGraph::Node* nbr) {
  Group* seed_group = groups_[seed_node->index]->FindRoot();
  OpPatternKind block_pat = seed_group->pattern;
  OpPatternKind nbr_pat = nbr->pattern;

  // Reject opaque / tuple at either end.
  if (block_pat == kOpaque || block_pat == kTuple) return false;
  if (nbr_pat == kOpaque || nbr_pat == kTuple) return false;

  // Block already contains a heavy op (kCommReduce / kOutEWiseFusable): saturated.
  if (block_pat > kInjective) return false;

  // Defensive: refuse anything heavier than kOutEWiseFusable as the neighbor
  // (kTuple/kOpaque already filtered above; this catches future enum additions).
  if (nbr_pat > kOutEWiseFusable) return false;

  return true;
}

void GraphPartitioner::ElmsdMergeIntoSeedGroup(Group* nbr_group, Group* seed_root) {
  nbr_group = nbr_group->FindRoot();
  seed_root = seed_root->FindRoot();
  if (nbr_group == seed_root) return;
  seed_root->num_nodes += nbr_group->num_nodes;
  seed_root->args_num += nbr_group->args_num;
  nbr_group->parent = seed_root;
  if (nbr_group->anchor_ref != nullptr) {
    ICHECK(seed_root->anchor_ref == nullptr)
        << "elmsd: merging two anchored groups; ElmsdIsFusible should have prevented this";
    seed_root->anchor_ref = nbr_group->anchor_ref;
  }
  // Pattern: take max. Bypass CombinePattern's fatal because ElmsdIsFusible already
  // enforces "block.pat <= kInjective when accepting a new neighbor", so the
  // dangerous double-anchor case never reaches here.
  if (nbr_group->pattern > seed_root->pattern) {
    seed_root->pattern = nbr_group->pattern;
  }
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

}  // namespace relax
}  // namespace tvm
