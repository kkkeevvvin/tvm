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

#include <functional>
#include <limits>
#include <unordered_set>
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

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);

  if (algorithm_ == Algorithm::kDnnfusion) {
    // DNNFusion §4.3 Listing 1: a single seed-driven plan generator pass,
    // not staged by phase.  The classifier + Table 3 inside RunFuseDnnfusion
    // already handles the mapping-type cascade that TVM's 3-phase loop
    // expresses through its OpPatternKind hierarchy.
    this->RunFuseDnnfusion(graph);
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

// ---------------------------------------------------------------------------
// DNNFusion §4.3 Listing 1 fusion plan generator.
//
// Layout:
//   - ComputeMappingTypes / ComputeIrsBytes prepass produces per-node
//     side data; cached so the recursion below is O(1) per query.
//   - PickSeed selects the smallest-IRS OtO node still unmerged.
//   - TryFuseSucc / TryFusePred walk forward / backward edges,
//     consulting Table 3 (FuseMappingCheck) and the conservative
//     yellow-policy fallback.  Each successful candidate is merged into
//     the seed's union-find group via the existing MergeFromTo helper.
// ---------------------------------------------------------------------------
void GraphPartitioner::RunFuseDnnfusion(const IndexedForwardGraph& graph) {
  using dnnfusion::FuseColor;
  using dnnfusion::FuseDecision;
  using dnnfusion::FuseMappingCheck;
  using dnnfusion::MappingType;
  const size_t n = graph.post_dfs_order.size();
  if (n == 0) return;

  // Clear anchor_refs set during initialisation: those guard TVM's original
  // kOutEWiseFusable (MtM) merging logic, but DNNFusion governs MtM fusions
  // via Table 3 instead.  Without this, MergeFromTo's ICHECK fires whenever
  // two kOutEWiseFusable groups are merged.
  for (size_t i = 0; i < n; ++i) {
    groups_[i]->anchor_ref = nullptr;
  }

  // --- Prepass: classify every node + compute IRS bytes.
  std::vector<MappingType> mapping_type(n, MappingType::kUnknown);
  std::vector<int64_t> irs_bytes(n, -1);
  for (size_t i = 0; i < n; ++i) {
    auto* node = graph.post_dfs_order[i];
    mapping_type[i] = dnnfusion::DeriveNodeMappingType(node->ref, node->pattern, var_to_op_name_);
    irs_bytes[i] = dnnfusion::ComputeIrsBytes(node->ref);
  }

  // --- Build immediate-predecessor lists (graph stores forward edges only).
  std::vector<std::vector<size_t>> preds(n);
  for (size_t i = 0; i < n; ++i) {
    auto* node = graph.post_dfs_order[i];
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      preds[link->value.node->index].push_back(i);
    }
  }

  // Returns true if the candidate node is eligible to be touched at all
  // (skip params, constants, externs, opaque pattern).
  auto eligible = [&](size_t idx) -> bool {
    auto* node = graph.post_dfs_order[idx];
    if (node->extern_ref) return false;
    if (node->pattern == kOpaque) return false;
    if (mapping_type[idx] == MappingType::kUnknown) return false;
    return true;
  };

  // --- Constraint: would merging child into parent's group blow past the
  // configured max_fuse_depth_ on number of fused nodes?  Args limit is
  // intentionally not enforced here; max_function_args_ is set to 0 by
  // the default fuse_ops.cc invocation, and the DNNFusion-specific cost
  // model is left to a follow-up M3.
  auto exceeds_depth = [&](size_t parent_idx, size_t child_idx) -> bool {
    if (max_fuse_depth_ == 0) return false;
    Group* p = groups_[parent_idx]->FindRoot();
    Group* c = groups_[child_idx]->FindRoot();
    if (p == c) return false;
    return (p->num_nodes + c->num_nodes) > max_fuse_depth_;
  };

  // --- Edge-level fusion test, returns true on commit.  Implements
  // Listing 1 lines 9-21 (mapping type analysis -> constraint check ->
  // commit) with the §4.3.2 profile-based oracle replaced by the
  // conservative / aggressive yellow-policy switch.
  //
  // `forward` controls argument order for FuseMappingCheck:
  //   forward=true  → op_idx produces output consumed by cand_idx
  //                   (Listing 1 Step II: FusionDecision(cur, succ))
  //   forward=false → cand_idx produces output consumed by op_idx
  //                   (Listing 1 Step III: FusionDecision(pred, cur))
  auto try_fuse_edge = [&](size_t op_idx, size_t cand_idx, bool forward) -> bool {
    if (cand_idx == op_idx) return false;
    if (!eligible(cand_idx)) return false;
    if (groups_[cand_idx]->FindRoot() != groups_[cand_idx]) return false;
    size_t first = forward ? op_idx : cand_idx;
    size_t second = forward ? cand_idx : op_idx;
    FuseDecision d = FuseMappingCheck(mapping_type[first], mapping_type[second]);
    if (d.color == FuseColor::kRed) return false;
    if (d.color == FuseColor::kYellow && !dnnfusion_options_.aggressive_yellow) return false;
    if (exceeds_depth(op_idx, cand_idx)) return false;
    Group* parent_root = groups_[op_idx]->FindRoot();
    MergeFromTo(groups_[cand_idx], parent_root);
    return true;
  };

  // --- Cycle guard for walk_succ: returns false if absorbing succ_idx into
  // G = groups_[cur_idx]->FindRoot() would create a group-level cycle.
  //
  // A cycle arises when succ_idx has a predecessor P outside G and some
  // ancestor of P (following pred links backward) already belongs to G.
  // That means G's output transitively feeds P, so after the merge:
  //   G depends on Group(P)  (via succ←P)
  //   Group(P) depends on G  (via P's ancestor in G)
  // — a bidirectional dependency that OperatorFusor rejects.
  auto safe_to_merge_succ = [&](size_t cur_idx, size_t succ_idx) -> bool {
    Group* G = groups_[cur_idx]->FindRoot();
    for (size_t p : preds[succ_idx]) {
      if (groups_[p]->FindRoot() == G) continue;  // already inside G
      if (graph.post_dfs_order[p]->extern_ref) continue;
      // Backward BFS from p: if we reach any node whose group root is G,
      // the merge would be cyclic.
      std::unordered_set<size_t> visited;
      std::vector<size_t> queue;
      queue.push_back(p);
      visited.insert(p);
      while (!queue.empty()) {
        size_t cur = queue.back();
        queue.pop_back();
        if (groups_[cur]->FindRoot() == G) return false;
        for (size_t pp : preds[cur]) {
          if (visited.insert(pp).second) {
            queue.push_back(pp);
          }
        }
      }
    }
    return true;
  };

  // --- Step II / III: forward and backward propagation from a freshly
  // merged node.  Recursion mirrors Listing 1 Lines 22-24 / 26-28; the
  // "current op" we hand to the next-level mapping check is the most
  // recent addition, not the seed, matching the paper's per-edge
  // pairwise check.
  std::function<void(size_t)> walk_succ = [&](size_t op_idx) {
    auto* node = graph.post_dfs_order[op_idx];
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      size_t s = link->value.node->index;
      if (safe_to_merge_succ(op_idx, s) && try_fuse_edge(op_idx, s, /*forward=*/true)) {
        walk_succ(s);
      }
    }
  };
  std::function<void(size_t)> walk_pred = [&](size_t op_idx) {
    for (size_t p : preds[op_idx]) {
      if (try_fuse_edge(op_idx, p, /*forward=*/false)) {
        walk_pred(p);
      }
    }
  };

  // --- Main outer loop (Listing 1 Lines 30-41).
  // seed_done marks OtO nodes already used as seeds so pick_seed() doesn't
  // return them again (the seed stays its own group root, so the group-root
  // check alone is not sufficient to prevent re-selection).
  std::vector<bool> seed_done(n, false);
  auto pick_seed_once = [&]() -> size_t {
    size_t best = std::numeric_limits<size_t>::max();
    int64_t best_bytes = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < n; ++i) {
      if (seed_done[i]) continue;
      if (!eligible(i)) continue;
      if (mapping_type[i] != MappingType::kOneToOne) continue;
      if (groups_[i]->FindRoot() != groups_[i]) continue;
      int64_t b = irs_bytes[i];
      int64_t key = (b < 0) ? std::numeric_limits<int64_t>::max() : b;
      if (key < best_bytes || (key == best_bytes && i < best)) {
        best_bytes = key;
        best = i;
      }
    }
    return best;
  };
  while (true) {
    size_t seed = pick_seed_once();
    if (seed == std::numeric_limits<size_t>::max()) break;
    seed_done[seed] = true;
    walk_succ(seed);
    walk_pred(seed);
  }
}

}  // namespace relax
}  // namespace tvm
