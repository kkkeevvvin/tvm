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

namespace {
enum class DNNFuseRelation {
  kFuseThrough,
  kFuseBreak,
  kFuseDepend,
};

const char* DNNFuseRelationName(DNNFuseRelation decision) {
  switch (decision) {
    case DNNFuseRelation::kFuseThrough:
      return "fuse_through";
    case DNNFuseRelation::kFuseBreak:
      return "fuse_break";
    case DNNFuseRelation::kFuseDepend:
      return "fuse_depend";
  }
  return "unknown";
}

DNNFuseRelation ClassifyDNNFuseRelation(OpPatternKind producer, OpPatternKind consumer) {
  if (producer == kOutEWiseFusable && (consumer == kBroadcast || consumer == kInjective)) {
    return DNNFuseRelation::kFuseDepend;
  }
  if (producer == kCommReduce && (consumer == kBroadcast || consumer == kInjective)) {
    return DNNFuseRelation::kFuseDepend;
  }
  if (producer == kBroadcast && consumer == kInjective) {
    return DNNFuseRelation::kFuseDepend;
  }
  if (producer > kBroadcast && consumer > kBroadcast) {
    return DNNFuseRelation::kFuseBreak;
  }
  return DNNFuseRelation::kFuseThrough;
}

IndexedForwardGraph::Node* FindMinElemWise(
    const std::unordered_set<IndexedForwardGraph::Node*>& unfused_ops) {
  IndexedForwardGraph::Node* min_node = nullptr;
  for (IndexedForwardGraph::Node* node : unfused_ops) {
    if (node->pattern != kElemWise) continue;
    if (node->output_size < 0) continue;
    if (min_node == nullptr || node->output_size < min_node->output_size ||
        (node->output_size == min_node->output_size && node->index < min_node->index)) {
      min_node = node;
    }
  }
  return min_node;
}
}  // namespace

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
  Group* group = groups_[src->index];
  ICHECK(group != nullptr);
  group = group->FindRoot();
  if (!fcond(group->pattern, src == sink)) return false;
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
  Group* group = groups_[src->index];
  ICHECK(group != nullptr);
  // merge the current group to the parent if possible.
  MergeFromTo(group, target);
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

template <typename F>
void GraphPartitioner::TryFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink,
                               F fcond) {
  if (CheckPath(src, sink, fcond)) {
    CommitFuse(src, sink);
  }
}

size_t GraphPartitioner::CountNodesUptoSink_(IndexedForwardGraph::Node* src,
                                             IndexedForwardGraph::Node* sink) {
  if (src == sink || visited_.count(src)) return 0;
  visited_.insert(src);
  Group* group = groups_[src->index];
  ICHECK(group != nullptr);
  auto sum = group->num_nodes;
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
  Group* group = groups_[src->index];
  ICHECK(group != nullptr);
  auto sum = group->args_num;
  visited_groups.insert(group->FindRoot());
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
    auto* group = arena_->make<Group>();
    group->pattern = graph_node->pattern;
    group->root_ref = graph_node->ref;
    // set anchor ref if necessary.
    if (group->pattern == kOutEWiseFusable) {
      group->anchor_ref = graph_node->ref;
    }
    group->args_num = args_counter(graph_node->ref);
    groups_[nid] = group;
  }
}

void GraphPartitioner::ProcessPostponedFusing(IndexedForwardGraph::Node* graph_node,
                                              const IndexedForwardGraph& graph,
                                              const DominatorTree& post_dom_tree) {
  if (!postponed_fusing_map_.count(graph_node)) return;
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

void GraphPartitioner::FuseInjectiveIntoTuple(IndexedForwardGraph::Node* graph_node,
                                              Group* group, DominatorTree::Node* dom_node,
                                              size_t dom_parent_index) {
  if (group->pattern > kInjective) return;
  Group* dom_parent_group = groups_[dom_parent_index];
  Group* dom_root_group = dom_parent_group->FindRoot();
  // If dom node group has a tuple as its root, we do not fuse tuple fields into it
  if (dom_root_group->pattern == kTuple) return;
  if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
    // Now we know the tuple has been fused into subsequent injective ops.
    // dom_root_group can also be tuple, as in inception layers — TryFuse's
    // CheckPath is needed to avoid fusing two intermediate tuples.
    auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
    TryFuse(graph_node, dom_node->parent->gnode, fcond);
  }
}

void GraphPartitioner::FuseToPostDominator(IndexedForwardGraph::Node* graph_node,
                                           Group* group, DominatorTree::Node* dom_node,
                                           size_t dom_parent_index, int phase) {
  // dom_node->parent and its gnode are guaranteed non-null by RunFuse's caller-side guard.
  IndexedForwardGraph::Node* dom_parent_gnode = dom_node->parent->gnode;
  ICHECK(dom_parent_gnode != nullptr);

  // Skip if current node is already fused to the parent.
  if (groups_[dom_parent_index] != nullptr &&
      group->FindRoot() == groups_[dom_parent_index]->FindRoot()) {
    return;
  }
  // Do not fuse into tuple for now.
  if (groups_[dom_parent_index]->pattern == kTuple) return;

  switch (group->pattern) {
    case kOutEWiseFusable: {
      // OutEWiseFusable (e.g. conv2d) fuses in phase 0, only when the dominator
      // relation is elemwise and all intermediate ops are still broadcast.
      if (phase != 0) return;
      if (dom_node->pattern != kElemWise) return;
      auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
      TryFuse(graph_node, dom_parent_gnode, fcond);
      return;
    }
    case kElemWise:
    case kBroadcast: {
      // Elemwise/broadcast can fuse to an injective or reduction parent.
      // Intermediate ops on parallel branches stay <= injective; the sink may
      // already be fused to a kOutEWiseFusable / kCommReduce / kInjective anchor.
      if (dom_node->pattern > kInjective && dom_node->pattern != kCommReduce) return;
      auto fcond = [](OpPatternKind kind, bool is_sink) {
        if (!is_sink) return kind <= kInjective;
        return kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
               kind == kOutEWiseFusable;
      };
      TryFuse(graph_node, dom_parent_gnode, fcond);
      return;
    }
    case kInjective:
    case kTuple: {
      // Deferred to phase 1 so conv2d (phase 0) finishes fusing first.
      if (phase != 1) return;
      auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
      TryFuse(graph_node, dom_parent_gnode, fcond);
      return;
    }
    case kCommReduce:
      return;
    default:
      LOG(FATAL) << "FuseToPostDominator: unexpected pattern " << group->pattern;
  }
}

void GraphPartitioner::RunFuse(const IndexedForwardGraph& graph,    //
                               const DominatorTree& post_dom_tree,  //
                               int phase) {
  for (size_t nid = 0; nid < groups_.size(); ++nid) {
    // the group of current node has been specified already.
    auto* graph_node = graph.post_dfs_order[nid];
    auto* dom_node = post_dom_tree.nodes[nid];
    Group* group = groups_[nid];
    ICHECK(group != nullptr);
    postpone_node_ = nullptr;

    ProcessPostponedFusing(graph_node, graph, post_dom_tree);

    // no actions for opaque nodes
    if (group->pattern == kOpaque) continue;
    // no actions needed if the current node have no dominator
    if (dom_node->parent == nullptr) continue;
    ICHECK(!graph_node->extern_ref);
    size_t dom_parent_index = dom_node->parent->gnode->index;

    // refuse the fusion if too many ops are going to be fused together
    if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
      continue;
    // Refuse the fusion if too many arguments are going to be in the fused function
    if (max_function_args_ > 0) {
      auto limit = CountArgsLimit_(graph_node);
      if (limit > 0 && CountFusedArgs(graph, graph_node) > limit) {
        continue;
      }
    }

    if (phase == 2) {
      FuseInjectiveIntoTuple(graph_node, group, dom_node, dom_parent_index);
      continue;
    }

    FuseToPostDominator(graph_node, group, dom_node, dom_parent_index, phase);
  }
}

void GraphPartitioner::FuseSuccessor(IndexedForwardGraph::Node* sp,
                                     IndexedForwardGraph::Node* successor,
                                     std::unordered_set<IndexedForwardGraph::Node*>* block) {
  LOG(INFO) << "  successor of node[" << sp->index << "]:"
            << " node[" << successor->index << "] " << ffi::GetRef<ObjectRef>(successor->ref)
            << " (pattern=" << successor->pattern << ", bytes=" << successor->output_size << ")";
  // check the mapping relationship
  OpPatternKind sp_pat = groups_[sp->index]->FindRoot()->pattern;
  OpPatternKind succ_pat = groups_[successor->index]->FindRoot()->pattern;
  DNNFuseRelation relation = ClassifyDNNFuseRelation(sp_pat, succ_pat);
  LOG(INFO) << "    relation = " << DNNFuseRelationName(relation);
  // return if successor can not be fused
  if (relation == DNNFuseRelation::kFuseBreak) return;
  // Check TVM path/codegen constraints before applying fuse_depend policy.
  auto fcond = [](OpPatternKind kind, bool is_sink) {
    if (is_sink) return kind <= kOutEWiseFusable;
    return kind <= kInjective;
  };
  if (!CheckPath(sp, successor, fcond)) {
    LOG(INFO) << "    CheckPath: false - Skip CommitFuse";
    return;
  }
  if (relation == DNNFuseRelation::kFuseDepend) return;
  LOG(INFO) << "    CheckPath: true - CommitFuse";
  CommitFuse(sp, successor);
  block->insert(successor);
  for (auto* link = successor->outputs.head; link != nullptr; link = link->next) {
    FuseSuccessor(successor, link->value.node, block);
  }
}

void GraphPartitioner::FusePredecessor(IndexedForwardGraph::Node* sp,
                                       IndexedForwardGraph::Node* predecessor,
                                       std::unordered_set<IndexedForwardGraph::Node*>* block) {
  LOG(INFO) << "  predecessor of node[" << sp->index << "]:"
            << " node[" << predecessor->index << "] "
            << ffi::GetRef<ObjectRef>(predecessor->ref)
            << " (pattern=" << predecessor->pattern
            << ", bytes=" << predecessor->output_size << ")";
  // check the mapping relationship
  OpPatternKind sp_pat = groups_[sp->index]->FindRoot()->pattern;
  OpPatternKind pred_pat = groups_[predecessor->index]->FindRoot()->pattern;
  DNNFuseRelation relation = ClassifyDNNFuseRelation(pred_pat, sp_pat);
  LOG(INFO) << "    relation = " << DNNFuseRelationName(relation);
  // return if predecessor can not be fused
  if (relation == DNNFuseRelation::kFuseBreak) return;
  // Check TVM path/codegen constraints before applying fuse_depend policy.
  auto fcond = [](OpPatternKind kind, bool is_sink) {
    if (is_sink) return kind <= kOutEWiseFusable;
    return kind <= kInjective;
  };
  if (!CheckPath(predecessor, sp, fcond)) {
    LOG(INFO) << "    CheckPath: false - Skip";
    return;
  }
  if (relation == DNNFuseRelation::kFuseDepend) return;
  LOG(INFO) << "    CheckPath: true - CommitFuse";
  CommitFuse(predecessor, sp);
  block->insert(predecessor);
  for (auto* link = predecessor->inputs.head; link != nullptr; link = link->next) {
    FusePredecessor(predecessor, link->value.node, block);
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
  while (seed = FindMinElemWise(unfused_ops)) {
    // block = [ seed ]
    std::unordered_set<IndexedForwardGraph::Node*> block{seed};
    LOG(INFO) << "\nkElemWise op with min output_size:"
              << " node[" << seed->index << "], " << ffi::GetRef<ObjectRef>(seed->ref)
              << " bytes=" << seed->output_size;
    // head to successor
    for (auto* link = seed->outputs.head; link != nullptr; link = link->next) {
      FuseSuccessor(seed, link->value.node, &block);
    }
    // head to predecessor
    for (auto* link = seed->inputs.head; link != nullptr; link = link->next) {
      FusePredecessor(seed, link->value.node, &block);
    }
    // unfused_ops = unfused_ops - block
    for (IndexedForwardGraph::Node* op : block) unfused_ops.erase(op);
  }
}

}  // namespace relax
}  // namespace tvm
