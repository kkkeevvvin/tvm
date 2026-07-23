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

#include "./dnnf_partitioner.h"

#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

// IndexedForwardGraph::DebugDump() lives here rather than as an inline method
// in graph_partitioner.h: its only caller is RunDNNFuse below, and its body
// calls MappingTypeName(), which is declared in this DNNFuse-only header.
// Keeping the inline body in graph_partitioner.h would need this header
// included there, creating a cycle (this header already includes
// graph_partitioner.h for IndexedForwardGraph / GraphPartitioner::Group).
void IndexedForwardGraph::DebugDump() const {
  std::ostringstream os;
  for (size_t i = 0; i < post_dfs_order.size(); ++i) {
    Node* node = post_dfs_order[i];
    std::string bytes_str = node->output_size < 0 ? "?" : std::to_string(node->output_size);
    os << "node[" << i << "], " << ffi::GetRef<ObjectRef>(node->ref)
       << " mapping=" << MappingTypeName(node->mapping_type) << " bytes=" << bytes_str
       << " outputs=[";
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      os << link->value.node->index << ", ";
    }
    os << "]\n";
  }
  LOG(INFO) << "\nIndexedForwardGraph: " << post_dfs_order.size() << " nodes\n" << os.str();
}

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

std::vector<GraphPartitioner::Group*> DNNFGraphPartitioner::Partition(
    const IndexedForwardGraph& graph, int64_t profile_tune_trials) {
  profile_tune_trials_ = profile_tune_trials;
  this->InitGroups(graph);
  this->RunDNNFuse(graph);
  return std::move(groups_);
}

void DNNFGraphPartitioner::InitGroups(const IndexedForwardGraph& graph) {
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
    auto* group_node = arena_->make<GraphPartitioner::Group>();
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

bool DNNFGraphPartitioner::CheckEdgeConvexity(const IndexedForwardGraph& graph,
                                              IndexedForwardGraph::Node* src,
                                              IndexedForwardGraph::Node* sink) {
  GraphPartitioner::Group* src_root = groups_[src->index]->FindRoot();
  GraphPartitioner::Group* sink_root = groups_[sink->index]->FindRoot();
  if (src_root == sink_root) return true;
  const size_t num_nodes = graph.post_dfs_order.size();
  // Membership masks of the two groups about to merge.
  std::vector<bool> in_src(num_nodes, false);
  std::vector<bool> in_sink(num_nodes, false);
  for (size_t i = 0; i < num_nodes; ++i) {
    GraphPartitioner::Group* root = groups_[i]->FindRoot();
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

void DNNFGraphPartitioner::CommitFuseEdge(IndexedForwardGraph::Node* src,
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
  GraphPartitioner::Group* src_root = groups_[src->index]->FindRoot();
  GraphPartitioner::Group* sink_root = groups_[sink->index]->FindRoot();
  if (src_root == sink_root) return;
  sink_root->num_nodes += src_root->num_nodes;
  sink_root->args_num += src_root->args_num;
  src_root->parent = sink_root;
  // Table 3's cell value: the mapping type of the operator after fusion.
  sink_root->mapping_type = relation.FusedType();
}

void DNNFGraphPartitioner::FuseSuccessor(const IndexedForwardGraph& graph,
                                         IndexedForwardGraph::Node* sp,
                                         IndexedForwardGraph::Node* successor,
                                         std::unordered_set<IndexedForwardGraph::Node*>* block) {
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
  // TODO: kFuseDepend fusion is profit-gated -- bail out until the profiler is implemented
  if (relation.IsDepend()) return;
  // Structural gate: skip merges that would leave a src->sink path outside the
  // group (e.g. a residual skip edge fused around its own branch). The edge may
  // become fusable later, once the branch nodes have joined either group.
  if (!CheckEdgeConvexity(graph, sp, successor)) {
    LOG(INFO) << "    skipped: merge would break group convexity";
    return;
  }
  CommitFuseEdge(sp, successor, relation);
  block->insert(successor);
  // Recurse into the fused successor to extend the chain past one hop.
  for (auto* link = successor->outputs.head; link != nullptr; link = link->next) {
    FuseSuccessor(graph, successor, link->value.node, block);
  }
}

void DNNFGraphPartitioner::FusePredecessor(const IndexedForwardGraph& graph,
                                           IndexedForwardGraph::Node* sp,
                                           IndexedForwardGraph::Node* predecessor,
                                           std::unordered_set<IndexedForwardGraph::Node*>* block) {
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
  // TODO: kFuseDepend fusion is profit-gated -- bail out until the profiler is implemented
  if (relation.IsDepend()) return;
  // Structural gate: mirror of FuseSuccessor's convexity check, oriented along
  // the direct edge predecessor -> sp.
  if (!CheckEdgeConvexity(graph, predecessor, sp)) {
    LOG(INFO) << "    skipped: merge would break group convexity";
    return;
  }
  CommitFuseEdge(predecessor, sp, relation);
  block->insert(predecessor);
  // Recurse into the fused predecessor to extend the chain past one hop.
  for (auto* link = predecessor->inputs.head; link != nullptr; link = link->next) {
    FusePredecessor(graph, predecessor, link->value.node, block);
  }
}

void DNNFGraphPartitioner::RunDNNFuse(const IndexedForwardGraph& graph) {
  graph.DebugDump();
  // unfused_ops = all_operaters
  std::unordered_set<IndexedForwardGraph::Node*> unfused_ops(graph.post_dfs_order.begin(),
                                                              graph.post_dfs_order.end());
  LOG(INFO) << "unfused_ops: " << unfused_ops.size() << " nodes";
  IndexedForwardGraph::Node* seed = nullptr;
  // generate seed
  while ((seed = FindMinOtO(unfused_ops)) != nullptr) {
    // block = [ seed ]
    std::unordered_set<IndexedForwardGraph::Node*> block{seed};
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

}  // namespace relax
}  // namespace tvm
