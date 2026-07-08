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

// Shared test scaffolding for the RunDNNFuse unit tests
// (relax_dnnfuse_run_test.cc, relax_dnnfuse_seed_test.cc).
//
// FuseSuccessor / FusePredecessor are private to GraphPartitioner, so those
// tests drive them through the public Partition() entry: opt_level==6 routes
// to RunDNNFuse. GraphBuilder hand-builds a small IndexedForwardGraph; the
// tests partition it and assert which nodes ended up unioned into the same
// group. FindMinOtO is declared in graph_partitioner.h and is called directly
// (the seed tests use only AddNode, for its index-assignment rule).

#ifndef TVM_TESTS_CPP_RELAX_DNNFUSE_GRAPH_BUILDER_H_
#define TVM_TESTS_CPP_RELAX_DNNFUSE_GRAPH_BUILDER_H_

#include <tvm/relax/op_attr_types.h>

#include <vector>

#include "../../src/relax/analysis/graph_partitioner.h"
#include "../../src/support/arena.h"

namespace relax_dnnfuse_test {

using namespace tvm;
using namespace tvm::relax;

// Minimal builder for an IndexedForwardGraph of nodes wired by directed edges.
// Nodes carry a real ref of nullptr -- the opt_level==6 path never consults
// node_map, and a null ObjectRef prints as "(nullptr)" in the debug dumps, so
// no concrete Relax expressions are required.
class GraphBuilder {
 public:
  // Add a node with the given mapping_type/output size; its index is its
  // position in post_dfs_order, matching the indexing
  // GraphPartitioner::groups_ relies on. Every node gets OpPatternKind
  // kElemWise so the structural CheckPath fcond in FuseSuccessor /
  // FusePredecessor never itself blocks a fusion; only `mapping_type` -- which
  // drives both seed selection (FindMinOtO looks for kOneToOne) and
  // DNNFuseRelation::Classify (DNNFusion Table 3) -- is under test.
  IndexedForwardGraph::Node* AddNode(MappingType mapping_type, int64_t output_size) {
    auto* node = arena_.make<IndexedForwardGraph::Node>();
    node->index = graph_.post_dfs_order.size();
    node->pattern = kElemWise;
    node->mapping_type = mapping_type;
    node->output_size = output_size;
    graph_.post_dfs_order.push_back(node);
    return node;
  }

  // Add a producer -> consumer data edge (forward on src, backward on dst).
  void AddEdge(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* dst) {
    auto* out = arena_.make<support::LinkNode<IndexedForwardGraph::Edge>>();
    out->value.node = dst;
    out->value.pattern = src->pattern;
    src->outputs.Push(out);

    auto* in = arena_.make<support::LinkNode<IndexedForwardGraph::Edge>>();
    in->value.node = src;
    in->value.pattern = src->pattern;
    dst->inputs.Push(in);
  }

  // Run the DNNFuse partition path (opt_level == 6). The returned Group objects
  // are allocated from part_arena_, kept alive by this builder so the caller can
  // inspect FindRoot() after the call returns.
  std::vector<GraphPartitioner::Group*> RunDNNFuse() {
    GraphPartitioner partitioner(&part_arena_, /*opt_level=*/6, /*max_fuse_depth=*/256,
                                 /*max_function_args=*/1024);
    return partitioner.Partition(graph_);
  }

 private:
  support::Arena arena_;
  support::Arena part_arena_;
  IndexedForwardGraph graph_;
};

// Whether two nodes were fused into the same union-find group.
inline bool SameGroup(const std::vector<GraphPartitioner::Group*>& groups,
                      IndexedForwardGraph::Node* a, IndexedForwardGraph::Node* b) {
  return groups[a->index]->FindRoot() == groups[b->index]->FindRoot();
}

}  // namespace relax_dnnfuse_test

#endif  // TVM_TESTS_CPP_RELAX_DNNFUSE_GRAPH_BUILDER_H_
