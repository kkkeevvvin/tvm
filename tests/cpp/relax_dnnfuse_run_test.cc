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

// Unit tests for GraphPartitioner::FuseSuccessor / FusePredecessor, the
// bidirectional expansion of RunDNNFuse (the opt_level==6 fusion path).
//
// Both functions are private, so the tests drive them through the public
// Partition() entry: opt_level==6 routes to RunDNNFuse, which seeds from the
// minimum-output element-wise node (FindMinElemWise) and walks forward via
// FuseSuccessor (Node::outputs) and backward via FusePredecessor (Node::inputs).
// We hand-build a small IndexedForwardGraph, partition it, then assert which
// nodes ended up unioned into the same group.

#include <gtest/gtest.h>
#include <tvm/relax/op_attr_types.h>

#include <vector>

#include "../../src/relax/analysis/graph_partitioner.h"
#include "../../src/support/arena.h"

using namespace tvm;
using namespace tvm::relax;

namespace {

// Minimal builder for an IndexedForwardGraph of nodes wired by directed edges.
// Nodes carry a real ref of nullptr -- the opt_level==6 path never consults
// node_map, and a null ObjectRef prints as "(nullptr)" in the debug dumps, so
// no concrete Relax expressions are required.
class GraphBuilder {
 public:
  // Add a node with the given pattern/output size; its index is its position in
  // post_dfs_order, matching the indexing GraphPartitioner::groups_ relies on.
  IndexedForwardGraph::Node* AddNode(OpPatternKind pattern, int64_t output_size) {
    auto* node = arena_.make<IndexedForwardGraph::Node>();
    node->index = graph_.post_dfs_order.size();
    node->pattern = pattern;
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
bool SameGroup(const std::vector<GraphPartitioner::Group*>& groups,
               IndexedForwardGraph::Node* a, IndexedForwardGraph::Node* b) {
  return groups[a->index]->FindRoot() == groups[b->index]->FindRoot();
}

}  // namespace

// --- FuseSuccessor: forward expansion along Node::outputs ------------------

// An element-wise seed fuses straight through a forward element-wise chain, with
// the recursion carrying past the first hop.
TEST(RunDNNFuseSuccessor, FusesElemWiseChainForward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);  // smallest output -> chosen as the seed
  auto* n1 = b.AddNode(kElemWise, 20);
  auto* n2 = b.AddNode(kElemWise, 30);
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
}

// An opaque successor classifies as kFuseBreak, so the seed is not fused with it.
TEST(RunDNNFuseSuccessor, StopsAtOpaqueSink) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);
  auto* n1 = b.AddNode(kOpaque, 20);
  b.AddEdge(n0, n1);

  auto groups = b.RunDNNFuse();
  EXPECT_FALSE(SameGroup(groups, n0, n1));
}

// In the recursion, a broadcast producer feeding a reduce consumer is a
// structural kFuseBreak: the chain fuses up to the broadcast, then stops.
TEST(RunDNNFuseSuccessor, StopsAtStructuralBreakInChain) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);   // seed
  auto* n1 = b.AddNode(kBroadcast, 20);  // elem -> broadcast classifies through
  auto* n2 = b.AddNode(kCommReduce, 30);  // broadcast -> reduce classifies break
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_FALSE(SameGroup(groups, n0, n2));
}

// kFuseDepend is profit-gated and currently bails out. injective -> broadcast
// classifies as depend, so the recursion stops there even though it is not a
// hard break.
TEST(RunDNNFuseSuccessor, BailsOnDependInChain) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);   // seed
  auto* n1 = b.AddNode(kInjective, 20);  // elem -> injective classifies through
  auto* n2 = b.AddNode(kBroadcast, 30);  // injective -> broadcast classifies depend
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_FALSE(SameGroup(groups, n0, n2));
}

// --- FusePredecessor: backward expansion along Node::inputs ----------------

// When the seed is the tail of an element-wise chain, the backward walk fuses
// the predecessors, recursing past the first hop.
TEST(RunDNNFusePredecessor, FusesElemWiseChainBackward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 30);
  auto* n1 = b.AddNode(kElemWise, 20);
  auto* n2 = b.AddNode(kElemWise, 5);  // smallest output -> seed; has no successors
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n2, n1));
  EXPECT_TRUE(SameGroup(groups, n2, n0));
}

// A non-element-wise predecessor still fuses into an element-wise seed: a
// broadcast feeding the seed classifies through (the sink is element-wise).
TEST(RunDNNFusePredecessor, FusesNonElemWiseProducerIntoSeed) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kBroadcast, 30);
  auto* n1 = b.AddNode(kElemWise, 10);  // seed
  b.AddEdge(n0, n1);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n1, n0));
}

// An opaque predecessor classifies as kFuseBreak and is left out of the group.
TEST(RunDNNFusePredecessor, StopsAtOpaqueSource) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOpaque, 30);
  auto* n1 = b.AddNode(kElemWise, 10);  // seed
  b.AddEdge(n0, n1);

  auto groups = b.RunDNNFuse();
  EXPECT_FALSE(SameGroup(groups, n1, n0));
}

// --- Diamond / multi-output topologies ------------------------------------
//
// The chain tests above keep every node single-input/single-output, so
// CheckPath_ / CommitFuse_ never recurse over branches. These exercise the
// forward (and backward) cone walk where a node fans out to -- or joins from --
// more than one neighbour.
//
// An all-element-wise diamond fuses into a single group. The seed has two
// outgoing edges: CommitFuse(seed, branch) merges the path to that branch, then
// FuseSuccessor recurses into the branch's outputs, so the join is reached and
// fused through both branches. The multi-output cone walk in CheckPath_ /
// CommitFuse_ (now restricted to the path to each sink) is what is under test.
//
//        n0 (seed)
//        /      \
//      n1        n2
//        \      /
//          n3 (join)

TEST(RunDNNFuseSuccessor, DiamondAllElemWiseFusesEntirely) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kElemWise, 20);
  auto* n2 = b.AddNode(kElemWise, 30);
  auto* n3 = b.AddNode(kElemWise, 40);
  b.AddEdge(n0, n1);
  b.AddEdge(n0, n2);
  b.AddEdge(n1, n3);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
  EXPECT_TRUE(SameGroup(groups, n0, n3));
}

// Same diamond, but the join is opaque. CheckPath / CommitFuse restrict the cone
// walk to the path that actually terminates at the sink, so fusing seed->branch
// no longer inspects (or pulls in) the opaque join reachable through the *sibling*
// branch: seed->n1 only sees n1, seed->n2 only sees n2. Both branches classify as
// through and fuse, so the element-wise body {n0,n1,n2} collapses into one group
// while the opaque join n3 -- left out as a downstream consumer -- stays separate.
// The resulting partition is still convex (n3 only consumes from the group, never
// feeds back), which is why excluding it is legal.
//
//        n0 (seed)
//        /      \
//      n1        n2
//        \      /
//        n3 (opaque join)

TEST(RunDNNFuseSuccessor, DiamondOpaqueJoinFusesElemWiseBody) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 10);  // seed
  auto* n1 = b.AddNode(kElemWise, 20);
  auto* n2 = b.AddNode(kElemWise, 30);
  auto* n3 = b.AddNode(kOpaque, 40);  // opaque join, excluded as a downstream consumer
  b.AddEdge(n0, n1);
  b.AddEdge(n0, n2);
  b.AddEdge(n1, n3);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
  EXPECT_TRUE(SameGroup(groups, n1, n2));
  EXPECT_FALSE(SameGroup(groups, n0, n3));
}

// The backward mirror: a diamond whose join is the seed, so FusePredecessor
// walks back through the join's two incoming edges (and the recursion fans out
// over the producers' inputs). All element-wise -> the diamond fuses entirely.
//
//           n0
//        /      \
//      n1        n2
//        \      /
//        n3 (seed)

TEST(RunDNNFusePredecessor, DiamondSeedAtJoinFusesEntirely) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kElemWise, 40);
  auto* n1 = b.AddNode(kElemWise, 30);
  auto* n2 = b.AddNode(kElemWise, 20);
  auto* n3 = b.AddNode(kElemWise, 5);  // smallest output -> seed; the join node
  b.AddEdge(n0, n1);
  b.AddEdge(n0, n2);
  b.AddEdge(n1, n3);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n3, n1));
  EXPECT_TRUE(SameGroup(groups, n3, n2));
  EXPECT_TRUE(SameGroup(groups, n3, n0));
}
