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
// minimum-output One-to-One node (FindMinOtO) and walks forward via
// FuseSuccessor (Node::outputs) and backward via FusePredecessor (Node::inputs).
// We hand-build a small IndexedForwardGraph, partition it, then assert which
// nodes ended up unioned into the same group. (Seed selection itself is
// covered separately in relax_dnnfuse_seed_test.cc.)

#include <gtest/gtest.h>

#include "relax_dnnfuse_graph_builder.h"

using namespace tvm;
using namespace tvm::relax;
using relax_dnnfuse_test::GraphBuilder;
using relax_dnnfuse_test::SameGroup;

// --- FuseSuccessor: forward expansion along Node::outputs ------------------

// A One-to-One seed fuses straight through a forward One-to-One chain, with
// the recursion carrying past the first hop (Table 3: O2O row is all-green).
TEST(RunDNNFuseSuccessor, FusesOneToOneChainForward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kOneToOne, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
}

// A kMappingOpaque successor classifies as kFuseBreak, so the seed is not
// fused with it (op fell outside the Table 2 lookup).
TEST(RunDNNFuseSuccessor, StopsAtOpaqueMappingTypeSink) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);
  auto* n1 = b.AddNode(kMappingOpaque, 20);
  b.AddEdge(n0, n1);

  auto groups = b.RunDNNFuse();
  EXPECT_FALSE(SameGroup(groups, n0, n1));
}

// In the recursion, a Many-to-Many producer feeding a Many-to-Many consumer
// is Table 3's explicit unprofitable "x" cell: the chain fuses up to the
// first Many-to-Many node, then stops before the second.
TEST(RunDNNFuseSuccessor, StopsAtManyToManyIntoManyToManyBreakInChain) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);     // seed
  auto* n1 = b.AddNode(kManyToMany, 20);   // O2O -> M2M classifies through
  auto* n2 = b.AddNode(kManyToMany, 30);   // M2M -> M2M classifies break
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_FALSE(SameGroup(groups, n0, n2));
}

// kFuseDepend is profit-gated and currently bails out. Reorganize ->
// One-to-Many classifies as depend, so the recursion stops there even though
// it is not a hard break.
TEST(RunDNNFuseSuccessor, BailsOnReorganizeIntoOneToManyDependInChain) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);    // seed
  auto* n1 = b.AddNode(kReorganize, 20);  // O2O -> Reorganize classifies through
  auto* n2 = b.AddNode(kOneToMany, 30);   // Reorganize -> O2M classifies depend
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_FALSE(SameGroup(groups, n0, n2));
}

// --- FusePredecessor: backward expansion along Node::inputs ----------------

// When the seed is the tail of a One-to-One chain, the backward walk fuses
// the predecessors, recursing past the first hop.
TEST(RunDNNFusePredecessor, FusesOneToOneChainBackward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 30);
  auto* n1 = b.AddNode(kOneToOne, 20);
  auto* n2 = b.AddNode(kOneToOne, 5);  // smallest output -> seed; no successors
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n2, n1));
  EXPECT_TRUE(SameGroup(groups, n2, n0));
}

// A Many-to-Many predecessor still fuses into a One-to-One seed: Table 3's
// One-to-One column is all-green, mirroring the paper's own example of
// fusing Add and GEMM in either order.
TEST(RunDNNFusePredecessor, FusesManyToManyProducerIntoOneToOneSeed) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kManyToMany, 30);
  auto* n1 = b.AddNode(kOneToOne, 10);  // seed
  b.AddEdge(n0, n1);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n1, n0));
}

// A kMappingOpaque predecessor classifies as kFuseBreak and is left out of
// the group.
TEST(RunDNNFusePredecessor, StopsAtOpaqueMappingTypeSource) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kMappingOpaque, 30);
  auto* n1 = b.AddNode(kOneToOne, 10);  // seed
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
// An all-One-to-One diamond fuses into a single group. The seed has two
// outgoing edges: CommitFuse(seed, branch) merges the path to that branch, then
// FuseSuccessor recurses into the branch's outputs, so the join is reached and
// fused through both branches. The multi-output cone walk in CheckPath_ /
// CommitFuse_ (now restricted to the path to each sink) is what is under test.
/*
        n0 (seed)
        /      \
      n1        n2
        \      /
          n3 (join)
*/

TEST(RunDNNFuseSuccessor, DiamondAllOneToOneFusesEntirely) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kOneToOne, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  auto* n3 = b.AddNode(kOneToOne, 40);
  b.AddEdge(n0, n1);
  b.AddEdge(n0, n2);
  b.AddEdge(n1, n3);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
  EXPECT_TRUE(SameGroup(groups, n0, n3));
}

// Same diamond, but the join is kMappingOpaque. CheckPath / CommitFuse restrict
// the cone walk to the path that actually terminates at the sink, so fusing
// seed->branch no longer inspects (or pulls in) the opaque join reachable
// through the *sibling* branch: seed->n1 only sees n1, seed->n2 only sees n2.
// Both branches classify as through and fuse, so the One-to-One body
// {n0,n1,n2} collapses into one group while the opaque join n3 -- left out as
// a downstream consumer -- stays separate. The resulting partition is still
// convex (n3 only consumes from the group, never feeds back), which is why
// excluding it is legal.
/*
        n0 (seed)
        /      \
      n1        n2
        \      /
        n3 (opaque join)
*/

TEST(RunDNNFuseSuccessor, DiamondOpaqueMappingTypeJoinFusesBody) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // seed
  auto* n1 = b.AddNode(kOneToOne, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  auto* n3 = b.AddNode(kMappingOpaque, 40);  // opaque join, excluded
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
// over the producers' inputs). All One-to-One -> the diamond fuses entirely.
/*
           n0
        /      \
      n1        n2
        \      /
        n3 (seed)
*/

TEST(RunDNNFusePredecessor, DiamondSeedAtJoinFusesEntirely) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 40);
  auto* n1 = b.AddNode(kOneToOne, 30);
  auto* n2 = b.AddNode(kOneToOne, 20);
  auto* n3 = b.AddNode(kOneToOne, 5);  // smallest output -> seed; the join node
  b.AddEdge(n0, n1);
  b.AddEdge(n0, n2);
  b.AddEdge(n1, n3);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n3, n1));
  EXPECT_TRUE(SameGroup(groups, n3, n2));
  EXPECT_TRUE(SameGroup(groups, n3, n0));
}
