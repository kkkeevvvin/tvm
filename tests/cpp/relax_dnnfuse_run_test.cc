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
// The chain tests above keep every node single-input/single-output. These
// exercise fan-out, where CommitFuseEdge's edge-locality matters: each merge
// touches exactly the two groups joined by the edge under consideration, so
// sibling branches neither veto nor get dragged into an unrelated fusion.
//
// An all-One-to-One diamond fuses into a single group. The seed has two
// outgoing edges: each fuses edge-locally, then FuseSuccessor recurses into
// the branch's outputs, so the join is reached and fused through both
// branches (the second arrival is a same-root no-op).
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

// Same diamond, but the join is kMappingOpaque. CommitFuseEdge is edge-local,
// so fusing seed->branch never inspects (or pulls in) the opaque join
// reachable through the *sibling* branch: seed->n1 merges only n0 and n1's
// groups, seed->n2 only n0's and n2's. Both branches classify as through and
// fuse, so the One-to-One body {n0,n1,n2} collapses into one group while the
// opaque join n3 -- rejected on its own edges by the relation break -- stays
// separate. The resulting partition is still convex (n3 only consumes from
// the group, never feeds back), which is why excluding it is legal.
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

// --- Derived group MappingType drives the next hop --------------------------
//
// CommitFuseEdge stamps the surviving root with Table 3's fused type
// (relation.FusedType()), so later edges classify against the group's evolved
// type rather than the stale InitGroups value of whichever node happens to be
// the union-find root.

// Forward: O2O -> M2M -> O2O -> M2M. The second hop merges into n2's group
// (initialized One-to-One) but derives Many-to-Many, so the third edge
// classifies as (M2M, M2M) = break and n3 stays out. Against the stale root
// type the edge would classify (O2O, M2M) = through and over-fuse.
TEST(RunDNNFuseSuccessor, DerivedGroupTypeStopsChainForward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kManyToMany, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  auto* n3 = b.AddNode(kManyToMany, 40);
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);
  b.AddEdge(n2, n3);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
  EXPECT_FALSE(SameGroup(groups, n0, n3));
  EXPECT_EQ(groups[n0->index]->FindRoot()->mapping_type, kManyToMany);
}

// The backward mirror: M2M -> M2M -> O2O(seed). The first backward hop merges
// n1 into the seed's group (initialized One-to-One) and derives Many-to-Many,
// so the next edge classifies as (M2M, M2M) = break and n0 stays out. Against
// the stale root type it would classify (M2M, O2O) = through and over-fuse.
TEST(RunDNNFusePredecessor, DerivedGroupTypeStopsChainBackward) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kManyToMany, 30);
  auto* n1 = b.AddNode(kManyToMany, 20);
  auto* n2 = b.AddNode(kOneToOne, 5);  // the only One-to-One -> seed
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n2, n1));
  EXPECT_FALSE(SameGroup(groups, n2, n0));
  EXPECT_EQ(groups[n2->index]->FindRoot()->mapping_type, kManyToMany);
}

// --- Residual (skip-edge) topologies: the CheckEdgeConvexity gate -----------
//
// A residual connection gives the walk a direct edge n0 -> n2 alongside a
// longer path n0 -> n1 -> n2. Fusing across the skip edge while n1 stays
// outside would make the group non-convex ({n0,n2} both feeds and consumes
// n1's path), which OperatorFuser cannot serialize into a fused function --
// exactly the mobilenet_v2 / resnet18 InternalError from issue #12's all_6
// run. CheckEdgeConvexity rejects such merges; the edge is retried naturally
// if a later merge brings the branch inside.

// Forward walk tries the skip edge first (it is n0's first output). The branch
// node is opaque, so it can never join the group: the skip merge must be
// rejected outright, in both the seed's forward walk and the follow-up seed
// n2's backward walk.
/*
      n0 (seed) ── n1 (opaque) ── n2
        └───────── skip ─────────┘
*/
TEST(RunDNNFuseSuccessor, ResidualSkipAroundOpaqueBranchStaysUnfused) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kMappingOpaque, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  b.AddEdge(n0, n2);  // skip edge first, so the walk tries it before the branch
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_FALSE(SameGroup(groups, n0, n2));
  EXPECT_FALSE(SameGroup(groups, n0, n1));
  EXPECT_FALSE(SameGroup(groups, n1, n2));
}

// The backward mirror: the seed sits at the join, so FusePredecessor tries the
// skip edge (n2's first input) before the branch. Same rejection.
/*
      n0 ── n1 (opaque) ── n2 (seed)
      └────── skip ────────┘
*/
TEST(RunDNNFusePredecessor, ResidualSkipAroundOpaqueBranchStaysUnfused) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 30);
  auto* n1 = b.AddNode(kMappingOpaque, 20);
  auto* n2 = b.AddNode(kOneToOne, 5);  // smallest output -> seed; the join node
  b.AddEdge(n0, n2);  // skip edge first, so the walk tries it before the branch
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_FALSE(SameGroup(groups, n2, n0));
  EXPECT_FALSE(SameGroup(groups, n2, n1));
}

// The rejection is per-attempt, not permanent: with a fusable branch, the skip
// merge is refused on the first try (n1 still outside), the branch then fuses
// n0 -> n1, and the recursion's n1 -> n2 merge passes the convexity check --
// after which the skip edge is internal. The whole residual block collapses
// into one group.
/*
      n0 (seed) ── n1 (O2O) ── n2
        └──────── skip ───────┘
*/
TEST(RunDNNFuseSuccessor, ResidualSkipFusesOnceBranchJoins) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 10);  // smallest output -> seed
  auto* n1 = b.AddNode(kOneToOne, 20);
  auto* n2 = b.AddNode(kOneToOne, 30);
  b.AddEdge(n0, n2);  // skip edge first, so the walk tries it before the branch
  b.AddEdge(n0, n1);
  b.AddEdge(n1, n2);

  auto groups = b.RunDNNFuse();
  EXPECT_TRUE(SameGroup(groups, n0, n1));
  EXPECT_TRUE(SameGroup(groups, n0, n2));
}
