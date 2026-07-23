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

// Unit tests for FindMinOtO, the RunDNNFuse seed selector: among unfused
// kOneToOne nodes with a known (non-negative) output_size it returns the one
// with the smallest output, ties broken by the smaller node index.
//
// FindMinOtO is called directly on a hand-built candidate set -- no edges and
// no DNNFPartition() run are involved. GraphBuilder is used only to allocate nodes
// with the same index-assignment rule (index == insertion order) the real
// IndexedForwardGraph construction uses, which is what the tie-break tests
// depend on.

#include <gtest/gtest.h>

#include <unordered_set>

#include "relax_dnnfuse_graph_builder.h"

using namespace tvm;
using namespace tvm::relax;
using relax_dnnfuse_test::GraphBuilder;

using NodeSet = std::unordered_set<IndexedForwardGraph::Node*>;

// The smallest output_size wins even against a smaller index: a regression to
// index-order (or unordered_set iteration order) would return nA instead.
TEST(FindMinOtO, PicksMinOutputSizeOverSmallerIndex) {
  GraphBuilder b;
  auto* nA = b.AddNode(kOneToOne, 20);  // index 0
  auto* nB = b.AddNode(kOneToOne, 10);  // index 1, min output

  EXPECT_EQ(FindMinOtO(NodeSet{nA, nB}), nB);
}

// On an output_size tie the smaller index wins, keeping the selection
// deterministic regardless of unordered_set iteration order.
TEST(FindMinOtO, BreaksOutputSizeTieBySmallerIndex) {
  GraphBuilder b;
  auto* nA = b.AddNode(kOneToOne, 10);  // index 0
  auto* nB = b.AddNode(kOneToOne, 10);  // index 1

  EXPECT_EQ(FindMinOtO(NodeSet{nA, nB}), nA);
}

// One-to-One nodes carrying the -1 "unknown output size" sentinel are not
// eligible: with every candidate at -1 there is no seed.
TEST(FindMinOtO, SkipsUnknownOutputSizeSentinel) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, -1);
  auto* n1 = b.AddNode(kOneToOne, -1);

  EXPECT_EQ(FindMinOtO(NodeSet{n0, n1}), nullptr);
}

// The sentinel is a "skip" filter, not a plain minimum: -1 compares smaller
// than any known size, so a regression to an unfiltered min would return n0.
TEST(FindMinOtO, SentinelDoesNotWinAsSmallestValue) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, -1);
  auto* n1 = b.AddNode(kOneToOne, 10);

  EXPECT_EQ(FindMinOtO(NodeSet{n0, n1}), n1);
}

// output_size == 0 sits exactly on the eligibility boundary (the filter is
// `< 0`, not `<= 0`): a zero-output node is still a valid seed.
TEST(FindMinOtO, ZeroOutputSizeIsEligible) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kOneToOne, 0);   // boundary: still eligible
  auto* n1 = b.AddNode(kOneToOne, -1);  // ineligible

  EXPECT_EQ(FindMinOtO(NodeSet{n0, n1}), n0);
}

// Only kOneToOne nodes are considered; other mapping types never seed even
// with known output sizes.
TEST(FindMinOtO, RequiresOneToOneMappingType) {
  GraphBuilder b;
  auto* n0 = b.AddNode(kReorganize, 10);
  auto* n1 = b.AddNode(kShuffle, 20);

  EXPECT_EQ(FindMinOtO(NodeSet{n0, n1}), nullptr);
}

// A non-One-to-One node with the globally smallest output must not shadow the
// One-to-One candidate: the mapping_type filter runs before the min compare.
TEST(FindMinOtO, IgnoresSmallerOutputOfNonOneToOneNode) {
  GraphBuilder b;
  auto* pM = b.AddNode(kManyToMany, 1);  // smallest output, wrong type
  auto* nA = b.AddNode(kOneToOne, 50);

  EXPECT_EQ(FindMinOtO(NodeSet{pM, nA}), nA);
}

// The empty candidate set (all ops already fused) yields no seed.
TEST(FindMinOtO, EmptySetReturnsNull) {
  EXPECT_EQ(FindMinOtO(NodeSet{}), nullptr);
}
