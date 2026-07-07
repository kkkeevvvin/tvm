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

// Unit tests for relax::DNNFuseRelation::Classify, the producer/consumer
// op-pattern classifier that drives the bidirectional RunDNNFuse merge.

#include <gtest/gtest.h>
#include <tvm/relax/op_attr_types.h>

#include "../../src/relax/analysis/graph_partitioner.h"

using namespace tvm;
using namespace tvm::relax;

namespace {

// Assert that the relation classified from (src, sink) is exactly one kind,
// checking the three mutually-exclusive predicates and the Name() string.
void ExpectThrough(OpPatternKind src, OpPatternKind sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsThrough()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsBreak());
  EXPECT_FALSE(rel.IsDepend());
  EXPECT_STREQ(rel.Name(), "fuse_through");
}

void ExpectBreak(OpPatternKind src, OpPatternKind sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsBreak()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsThrough());
  EXPECT_FALSE(rel.IsDepend());
  EXPECT_STREQ(rel.Name(), "fuse_break");
}

void ExpectDepend(OpPatternKind src, OpPatternKind sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsDepend()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsThrough());
  EXPECT_FALSE(rel.IsBreak());
  EXPECT_STREQ(rel.Name(), "fuse_depend");
}

}  // namespace

// --- kFuseBreak: an opaque op on either side always breaks -----------------

TEST(DNNFuseRelation, OpaqueSourceAlwaysBreaks) {
  // The opaque rule is checked first and overrides the through-rules below.
  ExpectBreak(kOpaque, kElemWise);
  ExpectBreak(kOpaque, kBroadcast);
  ExpectBreak(kOpaque, kInjective);
  ExpectBreak(kOpaque, kCommReduce);
  ExpectBreak(kOpaque, kOutEWiseFusable);
  ExpectBreak(kOpaque, kTuple);
  ExpectBreak(kOpaque, kOpaque);
}

TEST(DNNFuseRelation, OpaqueSinkAlwaysBreaks) {
  ExpectBreak(kElemWise, kOpaque);
  ExpectBreak(kBroadcast, kOpaque);
  ExpectBreak(kInjective, kOpaque);
  ExpectBreak(kCommReduce, kOpaque);
  ExpectBreak(kOutEWiseFusable, kOpaque);
  ExpectBreak(kTuple, kOpaque);
}

// --- kFuseThrough: an elementwise op on either side ------------------------

TEST(DNNFuseRelation, ElemWiseSourceFusesThroughAnyNonOpaqueSink) {
  ExpectThrough(kElemWise, kElemWise);
  ExpectThrough(kElemWise, kBroadcast);
  ExpectThrough(kElemWise, kInjective);
  ExpectThrough(kElemWise, kCommReduce);
  ExpectThrough(kElemWise, kOutEWiseFusable);
}

TEST(DNNFuseRelation, ElemWiseSinkFusesThroughAnyNonOpaqueSource) {
  // The elementwise rule fires on either operand, including the heavier
  // patterns that would otherwise hit the structural kFuseBreak rule.
  ExpectThrough(kBroadcast, kElemWise);
  ExpectThrough(kInjective, kElemWise);
  ExpectThrough(kCommReduce, kElemWise);
  ExpectThrough(kOutEWiseFusable, kElemWise);
}

TEST(DNNFuseRelation, InjectiveIntoInjectiveFusesThrough) {
  ExpectThrough(kInjective, kInjective);
}

// --- kFuseBreak: broadcast/reduce-or-heavier producer into reduce-or-heavier
//     consumer ------------------------------------------------------------

TEST(DNNFuseRelation, BroadcastIntoReduceOrHeavierBreaks) {
  ExpectBreak(kBroadcast, kCommReduce);
  ExpectBreak(kBroadcast, kOutEWiseFusable);
  ExpectBreak(kBroadcast, kTuple);
}

TEST(DNNFuseRelation, ReduceOrHeavierIntoReduceOrHeavierBreaks) {
  ExpectBreak(kCommReduce, kCommReduce);
  ExpectBreak(kOutEWiseFusable, kCommReduce);
  ExpectBreak(kOutEWiseFusable, kTuple);
  ExpectBreak(kTuple, kTuple);
}

// --- kFuseDepend: everything else (no decision on its own) -----------------

TEST(DNNFuseRelation, BroadcastIntoLightConsumerDepends) {
  // Producer qualifies for break, but the sink is lighter than kCommReduce.
  ExpectDepend(kBroadcast, kBroadcast);
  ExpectDepend(kBroadcast, kInjective);
}

TEST(DNNFuseRelation, InjectiveIntoNonInjectiveDepends) {
  // Not injective-into-injective, and src is too light to break.
  ExpectDepend(kInjective, kBroadcast);
  ExpectDepend(kInjective, kCommReduce);
  ExpectDepend(kInjective, kOutEWiseFusable);
}

TEST(DNNFuseRelation, ReduceOrHeavierIntoLightConsumerDepends) {
  // Producer qualifies for break, but the sink is lighter than kCommReduce.
  ExpectDepend(kCommReduce, kBroadcast);
  ExpectDepend(kCommReduce, kInjective);
  ExpectDepend(kOutEWiseFusable, kBroadcast);
  ExpectDepend(kOutEWiseFusable, kInjective);
}
