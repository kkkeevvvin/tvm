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
// mapping-type classifier (DNNFusion Table 3, \S3.2) that drives the
// bidirectional RunDNNFuse merge.

#include <gtest/gtest.h>
#include <tvm/relax/op_attr_types.h>

#include "../../src/relax/analysis/graph_partitioner.h"

using namespace tvm;
using namespace tvm::relax;

namespace {

// Assert that the relation classified from (src, sink) is exactly one kind,
// checking the three mutually-exclusive predicates and the Name() string.
void ExpectThrough(MappingType src, MappingType sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsThrough()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsBreak());
  EXPECT_FALSE(rel.IsDepend());
  EXPECT_STREQ(rel.Name(), "fuse_through");
}

void ExpectBreak(MappingType src, MappingType sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsBreak()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsThrough());
  EXPECT_FALSE(rel.IsDepend());
  EXPECT_STREQ(rel.Name(), "fuse_break");
}

void ExpectDepend(MappingType src, MappingType sink) {
  auto rel = DNNFuseRelation::Classify(src, sink);
  EXPECT_TRUE(rel.IsDepend()) << "(" << src << ", " << sink << ") -> " << rel.Name();
  EXPECT_FALSE(rel.IsThrough());
  EXPECT_FALSE(rel.IsBreak());
  EXPECT_STREQ(rel.Name(), "fuse_depend");
}

// The five non-opaque DNNFusion Table 2 mapping types, used to sweep whole
// rows/columns of Table 3 below.
constexpr MappingType kAllMappingTypes[] = {kOneToOne, kOneToMany, kManyToMany, kReorganize,
                                            kShuffle};

}  // namespace

// --- kFuseBreak: kMappingOpaque on either side always breaks ---------------
// (an op the Table 2 lookup couldn't classify; not part of Table 3 itself)

TEST(DNNFuseRelation, OpaqueSourceAlwaysBreaks) {
  for (MappingType sink : kAllMappingTypes) ExpectBreak(kMappingOpaque, sink);
  ExpectBreak(kMappingOpaque, kMappingOpaque);
}

TEST(DNNFuseRelation, OpaqueSinkAlwaysBreaks) {
  for (MappingType src : kAllMappingTypes) ExpectBreak(src, kMappingOpaque);
}

// --- kFuseThrough: One-to-One fuses with anything, in either order ---------
//
// "When a One-to-One operator (with the input I and the output O) is fused
// with an operator of any type (Op2) ... this fusion [is] correct and
// profitable." Table 3's One-to-One row and column are both all-green.

TEST(DNNFuseRelation, OneToOneSourceFusesThroughAnySink) {
  for (MappingType sink : kAllMappingTypes) ExpectThrough(kOneToOne, sink);
}

TEST(DNNFuseRelation, OneToOneSinkFusesThroughAnySource) {
  for (MappingType src : kAllMappingTypes) ExpectThrough(src, kOneToOne);
}

// --- kFuseThrough: Reorganize/Shuffle are variants of One-to-One and fuse --
//     directly with each other -----------------------------------------------
//
// "Both types are variants of One-to-One with a special mapping function
// between the input and the output. Above reasons for the correctness
// analysis are also applied here."

TEST(DNNFuseRelation, ReorganizeAndShuffleFuseThroughEachOther) {
  ExpectThrough(kReorganize, kReorganize);
  ExpectThrough(kReorganize, kShuffle);
  ExpectThrough(kShuffle, kReorganize);
  ExpectThrough(kShuffle, kShuffle);
}

// --- kFuseBreak: the two explicitly unprofitable (Table 3 "x") cells -------

TEST(DNNFuseRelation, OneToManyIntoManyToManyBreaks) {
  // "Take the case that Expand followed by Conv ... we consider this fusion
  // unprofitable."
  ExpectBreak(kOneToMany, kManyToMany);
}

TEST(DNNFuseRelation, ManyToManyIntoManyToManyBreaks) {
  // "When a Many-to-One mapping operator is followed by another Conv,
  // attempting a combined execution will be too complicated ... we consider
  // them unprofitable."
  ExpectBreak(kManyToMany, kManyToMany);
}

// --- kFuseDepend: legal, but needs profiling --------------------------------

TEST(DNNFuseRelation, ManyToManyIntoOneToManyDepends) {
  // "When a Many-to-One mapping operator is followed by a One-to-Many
  // operator, e.g. Conv followed by Expand or Resize, a combined execution
  // may or may not have a desirable data access pattern ... requiring
  // further profiling."
  ExpectDepend(kManyToMany, kOneToMany);
}

TEST(DNNFuseRelation, OneToManyIntoOneToManyDepends) {
  // Table 3 does not mark (One-to-Many, One-to-Many) with an "x" (unlike the
  // two Many-to-Many-involving cells above), so it is not illegal; it also
  // isn't one of the explicitly green One-to-One/Reorganize/Shuffle cells.
  // Treated as needing profiling.
  ExpectDepend(kOneToMany, kOneToMany);
}

TEST(DNNFuseRelation, ReorganizeOrShuffleProducerIntoOneToManyOrManyToManyDepends) {
  // "... however, when fusing with One-to-Many or Many-to-Many types
  // operators, profitability needs to be validated with further profiling."
  ExpectDepend(kReorganize, kOneToMany);
  ExpectDepend(kReorganize, kManyToMany);
  ExpectDepend(kShuffle, kOneToMany);
  ExpectDepend(kShuffle, kManyToMany);
}

TEST(DNNFuseRelation, OneToManyOrManyToManyProducerIntoReorganizeOrShuffleDepends) {
  // Same Reorder-or-Shuffle exception, other direction.
  ExpectDepend(kOneToMany, kReorganize);
  ExpectDepend(kOneToMany, kShuffle);
  ExpectDepend(kManyToMany, kReorganize);
  ExpectDepend(kManyToMany, kShuffle);
}
