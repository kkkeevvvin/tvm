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

// Unit tests for relax::StructInfoBytes, the helper that caches
// IndexedForwardGraph::Node::output_size (commit 916151140).

#include <gtest/gtest.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/var.h>

#include "../../src/relax/analysis/graph_partitioner.h"

using namespace tvm;
using namespace tvm::relax;
using tvm::ffi::Array;

namespace {

// Build a static-shape TensorStructInfo from integer dims.
TensorStructInfo MakeTensor(std::vector<int64_t> dims, DataType dtype) {
  Array<PrimExpr> values;
  for (int64_t d : dims) values.push_back(IntImm(DataType::Int(64), d));
  return TensorStructInfo(ShapeExpr(values), dtype);
}

}  // namespace

TEST(StructInfoBytes, StaticTensorFloat32) {
  // 10 * 20 elements * 4 bytes = 800.
  EXPECT_EQ(StructInfoBytes(MakeTensor({10, 20}, DataType::Float(32))), 800);
}

TEST(StructInfoBytes, DtypeWidthRespected) {
  // 3 * 4 = 12 elements; float16 = 2 bytes -> 24, int8 = 1 byte -> 12.
  EXPECT_EQ(StructInfoBytes(MakeTensor({3, 4}, DataType::Float(16))), 24);
  EXPECT_EQ(StructInfoBytes(MakeTensor({3, 4}, DataType::Int(8))), 12);
}

TEST(StructInfoBytes, VectorizedLanesCounted) {
  // float32x4 = 4 bytes * 4 lanes = 16 bytes per element; 5 elements -> 80.
  EXPECT_EQ(StructInfoBytes(MakeTensor({5}, DataType::Float(32, 4))), 80);
}

TEST(StructInfoBytes, ScalarTensor) {
  // Rank-0 tensor: empty shape -> num_elements 1 -> 4 bytes for float32.
  EXPECT_EQ(StructInfoBytes(MakeTensor({}, DataType::Float(32))), 4);
}

TEST(StructInfoBytes, UnknownShapeReturnsSentinel) {
  // TensorStructInfo with only a known ndim (no ShapeExpr) is dynamic.
  EXPECT_EQ(StructInfoBytes(TensorStructInfo(DataType::Float(32), /*ndim=*/2)), -1);
}

TEST(StructInfoBytes, SymbolicDimReturnsSentinel) {
  // A non-constant dim (tir::Var) cannot be measured statically.
  tir::Var n("n", DataType::Int(64));
  Array<PrimExpr> values{IntImm(DataType::Int(64), 8), n};
  TensorStructInfo sinfo(ShapeExpr(values), DataType::Float(32));
  EXPECT_EQ(StructInfoBytes(sinfo), -1);
}

TEST(StructInfoBytes, TupleSumsFields) {
  // 800 (10x20 f32) + 24 (3x4 f16) = 824.
  Array<StructInfo> fields{MakeTensor({10, 20}, DataType::Float(32)),
                           MakeTensor({3, 4}, DataType::Float(16))};
  EXPECT_EQ(StructInfoBytes(TupleStructInfo(fields)), 824);
}

TEST(StructInfoBytes, NestedTuple) {
  // Tuple( Tensor(4 f32=16), Tuple( Tensor(2 f32=8) ) ) -> 24.
  Array<StructInfo> inner{MakeTensor({2}, DataType::Float(32))};
  Array<StructInfo> outer{MakeTensor({4}, DataType::Float(32)), TupleStructInfo(inner)};
  EXPECT_EQ(StructInfoBytes(TupleStructInfo(outer)), 24);
}

TEST(StructInfoBytes, EmptyTupleIsZero) {
  EXPECT_EQ(StructInfoBytes(TupleStructInfo(Array<StructInfo>{})), 0);
}

TEST(StructInfoBytes, TupleWithDynamicFieldReturnsSentinel) {
  // A single unmeasurable field poisons the whole tuple.
  Array<StructInfo> fields{MakeTensor({10, 20}, DataType::Float(32)),
                           TensorStructInfo(DataType::Float(32), /*ndim=*/2)};
  EXPECT_EQ(StructInfoBytes(TupleStructInfo(fields)), -1);
}

TEST(StructInfoBytes, NonTensorNonTupleReturnsSentinel) {
  EXPECT_EQ(StructInfoBytes(ObjectStructInfo()), -1);
  EXPECT_EQ(StructInfoBytes(ShapeStructInfo(/*ndim=*/2)), -1);
}
