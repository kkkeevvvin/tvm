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

/*!
 * \file src/relax/analysis/dnnfusion_seed.cc
 * \brief Implementation of the DNNFusion seed-selection helpers used by the
 *        Relax GraphPartitioner outer loop. See dnnfusion_seed.h.
 */

#include "./dnnfusion_seed.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace tvm {
namespace relax {

// ---------------------------------------------------------------------------
// Mapping type
// ---------------------------------------------------------------------------

const char* MappingTypeToString(MappingType m) {
  switch (m) {
    case MappingType::kOtO:
      return "OtO";
    case MappingType::kOtM:
      return "OtM";
    case MappingType::kMtM:
      return "MtM";
    case MappingType::kReorg:
      return "Reorg";
    case MappingType::kShuffle:
      return "Shuffle";
    case MappingType::kBreak:
      return "break";
  }
  return "<unknown>";
}

bool IsShuffleByCalleeName(const std::string& name) {
  static const char* kShuffleNames[] = {"gather", "scatter", "embedding", "take"};
  for (const char* kw : kShuffleNames) {
    if (name.find(kw) != std::string::npos) return true;
  }
  return false;
}

MappingType DeriveMappingType(OpPatternKind pat, const std::string& callee_name) {
  switch (pat) {
    case kElemWise:
      return MappingType::kOtO;
    case kBroadcast:
      return MappingType::kOtM;
    case kInjective:
      return MappingType::kReorg;
    case kCommReduce:
    case kOutEWiseFusable:
      return MappingType::kMtM;
    case kTuple:
      return MappingType::kBreak;
    case kOpaque:
      return IsShuffleByCalleeName(callee_name) ? MappingType::kShuffle : MappingType::kBreak;
  }
  return MappingType::kBreak;
}

// ---------------------------------------------------------------------------
// IRS bytes
// ---------------------------------------------------------------------------

int64_t ComputeIRSBytes(const StructInfo& sinfo) {
  if (const auto* tsinfo = sinfo.as<TensorStructInfoNode>()) {
    if (!tsinfo->shape.defined()) return std::numeric_limits<int64_t>::max();
    const auto* shape = tsinfo->shape.value().as<ShapeExprNode>();
    if (shape == nullptr) return std::numeric_limits<int64_t>::max();
    int64_t numel = 1;
    for (const PrimExpr& dim : shape->values) {
      const auto* int_imm = dim.as<IntImmNode>();
      if (int_imm == nullptr) return std::numeric_limits<int64_t>::max();
      if (int_imm->value < 0) return std::numeric_limits<int64_t>::max();
      if (int_imm->value != 0 && numel > std::numeric_limits<int64_t>::max() / int_imm->value) {
        return std::numeric_limits<int64_t>::max();
      }
      numel *= int_imm->value;
    }
    int64_t bytes_per_elem =
        static_cast<int64_t>(tsinfo->dtype.bytes()) * static_cast<int64_t>(tsinfo->dtype.lanes());
    if (bytes_per_elem <= 0) return std::numeric_limits<int64_t>::max();
    if (numel > std::numeric_limits<int64_t>::max() / bytes_per_elem) {
      return std::numeric_limits<int64_t>::max();
    }
    return numel * bytes_per_elem;
  }
  if (const auto* tup = sinfo.as<TupleStructInfoNode>()) {
    int64_t total = 0;
    for (const StructInfo& field : tup->fields) {
      int64_t s = ComputeIRSBytes(field);
      if (s == std::numeric_limits<int64_t>::max()) return std::numeric_limits<int64_t>::max();
      if (total > std::numeric_limits<int64_t>::max() - s) {
        return std::numeric_limits<int64_t>::max();
      }
      total += s;
    }
    return total;
  }
  return std::numeric_limits<int64_t>::max();
}

namespace {

// IRS for a graph node: cast its ref to a relax::Expr and read struct_info_.
// The partitioner builds graph nodes with refs that are always Var/Constant
// (see fuse_ops.cc::GraphCreator). For nodes without struct info (or
// non-Expr refs), return INT64_MAX so they sink to the back of the seed
// ranking.
int64_t NodeIRSBytes(const IndexedForwardGraph::Node* node) {
  if (node == nullptr || node->ref == nullptr) {
    return std::numeric_limits<int64_t>::max();
  }
  ObjectRef ref = ffi::GetRef<ObjectRef>(node->ref);
  if (auto opt_expr = ref.as<Expr>()) {
    Expr expr = opt_expr.value();
    if (expr->struct_info_.defined()) {
      return ComputeIRSBytes(Downcast<StructInfo>(expr->struct_info_));
    }
  }
  return std::numeric_limits<int64_t>::max();
}

}  // namespace

// ---------------------------------------------------------------------------
// Seed ordering
// ---------------------------------------------------------------------------

std::vector<size_t> ComputeSeedOrder(const IndexedForwardGraph& graph) {
  const size_t n = graph.post_dfs_order.size();
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);

  // Precompute the sort key per node so we don't re-derive in the comparator.
  struct Key {
    bool is_seed;       // OtO with statically known IRS -> sort first
    int64_t irs_bytes;  // ascending
  };
  std::vector<Key> keys(n);
  for (size_t i = 0; i < n; ++i) {
    const auto* node = graph.post_dfs_order[i];
    // Pass empty callee_name: kOpaque -> kBreak. Shuffle-vs-Break does not
    // affect the seed ranking (neither is a seed candidate).
    MappingType mt = DeriveMappingType(node->pattern, /*callee_name=*/"");
    int64_t irs = NodeIRSBytes(node);
    keys[i].is_seed = (mt == MappingType::kOtO) && (irs != std::numeric_limits<int64_t>::max());
    keys[i].irs_bytes = irs;
  }

  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (keys[a].is_seed != keys[b].is_seed) return keys[a].is_seed;  // seeds first
    if (keys[a].irs_bytes != keys[b].irs_bytes) return keys[a].irs_bytes < keys[b].irs_bytes;
    return a < b;  // topological tiebreak
  });
  return order;
}

// ---------------------------------------------------------------------------
// FuseOps trace
// ---------------------------------------------------------------------------

namespace {
thread_local std::vector<FuseTraceRow> g_fuse_trace;
}  // namespace

void ResetFuseTrace() { g_fuse_trace.clear(); }

void RecordFuseTrace(int phase, size_t src_nid, size_t sink_nid, uint32_t group_size_after) {
  g_fuse_trace.push_back(FuseTraceRow{phase, src_nid, sink_nid, group_size_after});
}

const std::vector<FuseTraceRow>& GetFuseTrace() { return g_fuse_trace; }

// ---------------------------------------------------------------------------
// FFI hooks (exp2 Step 4)
// ---------------------------------------------------------------------------

namespace {

void ResetFuseTraceFFI() { ResetFuseTrace(); }

ffi::Array<ffi::Map<ffi::String, ffi::Any>> GetLastFuseTraceFFI() {
  const auto& trace = GetFuseTrace();
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> out;
  out.reserve(trace.size());
  for (const auto& row : trace) {
    ffi::Map<ffi::String, ffi::Any> m;
    m.Set("phase", static_cast<int64_t>(row.phase));
    m.Set("src_nid", static_cast<int64_t>(row.src_nid));
    m.Set("sink_nid", static_cast<int64_t>(row.sink_nid));
    m.Set("group_size_after", static_cast<int64_t>(row.group_size_after));
    out.push_back(m);
  }
  return out;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.ResetFuseTrace", ResetFuseTraceFFI)
      .def("relax.transform.GetLastFuseTrace", GetLastFuseTraceFFI);
}

}  // namespace relax
}  // namespace tvm
