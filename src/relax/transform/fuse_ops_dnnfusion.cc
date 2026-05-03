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
 * \file src/relax/transform/fuse_ops_dnnfusion.cc
 * \brief DNNFusion-style operator fusion for Relax (Phase 1, M1 + M2 stage).
 *
 * Phase 1 / M1 scope (per dnnfusion_fuse_ops_plan.md, Step 1+2):
 *   - Derive DNNFusion mapping type (OtO / OtM / MtM / Reorg / Shuffle / break)
 *     from the existing OpPatternKind annotation set by AnnotateTIROpPattern,
 *     without registering a new op attribute.
 *   - Compute IRS (intermediate-result size) per binding from TensorStructInfo
 *     so the seed-selection min-heap (M4) can sort one-to-one ops ascending.
 *   - Expose an FFI analysis hook so unit tests / driver scripts can verify
 *     the derivation table and the seed-order computation against hand-checked
 *     ground truth on ResNet50 / BERT.
 *
 * Phase 1 / M2 scope (per dnnfusion_fuse_ops_plan.md, Step 3):
 *   - Encode the 5x5 mapping matrix from the DNNFusion paper Table 3 as
 *     FuseRelation MappingCheck(producer, consumer).
 *   - Phase 1 decision wrapper: kThru -> accept, kDep / kBreak -> reject.
 *     The kDep branch is rejected outright in Phase 1; Phase 2 will reroute it
 *     through a profile-based latency oracle.
 *   - Expose two more FFI hooks: a per-pair query for unit-test coverage of
 *     the full 25-cell matrix, and a per-edge dataflow walk that uses
 *     BuildIndexedForwardGraph (exported from fuse_ops.cc) so the M2
 *     acceptance criterion ("dump every edge with its mapping check result")
 *     is satisfiable on real models.
 *
 * The Phase 1 main partitioning loop (Step 4-5, try-lower fuse_through
 * validation, bidirectional expansion) lands in M3-M4 and is intentionally
 * absent here.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/arena.h"
#include "../analysis/graph_partitioner.h"
#include "utils.h"

namespace tvm {
namespace relax {

// ---------------------------------------------------------------------------
// Mapping type
// ---------------------------------------------------------------------------

enum class MappingType : int {
  kOtO = 0,      // One-to-One   (relu, sigmoid, scalar add, ...)
  kOtM = 1,      // One-to-Many  (broadcast / expand)
  kMtM = 2,      // Many-to-Many (conv, matmul, reduce, pooling)
  kReorg = 3,    // Reorganize   (transpose, reshape, concat, ...)
  kShuffle = 4,  // Data-dependent indexing (gather, scatter, embedding, take)
  kBreak = 5,    // Cannot fuse
};

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

// ---------------------------------------------------------------------------
// FuseRelation: 5x5 mapping matrix (paper Table 3)
// ---------------------------------------------------------------------------

// One entry per (producer_mapping_type, consumer_mapping_type) cell of the
// DNNFusion matrix.
//
//   kThru  : safe to fuse purely from a mapping-type standpoint. Step 4
//            (try-lower) still gets to veto.
//   kDep   : fusion is profitability-dependent. Phase 1 rejects unconditionally;
//            Phase 2 routes this through a profile/cost oracle.
//   kBreak : never fuse.
enum class FuseRelation : int {
  kThru = 0,
  kDep = 1,
  kBreak = 2,
};

const char* FuseRelationToString(FuseRelation r) {
  switch (r) {
    case FuseRelation::kThru:
      return "thru";
    case FuseRelation::kDep:
      return "dep";
    case FuseRelation::kBreak:
      return "break";
  }
  return "<unknown>";
}

// Lookup the 5x5 mapping matrix from the DNNFusion paper Table 3.
//
//                 consumer
//   producer  | OtO  | OtM  | MtM  | Reorg | Shuffle
//   ----------+------+------+------+-------+--------
//   OtO       | thru | thru | thru | thru  | break
//   OtM       | thru | thru | dep  | dep   | break
//   MtM       | thru | dep  | dep  | dep   | break
//   Reorg     | thru | thru | dep  | thru  | break
//   Shuffle   | dep  | break| break| break | break
//
// Either side being kBreak (i.e., the op was not classified) collapses to
// kBreak: we never invent a fusion across an unknown boundary.
FuseRelation MappingCheck(MappingType producer, MappingType consumer) {
  if (producer == MappingType::kBreak || consumer == MappingType::kBreak) {
    return FuseRelation::kBreak;
  }
  // Indexed by [producer][consumer] over {OtO, OtM, MtM, Reorg, Shuffle} = 0..4.
  static constexpr FuseRelation kTable[5][5] = {
      // consumer:  OtO,             OtM,             MtM,             Reorg,           Shuffle
      /* OtO     */ {FuseRelation::kThru, FuseRelation::kThru, FuseRelation::kThru,
                    FuseRelation::kThru, FuseRelation::kBreak},
      /* OtM     */ {FuseRelation::kThru, FuseRelation::kThru, FuseRelation::kDep,
                    FuseRelation::kDep, FuseRelation::kBreak},
      /* MtM     */ {FuseRelation::kThru, FuseRelation::kDep, FuseRelation::kDep,
                    FuseRelation::kDep, FuseRelation::kBreak},
      /* Reorg   */ {FuseRelation::kThru, FuseRelation::kThru, FuseRelation::kDep,
                    FuseRelation::kThru, FuseRelation::kBreak},
      /* Shuffle */ {FuseRelation::kDep, FuseRelation::kBreak, FuseRelation::kBreak,
                    FuseRelation::kBreak, FuseRelation::kBreak},
  };
  return kTable[static_cast<int>(producer)][static_cast<int>(consumer)];
}

// Phase 1 fuse decision per the plan (§4.3 Step 3): kThru -> accept,
// everything else -> reject. The kDep branch is the one that requires the
// Phase 2 latency oracle, so it is *intentionally* rejected here even though
// the paper would profile it.
bool Phase1AllowFusion(MappingType producer, MappingType consumer) {
  return MappingCheck(producer, consumer) == FuseRelation::kThru;
}

// Phase 1 Shuffle whitelist. After LegalizeOps the high-level relax op identity
// is encoded in the lowered TIR PrimFunc's name (e.g. "take", "gather_nd"), so
// we substring-match against the GlobalVar / Op name. Limitation: any op that
// was not legalized via LegalizeOps and stays kOpaque is treated as kBreak.
// Phase 2 will replace this with a registered op attribute.
bool IsShuffleByCalleeName(const std::string& name) {
  static const char* kShuffleNames[] = {"gather", "scatter", "embedding", "take"};
  for (const char* kw : kShuffleNames) {
    if (name.find(kw) != std::string::npos) return true;
  }
  return false;
}

// Derive DNNFusion mapping type from TVM's OpPatternKind + the underlying
// callee name (used only for the kOpaque -> Shuffle whitelist).
//
// Mapping (per plan §4.3 Step 1):
//   kElemWise         -> OtO
//   kBroadcast        -> OtM
//   kInjective        -> Reorg
//   kCommReduce       -> MtM
//   kOutEWiseFusable  -> MtM
//   kTuple            -> kBreak       (partitioner sees through tuples;
//                                     classification is meaningless here)
//   kOpaque           -> Shuffle if callee name matches whitelist, else kBreak
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
// IRS (intermediate-result size) in bytes
// ---------------------------------------------------------------------------

// Return numel * dtype_bytes for a static-shape TensorStructInfo. For dynamic
// shape, unknown dtype, or any overflow, return INT64_MAX so the binding
// sorts to the back of the seed-selection min-heap.
//
// For a TupleStructInfo, sum the IRS of all fields. Any dynamic field
// poisons the whole tuple to INT64_MAX.
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
      // Overflow guard.
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

// ---------------------------------------------------------------------------
// Per-binding metadata collector
// ---------------------------------------------------------------------------

// Per-binding metadata (in post-DFS order of each Relax function body) that
// the analyze FFI exposes to Python. We walk the IRModule directly here rather
// than reusing fuse_ops.cc's GraphCreator: M1 needs no dataflow-edge info, and
// GraphCreator is not exported beyond its translation unit. M2+ will gain
// access to IndexedForwardGraph through a separate header-export refactor.
class DnnFusionAnalyzer : public ExprVisitor {
 public:
  struct Info {
    // Pointer to the Var defined by this binding. Doubles as the key in the
    // IndexedForwardGraph::node_map so the M2 edge walker can join graph nodes
    // back to per-binding metadata. Null for synthetic rows that do not
    // correspond to a binding.
    const tvm::Object* ref{nullptr};
    std::string var_name;
    std::string callee_name;
    OpPatternKind op_pattern{kOpaque};
    int64_t irs_bytes{std::numeric_limits<int64_t>::max()};
  };

  static std::vector<Info> Collect(const IRModule& mod) {
    DnnFusionAnalyzer a(mod);
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func = base_func.as<FunctionNode>();
      if (func == nullptr || func->HasNonzeroAttr(attr::kPrimitive) ||
          func->GetAttr<ffi::String>(attr::kCodegen).has_value()) {
        continue;
      }
      a(ffi::GetRef<Function>(func));
    }
    return std::move(a.rows_);
  }

 private:
  explicit DnnFusionAnalyzer(IRModule mod) : mod_(std::move(mod)) {}

  void VisitBinding_(const VarBindingNode* binding) final {
    Info info;
    info.ref = binding->var.get();
    info.var_name = binding->var->name_hint();
    if (const auto* call = binding->value.as<CallNode>()) {
      info.callee_name = ExtractCalleeName(call);
      info.op_pattern = ReadOpPattern(call);
    } else if (binding->value.as<TupleGetItemNode>()) {
      info.callee_name = "<tuple_get_item>";
      // Mirror GraphCreator: TupleGetItem is treated as kInjective when not
      // pulling from a packed-param tuple. M1 is satisfied with kInjective
      // here; the partitioner in M4 will see through tuples explicitly.
      info.op_pattern = kInjective;
    } else if (binding->value.as<TupleNode>()) {
      info.callee_name = "<tuple>";
      info.op_pattern = kTuple;
    } else {
      info.callee_name = "<other>";
      info.op_pattern = kOpaque;
    }
    if (binding->var->struct_info_.defined()) {
      info.irs_bytes = ComputeIRSBytes(Downcast<StructInfo>(binding->var->struct_info_));
    }
    rows_.push_back(std::move(info));
  }

  // Read OpPatternKind for a relax::Call. For call_tir / call_tir_inplace, the
  // pattern lives on the linked PrimFunc's "op_pattern" attribute (set by
  // AnnotateTIROpPattern). Falls back to kOpaque when missing — same default
  // GraphCreator uses.
  OpPatternKind ReadOpPattern(const CallNode* call) const {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    if (call->op.same_as(call_tir_op) || call->op.same_as(call_tir_inplace_op)) {
      if (call->args.empty()) return kOpaque;
      const auto* gv = call->args[0].as<GlobalVarNode>();
      if (gv == nullptr) return kOpaque;
      auto it = mod_->functions.find(ffi::GetRef<GlobalVar>(gv));
      if (it == mod_->functions.end()) return kOpaque;
      const auto* prim = (*it).second.as<tir::PrimFuncNode>();
      if (prim == nullptr) return kOpaque;
      auto opt = prim->GetAttr<Integer>("op_pattern");
      if (!opt.defined()) return kOpaque;
      return static_cast<OpPatternKind>(opt.value()->value);
    }
    return kOpaque;
  }

  static std::string ExtractCalleeName(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    if (call->op.same_as(call_tir_op) || call->op.same_as(call_tir_inplace_op)) {
      if (!call->args.empty()) {
        if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
          return gv->name_hint;
        }
      }
    }
    if (const auto* op = call->op.as<OpNode>()) {
      return op->name;
    }
    if (const auto* gv = call->op.as<GlobalVarNode>()) {
      return gv->name_hint;
    }
    return "<unknown>";
  }

  IRModule mod_;
  std::vector<Info> rows_;
};

// ---------------------------------------------------------------------------
// FFI hook: AnalyzeDnnFusionMappingTypes
// ---------------------------------------------------------------------------

// Returns one ffi::Map per Relax binding (post-DFS order, restricted to
// non-primitive Relax functions in the module) with the keys:
//   "name"          : binding-var name_hint (String)
//   "callee"        : callee name           (String)
//   "op_pattern"    : OpPatternKind         (int)
//   "mapping_type"  : MappingType           (int)
//   "mapping_name"  : MappingType label     (String, e.g. "OtO")
//   "irs_bytes"     : IRS in bytes          (int64; INT64_MAX if dynamic)
//   "is_seed_cand"  : true iff OtO + static IRS (bool)
ffi::Array<ffi::Map<ffi::String, ffi::Any>> AnalyzeDnnFusionMappingTypes(IRModule mod) {
  std::vector<DnnFusionAnalyzer::Info> rows = DnnFusionAnalyzer::Collect(mod);

  ffi::Array<ffi::Map<ffi::String, ffi::Any>> out;
  out.reserve(rows.size());

  for (const auto& info : rows) {
    MappingType mt = DeriveMappingType(info.op_pattern, info.callee_name);
    ffi::Map<ffi::String, ffi::Any> row;
    row.Set("name", ffi::String(info.var_name));
    row.Set("callee", ffi::String(info.callee_name));
    row.Set("op_pattern", static_cast<int64_t>(info.op_pattern));
    row.Set("mapping_type", static_cast<int64_t>(mt));
    row.Set("mapping_name", ffi::String(MappingTypeToString(mt)));
    row.Set("irs_bytes", info.irs_bytes);
    row.Set("is_seed_cand",
            mt == MappingType::kOtO && info.irs_bytes != std::numeric_limits<int64_t>::max());
    out.push_back(row);
  }
  return out;
}

// ---------------------------------------------------------------------------
// FFI hook: DnnFusionMappingCheck (per-pair query for unit tests)
// ---------------------------------------------------------------------------

// Pure lookup wrapper exposed so unit tests can verify all 25 cells of the
// 5x5 mapping matrix end-to-end without needing to construct a Relax IR.
// Inputs are the int values of MappingType (the Python side already has
// DNNFUSION_MAPPING_TYPES tuple to convert names <-> ints).
//
// Returns: { relation:int, name:str, phase1_allow:bool }
ffi::Map<ffi::String, ffi::Any> DnnFusionMappingCheckFFI(int producer_int, int consumer_int) {
  // Validate against the inclusive enum range (kOtO=0 .. kBreak=5).
  auto in_range = [](int v) {
    return v >= static_cast<int>(MappingType::kOtO) && v <= static_cast<int>(MappingType::kBreak);
  };
  CHECK(in_range(producer_int))
      << "DnnFusionMappingCheck: producer mapping type " << producer_int << " out of range";
  CHECK(in_range(consumer_int))
      << "DnnFusionMappingCheck: consumer mapping type " << consumer_int << " out of range";

  auto producer = static_cast<MappingType>(producer_int);
  auto consumer = static_cast<MappingType>(consumer_int);
  FuseRelation rel = MappingCheck(producer, consumer);

  ffi::Map<ffi::String, ffi::Any> out;
  out.Set("relation", static_cast<int64_t>(rel));
  out.Set("name", ffi::String(FuseRelationToString(rel)));
  out.Set("phase1_allow", rel == FuseRelation::kThru);
  return out;
}

// ---------------------------------------------------------------------------
// FFI hook: AnalyzeDnnFusionFuseEdges (per-edge dataflow walk)
// ---------------------------------------------------------------------------

// Walk every directed edge in the dataflow graph and report what the M4
// partitioner main loop would see when it queries MappingCheck on that edge.
//
// Reuses BuildIndexedForwardGraph (exported from fuse_ops.cc, see
// utils.h:162) so the per-edge view is the *same* graph the existing FuseOps
// path operates on -- no parallel graph builder.
//
// Each output row corresponds to a single edge with keys:
//   producer            : binding-var name_hint of the source node       (str)
//   consumer            : binding-var name_hint of the sink node         (str)
//   producer_pattern    : OpPatternKind on the source node               (int)
//   consumer_pattern    : OpPatternKind on the sink node                 (int)
//   producer_mapping    : MappingType of the source                      (int)
//   producer_mapping_name : MappingType label (e.g. "MtM")               (str)
//   consumer_mapping    : MappingType of the sink                        (int)
//   consumer_mapping_name : MappingType label                            (str)
//   relation            : FuseRelation int                                (int)
//   relation_name       : FuseRelation label ("thru" / "dep" / "break")  (str)
//   phase1_allow        : true iff Phase 1 would accept this edge        (bool)
//
// Edges where the producer is a function parameter (no Relax binding) are
// included with producer_mapping=kBreak (we never fuse "across a parameter")
// and producer="<param>" so the consumer's view is still complete.
ffi::Array<ffi::Map<ffi::String, ffi::Any>> AnalyzeDnnFusionFuseEdges(IRModule mod) {
  // Per-binding metadata indexed by Var* (the same Object* used as graph-node
  // ref). Walks the IR once.
  std::vector<DnnFusionAnalyzer::Info> rows = DnnFusionAnalyzer::Collect(mod);
  std::unordered_map<const tvm::Object*, const DnnFusionAnalyzer::Info*> by_ref;
  by_ref.reserve(rows.size());
  for (const auto& info : rows) {
    if (info.ref != nullptr) by_ref.emplace(info.ref, &info);
  }

  // Build the dataflow graph. Same arena lifetime as the call.
  support::Arena arena;
  IndexedForwardGraph graph = BuildIndexedForwardGraph(mod, &arena);

  // For nodes without a Relax binding (i.e. parameters), surface a sentinel.
  auto resolve = [&](const IndexedForwardGraph::Node* n)
      -> std::pair<std::string, MappingType> {
    auto it = by_ref.find(n->ref);
    if (it == by_ref.end()) {
      // Parameter or otherwise unmapped node. Default callee_name "" never
      // hits the Shuffle whitelist, so DeriveMappingType -> kBreak.
      return {std::string("<param>"), DeriveMappingType(n->pattern, "")};
    }
    return {it->second->var_name, DeriveMappingType(n->pattern, it->second->callee_name)};
  };

  ffi::Array<ffi::Map<ffi::String, ffi::Any>> out;
  for (IndexedForwardGraph::Node* node : graph.post_dfs_order) {
    if (node == nullptr) continue;
    auto [producer_name, producer_mt] = resolve(node);
    for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      const IndexedForwardGraph::Node* sink = link->value.node;
      if (sink == nullptr) continue;
      auto [consumer_name, consumer_mt] = resolve(sink);
      FuseRelation rel = MappingCheck(producer_mt, consumer_mt);

      ffi::Map<ffi::String, ffi::Any> row;
      row.Set("producer", ffi::String(producer_name));
      row.Set("consumer", ffi::String(consumer_name));
      row.Set("producer_pattern", static_cast<int64_t>(node->pattern));
      row.Set("consumer_pattern", static_cast<int64_t>(sink->pattern));
      row.Set("producer_mapping", static_cast<int64_t>(producer_mt));
      row.Set("producer_mapping_name", ffi::String(MappingTypeToString(producer_mt)));
      row.Set("consumer_mapping", static_cast<int64_t>(consumer_mt));
      row.Set("consumer_mapping_name", ffi::String(MappingTypeToString(consumer_mt)));
      row.Set("relation", static_cast<int64_t>(rel));
      row.Set("relation_name", ffi::String(FuseRelationToString(rel)));
      row.Set("phase1_allow", rel == FuseRelation::kThru);
      out.push_back(row);
    }
  }
  return out;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.AnalyzeDnnFusionMappingTypes", AnalyzeDnnFusionMappingTypes)
      .def("relax.transform.DnnFusionMappingCheck", DnnFusionMappingCheckFFI)
      .def("relax.transform.AnalyzeDnnFusionFuseEdges", AnalyzeDnnFusionFuseEdges);
}

}  // namespace relax
}  // namespace tvm
