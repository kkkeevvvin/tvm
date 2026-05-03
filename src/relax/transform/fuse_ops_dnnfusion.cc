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
 * \brief DNNFusion-style operator fusion for Relax (Phase 1, M1 stage).
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
 * The Phase 1 main partitioning loop (Step 3-5, fuse_depend rejection,
 * try-lower fuse_through validation, bidirectional expansion) lands in M2-M4
 * and is intentionally absent here.
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
#include <utility>
#include <vector>

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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AnalyzeDnnFusionMappingTypes",
                        AnalyzeDnnFusionMappingTypes);
}

}  // namespace relax
}  // namespace tvm
