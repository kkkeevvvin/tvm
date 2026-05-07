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
 * \file src/relax/analysis/dnnfusion_planner.cc
 */

#include "dnnfusion_planner.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace relax {
namespace dnnfusion {

namespace {

class OpNameCollector : public ExprVisitor {
 public:
  std::unordered_map<const Object*, std::string> result;

 private:
  void VisitBinding_(const VarBindingNode* binding) final {
    if (const auto* call = binding->value.as<CallNode>()) {
      RecordCall(binding->var.get(), call);
    }
    ExprVisitor::VisitBinding_(binding);
  }

  void RecordCall(const Object* var_ref, const CallNode* call) {
    const auto* op = call->op.as<OpNode>();
    if (op == nullptr) return;
    const std::string& outer_name = op->name;
    if ((outer_name == "relax.call_tir" || outer_name == "relax.call_tir_inplace") &&
        call->args.size() >= 1) {
      // call_tir: surface the wrapped PrimFunc.  We try to reverse the
      // suffix LegalizeOps adds for disambiguation (e.g. "add1", "add_2",
      // "add_inplace") so the resulting name has a chance of matching a
      // "relax.<base>" entry in Table 2.  We record BOTH names so the
      // classifier can try the relax-prefixed form first and fall back to
      // the verbatim "tir.<gv_name>".
      if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
        result[var_ref] = std::string("tir.") + std::string(gv->name_hint);
        return;
      }
    }
    result[var_ref] = outer_name;
  }
};

// Strip LegalizeOps disambiguation suffixes from a PrimFunc name and try
// the corresponding relax.<base> entry in Table 2.  Returns the empty
// string if no relax-prefixed alias is plausible.
//
// LegalizeOps generates PrimFunc names like:
//   "add"            (single occurrence)
//   "add1", "add2"   (multiple occurrences)
//   "add_inplace"    (suffix from in-place ops)
// Strip trailing digits and known suffixes, then prepend "relax.".
//
// Some TOPI op names differ from the relax op name they were legalized
// from (e.g. relax.permute_dims uses topi.transpose internally).  We
// keep an alias map so SanitizeTirName resolves "tir.transpose" to
// "relax.permute_dims" — without it, transpose ops in legalized BERT/
// DistilBERT IR fall through to the kInjective fallback and are
// classified as Reorganize instead of Shuffle (paper Table 2).  See
// findings_critical_paper_audit.md item 4.
std::string SanitizeTirName(const std::string& tir_with_prefix) {
  static const std::string kPrefix = "tir.";
  if (tir_with_prefix.size() <= kPrefix.size() ||
      tir_with_prefix.compare(0, kPrefix.size(), kPrefix) != 0) {
    return std::string();
  }
  std::string body = tir_with_prefix.substr(kPrefix.size());
  // Strip known suffixes
  static const std::vector<std::string> kSuffixes = {"_inplace", "_fwd"};
  for (const std::string& suf : kSuffixes) {
    if (body.size() > suf.size() &&
        body.compare(body.size() - suf.size(), suf.size(), suf) == 0) {
      body = body.substr(0, body.size() - suf.size());
      break;
    }
  }
  // Strip trailing digits (and the optional "_" that may precede them)
  size_t end = body.size();
  while (end > 0 && std::isdigit(static_cast<unsigned char>(body[end - 1]))) {
    --end;
  }
  if (end > 0 && body[end - 1] == '_') --end;
  if (end == 0) return std::string();  // entirely digits
  body = body.substr(0, end);
  // Some PrimFunc names already have submodule prefixes baked in (e.g.
  // "nn_relu"); convert underscores in the prefix portion to dots.  We
  // recognize the "nn_" prefix specifically since that's the most common
  // legalize-out form for relax.nn.* ops.
  if (body.compare(0, 3, "nn_") == 0) {
    body = "nn." + body.substr(3);
  }
  // TOPI-name → relax-op-name aliases (kept tiny — extend on demand).
  static const std::unordered_map<std::string, std::string> kTopiAlias = {
      {"transpose", "permute_dims"},  // TOPI uses transpose; relax registers permute_dims
  };
  auto it = kTopiAlias.find(body);
  if (it != kTopiAlias.end()) body = it->second;
  return std::string("relax.") + body;
}

int64_t StructInfoBytes(const StructInfo& sinfo) {
  if (const auto* tsi = sinfo.as<TensorStructInfoNode>()) {
    const auto* shape = tsi->shape.as<ShapeExprNode>();
    if (shape == nullptr) return -1;
    int64_t count = 1;
    for (const PrimExpr& v : shape->values) {
      const auto* imm = v.as<IntImmNode>();
      if (imm == nullptr) return -1;
      count *= imm->value;
    }
    int64_t bytes_per = static_cast<int64_t>(tsi->dtype.bytes()) *
                        static_cast<int64_t>(tsi->dtype.lanes());
    return count * bytes_per;
  }
  if (const auto* tup = sinfo.as<TupleStructInfoNode>()) {
    int64_t total = 0;
    for (const StructInfo& f : tup->fields) {
      int64_t b = StructInfoBytes(f);
      if (b < 0) return -1;
      total += b;
    }
    return total;
  }
  return -1;  // ShapeStructInfo / PrimStructInfo / ObjectStructInfo: no bytes.
}

}  // namespace

std::unordered_map<const Object*, std::string> CollectVarToOpName(const IRModule& mod) {
  OpNameCollector collector;
  for (const auto& it : mod->functions) {
    const auto* func = it.second.as<FunctionNode>();
    if (func == nullptr || func->HasNonzeroAttr(attr::kPrimitive) ||
        func->GetAttr<ffi::String>(attr::kCodegen).has_value()) {
      continue;
    }
    collector(ffi::GetRef<Function>(func));
  }
  return std::move(collector.result);
}

int64_t ComputeIrsBytes(const Object* ref) {
  if (ref == nullptr) return -1;
  const StructInfoNode* sinfo_node = nullptr;
  if (ref->IsInstance<VarNode>()) {
    sinfo_node = static_cast<const VarNode*>(ref)->struct_info_.as<StructInfoNode>();
  } else if (ref->IsInstance<ConstantNode>()) {
    sinfo_node = static_cast<const ConstantNode*>(ref)->struct_info_.as<StructInfoNode>();
  }
  if (sinfo_node == nullptr) return -1;
  return StructInfoBytes(ffi::GetRef<StructInfo>(sinfo_node));
}

MappingType DeriveNodeMappingType(
    const Object* ref, OpPatternKind pattern,
    const std::unordered_map<const Object*, std::string>& var_to_op_name) {
  std::string op_name;
  if (ref != nullptr) {
    auto it = var_to_op_name.find(ref);
    if (it != var_to_op_name.end()) op_name = it->second;
  }
  // If op_name starts with "tir." (i.e. came from a legalized call_tir),
  // first try a sanitized "relax.<base>" lookup so Table 2 has a chance of
  // hitting.  Falls back to OpPatternKind if the sanitized name isn't in
  // Table 2 either.
  if (!op_name.empty() && op_name.compare(0, 4, "tir.") == 0) {
    std::string relax_alias = SanitizeTirName(op_name);
    if (!relax_alias.empty()) {
      MappingType from_alias = DeriveMappingType(relax_alias, pattern);
      // DeriveMappingType returns the OpPatternKind fallback when the
      // explicit lookup misses; we only prefer the alias result when it
      // actually came from Table 2 (i.e. would differ from the verbatim
      // tir.<name> lookup, which currently always misses Table 2).
      if (from_alias != DeriveMappingType(std::string(), pattern)) {
        return from_alias;
      }
    }
  }
  return DeriveMappingType(op_name, pattern);
}

bool TvmCompatAllowsYellow(MappingType producer, MappingType consumer) {
  using MT = MappingType;
  // Only the three cells where TVM's RunFuse already allows the corresponding
  // OpPatternKind transition.
  if (producer == MT::kOneToOne && consumer == MT::kManyToMany) return true;
  if (producer == MT::kManyToMany && consumer == MT::kOneToMany) return true;
  if (producer == MT::kOneToMany && consumer == MT::kReorganize) return true;
  return false;
}

std::string MakeCellSignature(MappingType producer, MappingType consumer) {
  return std::string(MappingTypeName(producer)) + "_x_" + MappingTypeName(consumer);
}

namespace {

std::string ShapeStringFromStructInfo(const StructInfo& sinfo) {
  if (const auto* tsi = sinfo.as<TensorStructInfoNode>()) {
    const auto* shape = tsi->shape.as<ShapeExprNode>();
    if (shape == nullptr) return "";  // dynamic
    std::string out;
    for (const PrimExpr& v : shape->values) {
      const auto* imm = v.as<IntImmNode>();
      if (imm == nullptr) return "";  // dynamic
      if (!out.empty()) out += "x";
      out += std::to_string(imm->value);
    }
    return out;
  }
  return "";
}

}  // namespace

std::string MakeEdgeSignature(
    const Object* p_ref, const Object* c_ref,
    const std::unordered_map<const Object*, std::string>& var_to_op_name) {
  std::string p_callee, c_callee;
  if (p_ref != nullptr) {
    auto it = var_to_op_name.find(p_ref);
    if (it != var_to_op_name.end()) p_callee = it->second;
  }
  if (c_ref != nullptr) {
    auto it = var_to_op_name.find(c_ref);
    if (it != var_to_op_name.end()) c_callee = it->second;
  }
  std::string p_shape, c_shape;
  if (p_ref != nullptr && p_ref->IsInstance<VarNode>()) {
    const auto* var = static_cast<const VarNode*>(p_ref);
    if (auto* sinfo_node = var->struct_info_.as<StructInfoNode>()) {
      p_shape = ShapeStringFromStructInfo(ffi::GetRef<StructInfo>(sinfo_node));
    }
  }
  if (c_ref != nullptr && c_ref->IsInstance<VarNode>()) {
    const auto* var = static_cast<const VarNode*>(c_ref);
    if (auto* sinfo_node = var->struct_info_.as<StructInfoNode>()) {
      c_shape = ShapeStringFromStructInfo(ffi::GetRef<StructInfo>(sinfo_node));
    }
  }
  return p_callee + "|" + c_callee + "|" + p_shape + "|" + c_shape;
}

ProfileLookupResult LookupProfile(
    const std::unordered_map<std::string, bool>& db,
    const std::string& edge_signature,
    const std::string& cell_signature) {
  // 1. Try the per-edge signature first.
  auto it = db.find(edge_signature);
  if (it != db.end()) return {true, it->second};
  // 2. Fall back to cell-type signature.
  it = db.find(cell_signature);
  if (it != db.end()) return {true, it->second};
  // 3. Miss.
  return {false, false};
}

// FFI: small helpers accessible from Python tests / drivers.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.DnnfusionComputeIrsBytes",
                        [](Var var) -> int64_t { return ComputeIrsBytes(var.get()); });
  // SanitizeTirName so Python-side enumerate_yellow_edges (in
  // experiments/dnnfusion_port/runtime_profile.py) classifies mapping types
  // identically to the C++ partitioner.  Removes the previously duplicated
  // sanitize logic that risked drifting between languages.
  refl::GlobalDef().def("relax.analysis.DnnfusionSanitizeTirName",
                        [](ffi::String tir_name) -> ffi::String {
                          return SanitizeTirName(std::string(tir_name));
                        });
}

}  // namespace dnnfusion
}  // namespace relax
}  // namespace tvm
