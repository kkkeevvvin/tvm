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
      // call_tir: surface the wrapped PrimFunc under the "tir." prefix so
      // the classifier can future-proof a Table-2 alias for legalized ops.
      if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
        result[var_ref] = std::string("tir.") + std::string(gv->name_hint);
        return;
      }
    }
    result[var_ref] = outer_name;
  }
};

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
  return DeriveMappingType(op_name, pattern);
}

// FFI: small helper accessible from Python tests / drivers for IRS sanity
// checks.  CollectVarToOpName / DeriveNodeMappingType have no Python use
// case at present (consumed only by GraphPartitioner internally), so they
// stay C++-only to keep the FFI surface small.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.DnnfusionComputeIrsBytes",
                        [](Var var) -> int64_t { return ComputeIrsBytes(var.get()); });
}

}  // namespace dnnfusion
}  // namespace relax
}  // namespace tvm
