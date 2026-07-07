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
 * \file src/relax/transform/annotate_tir_op_mapping_type.cc
 * \brief Annotate DNNFusion Table 2 MappingType (issue #8) for call_tir callee PrimFuncs,
 *        classified directly from the operator name rather than derived from OpPatternKind.
 *
 * Unlike AnnotateTIROpPattern (a tir::PrimFuncPass that derives OpPatternKind structurally
 * from the lowered TIR), this classifier needs the operator's *name*, which is only recorded
 * on the call_tir callee's GlobalVar (the same op-identity convention used by the graph
 * builder in fuse_ops.cc and by demo/mod_import/op_set.py: the callee name with its numeric
 * dedup suffix stripped, e.g. "conv2d", "add1" -> "add"). tir::CreatePrimFuncPass does not
 * expose that GlobalVar to its callback, so this pass is implemented at the IRModule level
 * instead, walking `mod->functions` directly.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

namespace {

// Table 2 (DNNFusion, \S3.1), restricted to the op set surveyed in issue #6 t1 (all_6
// models run through DecomposeOpsForInference -> LegalizeOps -> AnnotateTIROpPattern ->
// FoldConstant). Keyed on the call_tir callee's GlobalVar name with the numeric dedup
// suffix stripped. add/subtract/multiply/divide are deliberately absent -- see
// BroadcastCapableOps below.
const std::unordered_map<std::string, MappingType>& Table2Lookup() {
  static const std::unordered_map<std::string, MappingType> table = {
      {"adaptive_avg_pool2d", kManyToMany},
      {"concatenate", kOneToOne},
      {"conv2d", kManyToMany},
      {"matmul", kManyToMany},
      {"max_pool2d", kManyToMany},
      {"mean", kManyToMany},
      {"relu", kOneToOne},
      {"reshape", kReorganize},
      // silu has no ONNX counterpart; ONNX expresses it as Sigmoid + Mul (both
      // One-to-One same-shape), so the composite is effectively One-to-One.
      {"silu", kOneToOne},
      {"tir_clip", kOneToOne},
      {"tir_sigmoid", kOneToOne},
  };
  return table;
}

// add/subtract/multiply/divide are One-to-One when same-shape but One-to-Many when
// broadcasting -- Table 2 cannot resolve them by name alone. DNNFusion itself
// disambiguates these by inspecting operand shapes at the call site; we do the same
// by comparing each input buffer's shape against the output buffer's shape.
const std::unordered_set<std::string>& BroadcastCapableOps() {
  static const std::unordered_set<std::string> ops = {"add", "subtract", "multiply", "divide"};
  return ops;
}

std::string StripNumericSuffix(const ffi::String& name) {
  std::string s = name;
  size_t end = s.size();
  while (end > 0 && std::isdigit(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(0, end);
}

// Returns the static shape of a buffer as int64 extents, or std::nullopt if any
// dimension is not a static IntImm.
std::optional<std::vector<int64_t>> StaticShape(const tir::Buffer& buffer) {
  std::vector<int64_t> shape;
  shape.reserve(buffer->shape.size());
  for (const PrimExpr& dim : buffer->shape) {
    const auto* imm = dim.as<IntImmNode>();
    if (imm == nullptr) return std::nullopt;
    shape.push_back(imm->value);
  }
  return shape;
}

// Classifies a broadcast-capable binary elementwise PrimFunc as One-to-One (every input
// buffer's shape matches the output buffer's shape) or One-to-Many (some input is
// broadcast); falls back to kMappingOpaque if any buffer's shape is not statically known.
// Assumes TVM's arg-list convention of inputs followed by the (sole) output buffer.
MappingType ClassifyElemwiseShape(const tir::PrimFunc& f) {
  std::vector<std::vector<int64_t>> shapes;
  for (const tir::Var& param : f->params) {
    auto it = f->buffer_map.find(param);
    if (it == f->buffer_map.end()) continue;
    std::optional<std::vector<int64_t>> shape = StaticShape((*it).second);
    if (!shape.has_value()) return kMappingOpaque;
    shapes.push_back(std::move(shape.value()));
  }
  if (shapes.size() < 2) return kMappingOpaque;

  const std::vector<int64_t>& out_shape = shapes.back();
  for (size_t i = 0; i + 1 < shapes.size(); ++i) {
    if (shapes[i] != out_shape) return kOneToMany;
  }
  return kOneToOne;
}

MappingType ClassifyMappingType(const ffi::String& gvar_name, const tir::PrimFunc& f) {
  std::string stripped = StripNumericSuffix(gvar_name);
  if (BroadcastCapableOps().count(stripped)) {
    return ClassifyElemwiseShape(f);
  }
  const auto& table = Table2Lookup();
  auto it = table.find(stripped);
  if (it != table.end()) return it->second;
  return kMappingOpaque;
}

}  // namespace

namespace transform {

Pass AnnotateTIROpMappingType() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    IRModule updates;
    for (const auto& [gvar, func] : mod->functions) {
      const auto* prim_func = func.as<tir::PrimFuncNode>();
      if (prim_func == nullptr) continue;
      if (prim_func->GetAttr<Integer>("mapping_type").has_value()) continue;

      MappingType mapping_type =
          ClassifyMappingType(gvar->name_hint, ffi::GetRef<tir::PrimFunc>(prim_func));
      updates->Add(gvar, WithAttr(ffi::GetRef<tir::PrimFunc>(prim_func), "mapping_type",
                                   static_cast<int>(mapping_type)));
    }
    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AnnotateTIROpMappingType", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AnnotateTIROpMappingType", AnnotateTIROpMappingType);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
