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
#include <tvm/relax/expr_functor.h>
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
// models) plus the ops issue #37 found uncovered on the paper_6 models (vgg16, unet,
// c3d, s3d, mobilenet_v1_ssd, yolov4) -- both surveys run the model through
// DecomposeOpsForInference -> LegalizeOps -> AnnotateTIROpPattern -> FoldConstant.
// Keyed on the call_tir callee's GlobalVar name with the numeric dedup suffix stripped.
// add/subtract/multiply/divide are deliberately absent -- see BroadcastCapableOps below.
const std::unordered_map<std::string, MappingType>& Table2Lookup() {
  static const std::unordered_map<std::string, MappingType> table = {
      {"adaptive_avg_pool2d", kManyToMany},
      {"avg_pool3d", kManyToMany},
      {"concatenate", kOneToOne},
      {"conv2d", kManyToMany},
      {"conv2d_transpose", kManyToMany},
      {"conv3d", kManyToMany},
      {"leaky_relu", kOneToOne},
      {"matmul", kManyToMany},
      {"max_pool2d", kManyToMany},
      {"max_pool3d", kManyToMany},
      {"mean", kManyToMany},
      {"relu", kOneToOne},
      {"reshape", kReorganize},
      {"resize2d", kOneToMany},
      // silu has no ONNX counterpart; ONNX expresses it as Sigmoid + Mul (both
      // One-to-One same-shape), so the composite is effectively One-to-One.
      {"silu", kOneToOne},
      // softplus is not listed in Table 2, but softplus(x) = log(1 + exp(beta*x)) / beta
      // reads each input element exactly once at the same position -- One-to-One.
      {"softplus", kOneToOne},
      {"tir_clip", kOneToOne},
      {"tir_sigmoid", kOneToOne},
      {"tir_tanh", kOneToOne},
      // relax.permute_dims legalizes to topi.transpose, whose PrimFunc is named
      // "transpose": an axis-permuting reindex, i.e. Table 2's Shuffle.
      {"transpose", kShuffle},
  };
  return table;
}

// add/subtract/multiply/divide are One-to-One when same-shape but One-to-Many when
// broadcasting -- Table 2 cannot resolve them by name alone. DNNFusion itself
// disambiguates these by inspecting operand shapes at the call site; we do the same
// by comparing each input buffer's shape against the output buffer's shape, ignoring
// weight operands (see ConstArgMask below).
const std::unordered_set<std::string>& BroadcastCapableOps() {
  static const std::unordered_set<std::string> ops = {"add", "subtract", "multiply", "divide"};
  return ops;
}

// Per input position of a call_tir callee: true iff every call site passes a weight
// operand in that position. Model weights/parameters arrive as relax::Constant
// (from_fx embeds them; this pass runs before FoldConstant) -- either directly, or
// through a chain of bindings computed purely from Constants (e.g. from_fx emits a
// conv bias as `reshape(bias_const, (1, -1, 1, 1))` before the bias add). A broadcast
// against a weight is not a data-flow One-to-Many mapping in DNNFusion's sense -- the
// mapping type describes the activation edge the fusion planner walks -- so such
// operands are excluded from the shape comparison.
using ConstArgMask = std::vector<bool>;

std::unordered_map<const GlobalVarNode*, ConstArgMask> CollectConstArgMasks(const IRModule& mod) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");

  std::unordered_map<const GlobalVarNode*, ConstArgMask> masks;
  for (const auto& kv : mod->functions) {
    const auto* relax_func = kv.second.as<FunctionNode>();
    if (relax_func == nullptr) continue;
    const auto* seq = relax_func->body.as<SeqExprNode>();
    if (seq == nullptr) continue;

    // Vars whose value is computed purely from Constants (weight-derived). Function
    // params and MatchCast-bound vars never enter the set, so anything touching real
    // data drops out. Bindings are walked in def order; defs precede uses.
    std::unordered_set<const VarNode*> const_vars;
    auto is_weight_arg = [&const_vars](const Expr& arg) {
      if (arg->IsInstance<ConstantNode>()) return true;
      const auto* var = arg.as<VarNode>();
      return var != nullptr && const_vars.count(var) > 0;
    };

    for (const BindingBlock& block : seq->blocks) {
      for (const Binding& binding : block->bindings) {
        const auto* var_binding = binding.as<VarBindingNode>();
        if (var_binding == nullptr) continue;

        bool const_derived = true;
        PostOrderVisit(var_binding->value, [&](const Expr& e) {
          const auto* var = e.as<VarNode>();
          if (var != nullptr && const_vars.count(var) == 0) const_derived = false;
        });
        if (const_derived) const_vars.insert(var_binding->var.get());

        const auto* call = var_binding->value.as<CallNode>();
        if (call == nullptr || (call->op != call_tir_op && call->op != call_tir_inplace_op)) {
          continue;
        }
        const auto* callee = call->args[0].as<GlobalVarNode>();
        const auto* args = call->args[1].as<TupleNode>();
        if (callee == nullptr || args == nullptr) continue;

        ConstArgMask site(args->fields.size());
        for (size_t i = 0; i < args->fields.size(); ++i) {
          site[i] = is_weight_arg(args->fields[i]);
        }
        auto it = masks.find(callee);
        if (it == masks.end()) {
          masks.emplace(callee, std::move(site));
          continue;
        }
        // Callee shared by several call sites: only positions that are weights at
        // *every* site stay masked, so a shared PrimFunc never borrows another call
        // site's weight operand.
        ConstArgMask& merged = it->second;
        if (merged.size() != site.size()) {
          merged.assign(merged.size(), false);
          continue;
        }
        for (size_t i = 0; i < merged.size(); ++i) {
          merged[i] = merged[i] && site[i];
        }
      }
    }
  }
  return masks;
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
// Inputs marked constant in `const_args` (weights/params at every call site) are ignored
// in the comparison, unless every input is constant (then all inputs are compared, as a
// fully-constant op has no activation edge to prefer -- FoldConstant erases it anyway).
MappingType ClassifyElemwiseShape(const tir::PrimFunc& f, const ConstArgMask& const_args) {
  std::vector<std::vector<int64_t>> shapes;
  for (const tir::Var& param : f->params) {
    auto it = f->buffer_map.find(param);
    if (it == f->buffer_map.end()) continue;
    std::optional<std::vector<int64_t>> shape = StaticShape((*it).second);
    if (!shape.has_value()) return kMappingOpaque;
    shapes.push_back(std::move(shape.value()));
  }
  if (shapes.size() < 2) return kMappingOpaque;

  auto is_weight = [&](size_t i) { return i < const_args.size() && const_args[i]; };
  bool has_data_input = false;
  for (size_t i = 0; i + 1 < shapes.size(); ++i) {
    if (!is_weight(i)) has_data_input = true;
  }

  const std::vector<int64_t>& out_shape = shapes.back();
  for (size_t i = 0; i + 1 < shapes.size(); ++i) {
    if (has_data_input && is_weight(i)) continue;
    if (shapes[i] != out_shape) return kOneToMany;
  }
  return kOneToOne;
}

MappingType ClassifyMappingType(const ffi::String& gvar_name, const tir::PrimFunc& f,
                                const ConstArgMask& const_args) {
  std::string stripped = StripNumericSuffix(gvar_name);
  if (BroadcastCapableOps().count(stripped)) {
    return ClassifyElemwiseShape(f, const_args);
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
    std::unordered_map<const GlobalVarNode*, ConstArgMask> const_arg_masks =
        CollectConstArgMasks(mod);
    static const ConstArgMask empty_mask;

    IRModule updates;
    for (const auto& [gvar, func] : mod->functions) {
      const auto* prim_func = func.as<tir::PrimFuncNode>();
      if (prim_func == nullptr) continue;
      if (prim_func->GetAttr<Integer>("mapping_type").has_value()) continue;

      auto mask_it = const_arg_masks.find(gvar.get());
      const ConstArgMask& const_args =
          mask_it != const_arg_masks.end() ? mask_it->second : empty_mask;
      MappingType mapping_type =
          ClassifyMappingType(gvar->name_hint, ffi::GetRef<tir::PrimFunc>(prim_func), const_args);
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
