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

#include "./dnnfusion_op_classifier.h"

#include <tvm/relax/expr.h>
#include <tvm/ir/op.h>

#include <algorithm>
#include <cctype>

namespace tvm {
namespace relax {

namespace {

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool Contains(const std::string& value, const std::string& needle) {
  return value.find(needle) != std::string::npos;
}

std::string GetNodeOpName(const IndexedForwardGraph::Node* node) {
  if (node == nullptr || node->ref == nullptr) return "";
  ObjectRef ref = ffi::GetRef<ObjectRef>(node->ref);
  if (const auto* call = ref.as<CallNode>()) {
    if (const auto* op = call->op.as<OpNode>()) {
      std::string op_name = op->name;
      if (op_name == "relax.call_tir" && !call->args.empty()) {
        if (const auto* gv = call->args[0].as<GlobalVarNode>()) {
          return gv->name_hint;
        }
      }
      return op_name;
    }
    if (const auto* gv = call->op.as<GlobalVarNode>()) {
      return gv->name_hint;
    }
    if (const auto* extern_func = call->op.as<ExternFuncNode>()) {
      return extern_func->global_symbol;
    }
  }
  if (ref.as<TupleNode>()) return "relax.Tuple";
  if (ref.as<TupleGetItemNode>()) return "relax.TupleGetItem";
  if (ref.as<VarNode>()) return "relax.Var";
  return "";
}

DnnfClassification Make(DnnfOpClass cls, DnnfClassSource source, const std::string& reason) {
  return DnnfClassification{cls, source, reason};
}

}  // namespace

std::string DnnfOpClassToString(DnnfOpClass cls) {
  switch (cls) {
    case DnnfOpClass::kOneToOne:
      return "OneToOne";
    case DnnfOpClass::kOneToMany:
      return "OneToMany";
    case DnnfOpClass::kManyToMany:
      return "ManyToMany";
    case DnnfOpClass::kReorganize:
      return "Reorganize";
    case DnnfOpClass::kShuffle:
      return "Shuffle";
    case DnnfOpClass::kUnknown:
      return "Unknown";
  }
  LOG(FATAL) << "Unknown DNNFusion op class";
}

DnnfClassification ClassifyDnnfOp(const IndexedForwardGraph::Node* node,
                                  OpPatternKind tvm_pattern) {
  std::string name = Lower(GetNodeOpName(node));

  if (name.empty() || tvm_pattern == kOpaque || tvm_pattern == kTuple) {
    return Make(DnnfOpClass::kUnknown, DnnfClassSource::kFallbackUnknown,
                "opaque, tuple, or unnamed Relax graph node");
  }

  if (Contains(name, "conv") || Contains(name, "dense") || Contains(name, "matmul") ||
      Contains(name, "batch_matmul") || Contains(name, "pool") || Contains(name, "softmax")) {
    return Make(DnnfOpClass::kManyToMany, DnnfClassSource::kPaperTable,
                "paper-table compute operator seed");
  }

  if (Contains(name, "reshape") || Contains(name, "squeeze") || Contains(name, "expand_dims") ||
      Contains(name, "flatten")) {
    return Make(DnnfOpClass::kReorganize, DnnfClassSource::kPaperTable,
                "paper-table shape-only data reorganization");
  }

  if (Contains(name, "transpose") || Contains(name, "permute") ||
      Contains(name, "layout_transform") || Contains(name, "concat") || Contains(name, "split") ||
      Contains(name, "gather") || Contains(name, "scatter") || Contains(name, "take")) {
    return Make(DnnfOpClass::kShuffle, DnnfClassSource::kPaperTable,
                "paper-table data movement or permutation operator");
  }

  if (tvm_pattern == kElemWise) {
    return Make(DnnfOpClass::kOneToOne, DnnfClassSource::kSemanticRule,
                "TVM elementwise pattern maps each output element to one input element");
  }

  if (tvm_pattern == kBroadcast) {
    return Make(DnnfOpClass::kOneToMany, DnnfClassSource::kSemanticRule,
                "TVM broadcast pattern maps one input element to multiple output elements");
  }

  if (tvm_pattern == kInjective) {
    return Make(DnnfOpClass::kReorganize, DnnfClassSource::kSemanticRule,
                "TVM injective pattern preserves one-to-one storage correspondence");
  }

  if (tvm_pattern == kOutEWiseFusable || tvm_pattern == kCommReduce) {
    return Make(DnnfOpClass::kManyToMany, DnnfClassSource::kSemanticRule,
                "TVM compute/reduction pattern has many-input dependence per output");
  }

  return Make(DnnfOpClass::kUnknown, DnnfClassSource::kFallbackUnknown,
              "no safe DNNFusion classification rule matched");
}

}  // namespace relax
}  // namespace tvm
