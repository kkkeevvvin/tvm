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
 * \file src/relax/op/dnnfusion_op_class_register.cc
 * \brief Per-op DnnfOpClass registrations from DNNFusion paper Table 2.
 *
 * \note Phase 1 scope: ONLY the One-to-One row of Table 2 is registered.
 *       Other classes (One-to-Many, Many-to-Many, Reorganize, Shuffle) are
 *       deferred to a later session — see
 *       docs/wiki/concepts/dnnfusion_taxonomy.md for the full taxonomy.
 *
 * Each TVM_REGISTER_OP attaches an Integer "TDnnfOpClass" attribute to a
 * Relax op so that Op::GetAttrMap<Integer>(kDnnfOpClassAttr) becomes an
 * O(1) lookup.
 */
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/relax/dnnfusion_op_class.h>

#define DNNF_REGISTER(op_name, cls)                                            \
  TVM_REGISTER_OP(op_name).set_attr<::tvm::Integer>(                           \
      ::tvm::relax::kDnnfOpClassAttr,                                          \
      ::tvm::Integer(static_cast<int>(::tvm::relax::DnnfOpClass::cls)))

// =========================================================================
// One-to-One — paper Table 2 (the entire row).
// =========================================================================
// "Add, Asin, BatchNormalization, Cast, Ceil, Clip, Concat, Cos, Erf, Exp,
//  Greater, LeakyRelu, Log, Not, PRelu, Reciprocal, Relu, Round, Sigmoid,
//  Sin, Slice, Split, Sqrt, Tanh, Where"
//
// Paper representative ops: Add, Relu.

// --- Binary arithmetic / logical (paper "Add" representative) -------------
// Paper Table 2 lists "Add" specifically; we register the full elementwise
// arithmetic family that shares its 1-1 index mapping (output[i] depends on
// same-position input element(s)). Each is OtO regardless of broadcasting —
// pure broadcast-as-fan-out is handled by relax.broadcast_to (deferred).
DNNF_REGISTER("relax.add", kOneToOne);
DNNF_REGISTER("relax.subtract", kOneToOne);
DNNF_REGISTER("relax.multiply", kOneToOne);
DNNF_REGISTER("relax.divide", kOneToOne);
DNNF_REGISTER("relax.maximum", kOneToOne);
DNNF_REGISTER("relax.minimum", kOneToOne);
DNNF_REGISTER("relax.power", kOneToOne);
DNNF_REGISTER("relax.mod", kOneToOne);
DNNF_REGISTER("relax.floor_divide", kOneToOne);
DNNF_REGISTER("relax.floor_mod", kOneToOne);
DNNF_REGISTER("relax.log_add_exp", kOneToOne);

// --- Unary arithmetic / transcendental ------------------------------------
// Paper Table 2: "Asin, Cos, Erf, Exp, Log, Sin, Sqrt, Tanh".
// Sibling functions (acos, atan, sinh, etc.) share the same 1-1 mapping.
DNNF_REGISTER("relax.abs", kOneToOne);
DNNF_REGISTER("relax.acos", kOneToOne);
DNNF_REGISTER("relax.acosh", kOneToOne);
DNNF_REGISTER("relax.asin", kOneToOne);
DNNF_REGISTER("relax.asinh", kOneToOne);
DNNF_REGISTER("relax.atan", kOneToOne);
DNNF_REGISTER("relax.atanh", kOneToOne);
DNNF_REGISTER("relax.cos", kOneToOne);
DNNF_REGISTER("relax.cosh", kOneToOne);
DNNF_REGISTER("relax.erf", kOneToOne);
DNNF_REGISTER("relax.exp", kOneToOne);
DNNF_REGISTER("relax.log", kOneToOne);
DNNF_REGISTER("relax.negative", kOneToOne);
DNNF_REGISTER("relax.rsqrt", kOneToOne);
DNNF_REGISTER("relax.sigmoid", kOneToOne);
DNNF_REGISTER("relax.sign", kOneToOne);
DNNF_REGISTER("relax.sin", kOneToOne);
DNNF_REGISTER("relax.sinh", kOneToOne);
DNNF_REGISTER("relax.sqrt", kOneToOne);
DNNF_REGISTER("relax.square", kOneToOne);
DNNF_REGISTER("relax.tan", kOneToOne);
DNNF_REGISTER("relax.tanh", kOneToOne);

// --- Rounding / comparison / casting --------------------------------------
// Paper Table 2: "Ceil, Round, Greater, Not, Cast, Clip, Where".
DNNF_REGISTER("relax.ceil", kOneToOne);
DNNF_REGISTER("relax.floor", kOneToOne);
DNNF_REGISTER("relax.round", kOneToOne);
DNNF_REGISTER("relax.trunc", kOneToOne);
DNNF_REGISTER("relax.logical_not", kOneToOne);
DNNF_REGISTER("relax.bitwise_not", kOneToOne);
DNNF_REGISTER("relax.astype", kOneToOne);  // ONNX Cast
DNNF_REGISTER("relax.clip", kOneToOne);
DNNF_REGISTER("relax.where", kOneToOne);

// --- Slice / Split — paper Table 2 lists both as OtO ----------------------
// Each output element comes from exactly one input element via offset/index
// arithmetic — index mapping is 1-1.
DNNF_REGISTER("relax.strided_slice", kOneToOne);
DNNF_REGISTER("relax.dynamic_strided_slice", kOneToOne);
DNNF_REGISTER("relax.split", kOneToOne);

// --- Surprising-but-correct: Concat, BatchNormalization -------------------
// These two are the OtO entries most likely to be miscategorized. Paper
// Table 2 explicitly places both under One-to-One — see
// docs/wiki/concepts/dnnfusion_taxonomy.md §五個容易誤判的條目.
DNNF_REGISTER("relax.concat", kOneToOne);            // axis-offset routing → 1-1
DNNF_REGISTER("relax.nn.batch_norm", kOneToOne);     // inference: (x-μ)/σ·γ+β

// --- Activations: Relu/LeakyRelu/PRelu (paper representatives) ------------
DNNF_REGISTER("relax.nn.relu", kOneToOne);
DNNF_REGISTER("relax.nn.leakyrelu", kOneToOne);
DNNF_REGISTER("relax.nn.prelu", kOneToOne);
DNNF_REGISTER("relax.nn.softplus", kOneToOne);

#undef DNNF_REGISTER
