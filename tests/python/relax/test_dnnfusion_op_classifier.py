# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for DNNFusion (PLDI '21) operator classifier — One-to-One scope.

Phase 1 narrow scope: this session only validates the One-to-One row of
paper Table 2. Other classes (One-to-Many / Many-to-Many / Reorganize /
Shuffle) are deferred. See:
  - MLC-oto/tvm/oto/01_op_classification_plan.md (plan)
  - docs/wiki/concepts/dnnfusion_taxonomy.md (full taxonomy)

Layers covered here:
  1. Per-op golden table  — every OtO row of paper Table 2 asserts class.
  2. Heuristic fallback   — kElemWise/kBroadcast → kOneToOne.
  3. Annotation pass      — AnnotateTIROpPattern → AnnotateDnnfusionClass
                             on an elemwise PrimFunc lands kOneToOne.
  4. Skip-condition guard — user-pinned kOneToOne (value 0) survives.
  5. ClassifyDnnfOp(Call) — direct Relax op call routes to AttrTable hit.
"""
import enum

import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import relax
from tvm.relax.analysis import (
    DnnfClassSource,
    DnnfOpClass,
    classify_dnnf_op,
    dnnf_class_from_op_pattern,
    infer_dnnf_class_from_tir,
)
from tvm.script import tir as T


class OpPatternKind(enum.IntEnum):
    kElemWise = 0
    kBroadcast = 1
    kInjective = 2
    kCommReduce = 3
    kOutEWiseFusable = 4
    kTuple = 7
    kOpaque = 8


# ----------------------------------------------------------------------------
# Layer 1 — Per-op golden table (paper Table 2, One-to-One row only)
# ----------------------------------------------------------------------------
# Every entry is an OtO assertion. ONNX-name → Relax-name mapping in
# parentheses where the names diverge.

_OTO_OPS = [
    # Binary arithmetic / logical
    "relax.add",            # ONNX Add
    "relax.subtract",
    "relax.multiply",
    "relax.divide",
    "relax.maximum",
    "relax.minimum",
    # Unary arithmetic / transcendental
    "relax.exp",            # ONNX Exp
    "relax.log",            # ONNX Log
    "relax.tanh",           # ONNX Tanh
    "relax.sqrt",           # ONNX Sqrt
    "relax.sin",            # ONNX Sin
    "relax.cos",            # ONNX Cos
    "relax.asin",           # ONNX Asin
    "relax.erf",            # ONNX Erf
    "relax.sigmoid",        # ONNX Sigmoid
    # Rounding / comparison / casting
    "relax.ceil",           # ONNX Ceil
    "relax.round",          # ONNX Round
    "relax.logical_not",    # ONNX Not
    "relax.astype",         # ONNX Cast
    "relax.clip",           # ONNX Clip
    "relax.where",          # ONNX Where
    # Slice / Split
    "relax.strided_slice",  # ONNX Slice
    "relax.split",          # ONNX Split
    # Surprising-but-correct (paper Table 2 explicitly puts these under OtO)
    "relax.concat",            # ONNX Concat — axis-offset routing → 1-1 mapping
    "relax.nn.batch_norm",     # ONNX BatchNormalization — inference is elementwise
    # Activations (paper "Relu" representative + family)
    "relax.nn.relu",        # ONNX Relu
    "relax.nn.leakyrelu",   # ONNX LeakyRelu
    "relax.nn.prelu",       # ONNX PRelu
]


@pytest.mark.parametrize("op_name", _OTO_OPS)
def test_oto_attr_table(op_name):
    """Verify the registered TDnnfOpClass attribute equals kOneToOne."""
    op = tvm.ir.Op.get(op_name)
    attr = op.get_attr("TDnnfOpClass")
    assert attr is not None, f"{op_name} has no TDnnfOpClass attribute"
    cls = DnnfOpClass(int(attr))
    assert cls == DnnfOpClass.ONE_TO_ONE, (
        f"{op_name}: expected ONE_TO_ONE, got {cls.name}"
    )


# ----------------------------------------------------------------------------
# Layer 2 — Heuristic fallback for the OtO row
# ----------------------------------------------------------------------------
# Only kElemWise heuristically maps to kOneToOne. kBroadcast deliberately
# maps to kOneToMany — see DnnfClassFromOpPattern for rationale (TVM's
# kBroadcast is structurally a fan-out, not 1-1). A negative test guards
# the kBroadcast → kOneToMany choice so it cannot silently regress.


def test_heuristic_elemwise_is_oto():
    assert dnnf_class_from_op_pattern(int(OpPatternKind.kElemWise)) == (
        DnnfOpClass.ONE_TO_ONE
    )


def test_heuristic_broadcast_is_not_oto():
    """Regression guard: kBroadcast must NOT map to kOneToOne (it's a fan-out)."""
    cls = dnnf_class_from_op_pattern(int(OpPatternKind.kBroadcast))
    assert cls != DnnfOpClass.ONE_TO_ONE
    assert cls == DnnfOpClass.ONE_TO_MANY


# ----------------------------------------------------------------------------
# Layer 3 — Annotation pass on an elemwise PrimFunc lands kOneToOne
# ----------------------------------------------------------------------------


def _elemwise_module():
    """Module with a single elemwise PrimFunc (op_pattern → kElemWise)."""

    @tvm.script.ir_module
    class Mod:
        @T.prim_func
        def add(
            x: T.handle, y: T.handle, z: T.handle, m: T.int64, n: T.int64
        ) -> None:
            T.func_attr({"global_symbol": "add"})
            X = T.match_buffer(x, (m, n))
            Y = T.match_buffer(y, (m, n))
            Z = T.match_buffer(z, (m, n))
            for i, j in T.grid(m, n):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Z[vi, vj] = X[vi, vj] + Y[vi, vj]

    return Mod


def test_annotate_pass_elemwise_to_oto():
    """End-to-end: AnnotateTIROpPattern → AnnotateDnnfusionClass yields OtO."""
    mod = _elemwise_module()
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert mod["add"].attrs["op_pattern"] == OpPatternKind.kElemWise

    mod = relax.transform.AnnotateDnnfusionClass()(mod)
    assert mod["add"].attrs["dnnf_op_class"] == int(DnnfOpClass.ONE_TO_ONE)


# ----------------------------------------------------------------------------
# Layer 4 — Skip-condition: user-pinned kOneToOne (value 0) is not overwritten
# ----------------------------------------------------------------------------


def test_annotate_pass_respects_user_pinned_oto():
    """The skip-condition gotcha: kOneToOne == 0 must not be misread as "unset".

    Plan §5.3 calls out HasNonzeroAttr as wrong for enum-valued attrs. The
    pass uses GetAttr(...).defined() instead. This test pins kOneToOne (= 0)
    on a function where the heuristic would otherwise compute something
    different, and asserts the user value survives the pass.
    """

    @tvm.script.ir_module
    class Mod:
        @T.prim_func
        def matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "matmul"})
            m = T.int32()
            n = T.int32()
            k = T.int32()
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))
            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = Mod
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    # Heuristic on matmul would produce a non-OtO class. User pins OtO (=0).
    pinned = mod["matmul"].with_attr("dnnf_op_class", int(DnnfOpClass.ONE_TO_ONE))
    mod = mod.clone()
    mod["matmul"] = pinned

    mod = relax.transform.AnnotateDnnfusionClass()(mod)
    # If the pass mistakenly used HasNonzeroAttr, kOneToOne (= 0) would be
    # overwritten. Assert the user pin survives.
    assert mod["matmul"].attrs["dnnf_op_class"] == int(DnnfOpClass.ONE_TO_ONE)


# ----------------------------------------------------------------------------
# Layer 5 — ClassifyDnnfOp on a direct Relax Call to an OtO op
# ----------------------------------------------------------------------------


def test_classify_oto_call():
    """Build a Relax Call to relax.add; classifier hits the AttrTable lookup."""
    x = relax.Var("x", relax.TensorStructInfo([16, 16], "float32"))
    call = relax.op.add(x, x)
    cls, source, reason = classify_dnnf_op(call)
    assert cls == DnnfOpClass.ONE_TO_ONE
    assert source == DnnfClassSource.ATTR_TABLE
    assert "relax.add" in reason


if __name__ == "__main__":
    tvm.testing.main()
