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
"""Tests for the DNNFusion mapping-type analyzer (Phase 1 / M1).

Covers:
  - DeriveMappingType table: OpPatternKind -> MappingType.
  - IRS (intermediate-result size) computation, including dynamic-shape fallback.
  - Seed-candidate filtering and IRS-ordered seed selection.
  - kOpaque -> Shuffle whitelist by callee (PrimFunc) name.

The IR fixtures use empty PrimFunc bodies (`T.evaluate(0)`) with the op_pattern
attribute set explicitly. The analyzer only reads op_pattern + StructInfo so the
bodies need not be meaningful.
"""

import sys

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.transform import (
    DNNFUSION_MAPPING_TYPES,
    analyze_dnnfusion_mapping_types,
)
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


# Mapping-type label aliases for readability in the tests.
OTO, OTM, MTM, REORG, SHUFFLE, BREAK = DNNFUSION_MAPPING_TYPES


def _by_name(rows):
    """Index rows by binding-var name for assertion convenience."""
    return {row["name"]: row for row in rows}


# ---------------------------------------------------------------------------
# Derivation table coverage
# ---------------------------------------------------------------------------


def test_derive_mapping_type_table():
    """Every OpPatternKind that appears in the post-LegalizeOps IR must derive
    to the documented mapping type (plan §4.3 Step 1)."""

    @I.ir_module
    class M:
        # OtO  — kElemWise = 0
        @T.prim_func(private=True)
        def relu(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        # OtM  — kBroadcast = 1
        @T.prim_func(private=True)
        def bcast(A: T.Buffer((1, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 1, "tir.noalias": True})
            T.evaluate(0)

        # Reorg — kInjective = 2
        @T.prim_func(private=True)
        def transpose(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 2, "tir.noalias": True})
            T.evaluate(0)

        # MtM   — kCommReduce = 3
        @T.prim_func(private=True)
        def sum_reduce(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4,), "float32")):
            T.func_attr({"op_pattern": 3, "tir.noalias": True})
            T.evaluate(0)

        # MtM   — kOutEWiseFusable = 4
        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((4, 4), "float32"),
            B: T.Buffer((4, 4), "float32"),
            C: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            T.evaluate(0)

        # break — kOpaque = 8 (callee name not on Shuffle whitelist)
        @T.prim_func(private=True)
        def opaque_op(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((4, 4), "float32"), y: R.Tensor((1, 4), "float32")):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                b = R.call_tir(cls.bcast, (y,), out_sinfo=R.Tensor((4, 4), "float32"))
                c = R.call_tir(cls.transpose, (a,), out_sinfo=R.Tensor((4, 4), "float32"))
                d = R.call_tir(cls.sum_reduce, (c,), out_sinfo=R.Tensor((4,), "float32"))
                e = R.call_tir(cls.matmul, (a, b), out_sinfo=R.Tensor((4, 4), "float32"))
                f = R.call_tir(cls.opaque_op, (e,), out_sinfo=R.Tensor((4, 4), "float32"))
                R.output(f)
            return f

    rows = _by_name(analyze_dnnfusion_mapping_types(M))
    assert rows["a"]["mapping_name"] == OTO
    assert rows["b"]["mapping_name"] == OTM
    assert rows["c"]["mapping_name"] == REORG
    assert rows["d"]["mapping_name"] == MTM  # kCommReduce
    assert rows["e"]["mapping_name"] == MTM  # kOutEWiseFusable
    assert rows["f"]["mapping_name"] == BREAK  # kOpaque without whitelist hit


# ---------------------------------------------------------------------------
# Shuffle whitelist
# ---------------------------------------------------------------------------


def test_opaque_shuffle_whitelist():
    """kOpaque PrimFuncs whose name matches the shuffle whitelist substrings
    (take / gather / scatter / embedding) classify as Shuffle, not break.

    The whitelist matches by substring so prefixed/suffixed variants like
    `gather_nd`, `scatter_elements`, `embedding_fwd` all get caught."""

    @I.ir_module
    class M:
        # Each PrimFunc is kOpaque (op_pattern=8) and named so the substring
        # check in IsShuffleByCalleeName fires.
        @T.prim_func(private=True)
        def take(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def gather_nd(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def scatter_elements(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def embedding_fwd(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        # Control: same op_pattern, name not on whitelist -> classifies as break.
        @T.prim_func(private=True)
        def some_opaque(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 8, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((4, 4), "float32")):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.take, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                b = R.call_tir(cls.gather_nd, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                c = R.call_tir(cls.scatter_elements, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                d = R.call_tir(cls.embedding_fwd, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                e = R.call_tir(cls.some_opaque, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                R.output(a, b, c, d, e)
            return (a, b, c, d, e)

    rows = _by_name(analyze_dnnfusion_mapping_types(M))
    assert rows["a"]["mapping_name"] == SHUFFLE  # take
    assert rows["b"]["mapping_name"] == SHUFFLE  # gather_nd
    assert rows["c"]["mapping_name"] == SHUFFLE  # scatter_elements
    assert rows["d"]["mapping_name"] == SHUFFLE  # embedding_fwd
    assert rows["e"]["mapping_name"] == BREAK    # control: not on whitelist


# ---------------------------------------------------------------------------
# IRS computation
# ---------------------------------------------------------------------------


def test_irs_bytes_static_shape():
    """IRS in bytes = numel * dtype.bytes for a static-shape TensorStructInfo."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def relu_small(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def relu_large(A: T.Buffer((8, 8), "float16"), B: T.Buffer((8, 8), "float16")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((8, 8), "float16")):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu_small, (x,), out_sinfo=R.Tensor((2, 3), "float32"))
                b = R.call_tir(cls.relu_large, (y,), out_sinfo=R.Tensor((8, 8), "float16"))
                R.output(a, b)
            return (a, b)

    rows = _by_name(analyze_dnnfusion_mapping_types(M))
    assert rows["a"]["irs_bytes"] == 2 * 3 * 4   # float32 = 4 bytes
    assert rows["b"]["irs_bytes"] == 8 * 8 * 2   # float16 = 2 bytes
    assert rows["a"]["is_seed_cand"] is True
    assert rows["b"]["is_seed_cand"] is True


def test_irs_bytes_dynamic_shape_falls_back_to_max():
    """Dynamic shape => IRS = INT64_MAX, so the binding is *not* a seed
    candidate even if its mapping type is OtO."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def relu_dyn(var_A: T.handle, var_B: T.handle):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor(("n", 4), "float32")) -> R.Tensor(("n", 4), "float32"):
            n = T.int64()
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu_dyn, (x,), out_sinfo=R.Tensor((n, 4), "float32"))
                R.output(a)
            return a

    rows = _by_name(analyze_dnnfusion_mapping_types(M))
    int64_max = (1 << 63) - 1
    assert rows["a"]["mapping_name"] == OTO
    assert rows["a"]["irs_bytes"] == int64_max
    # Seed candidate requires both OtO and a finite IRS.
    assert rows["a"]["is_seed_cand"] is False


# ---------------------------------------------------------------------------
# Seed selection ordering
# ---------------------------------------------------------------------------


def test_seed_order_by_irs_ascending():
    """Phase 1 / M2 will pick seeds from a min-heap over OtO bindings keyed by
    IRS. Verify here that the data the heap will consume is correct: among
    OtO bindings, sorting by IRS ascending picks the smallest first, ties
    broken by post-DFS index (deterministic)."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def relu_a(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def relu_b(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def relu_c(A: T.Buffer((8, 8), "float32"), B: T.Buffer((8, 8), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        # MtM — must NOT be a seed candidate even though its IRS is small.
        @T.prim_func(private=True)
        def mm(
            A: T.Buffer((2, 2), "float32"),
            B: T.Buffer((2, 2), "float32"),
            C: T.Buffer((2, 2), "float32"),
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(
            x_a: R.Tensor((16, 16), "float32"),
            x_b: R.Tensor((4, 4), "float32"),
            x_c: R.Tensor((8, 8), "float32"),
            y: R.Tensor((2, 2), "float32"),
        ):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu_a, (x_a,), out_sinfo=R.Tensor((16, 16), "float32"))
                b = R.call_tir(cls.relu_b, (x_b,), out_sinfo=R.Tensor((4, 4), "float32"))
                c = R.call_tir(cls.relu_c, (x_c,), out_sinfo=R.Tensor((8, 8), "float32"))
                d = R.call_tir(cls.mm, (y, y), out_sinfo=R.Tensor((2, 2), "float32"))
                R.output(a, b, c, d)
            return (a, b, c, d)

    rows = analyze_dnnfusion_mapping_types(M)
    seed_cands = [r for r in rows if r["is_seed_cand"]]
    seed_order = sorted(seed_cands, key=lambda r: r["irs_bytes"])
    seed_names = [r["name"] for r in seed_order]
    # Smallest IRS: relu_b (4*4*4=64) < relu_c (8*8*4=256) < relu_a (16*16*4=1024).
    assert seed_names == ["b", "c", "a"]
    # The MtM matmul binding "d" must not appear among seed candidates.
    assert "d" not in seed_names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
