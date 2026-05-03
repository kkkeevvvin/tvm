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
"""Tests for the DNNFusion mapping-check matrix and edge walker (Phase 1 / M2).

Covers (per dnnfusion_fuse_ops_plan.md §4.3 Step 3 + §4.5 M2):

  - The full 5x5 mapping matrix (paper Table 3) returned by
    ``dnnfusion_mapping_check`` matches the documented relation per cell.
  - Either input being ``"break"`` collapses to ``"break"`` (we never invent
    a fusion across an unknown boundary).
  - Phase 1's ``phase1_allow`` flag is True iff the cell is ``"thru"``;
    every ``"dep"`` cell is rejected outright until the Phase 2 latency
    oracle lands.
  - The Python wrapper accepts both string labels and the underlying enum
    integer values.
  - ``analyze_dnnfusion_fuse_edges`` walks the BuildIndexedForwardGraph
    output and reports the right relation on every dataflow edge of a small
    synthetic IR module, including parameter-input edges.
"""

import sys

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.transform import (
    DNNFUSION_FUSE_RELATIONS,
    DNNFUSION_MAPPING_TYPES,
    analyze_dnnfusion_fuse_edges,
    dnnfusion_mapping_check,
)
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


# Mapping-type label aliases for readability in the tests.
OTO, OTM, MTM, REORG, SHUFFLE, BREAK = DNNFUSION_MAPPING_TYPES
THRU, DEP, BRK = DNNFUSION_FUSE_RELATIONS


# ---------------------------------------------------------------------------
# 5x5 matrix coverage (paper Table 3)
# ---------------------------------------------------------------------------

# Per-cell expected relation. Indexed by (producer_label, consumer_label).
# Source: dnnfusion_fuse_ops_plan.md §3.3.
EXPECTED_MATRIX = {
    (OTO,     OTO):     THRU, (OTO,     OTM):     THRU, (OTO,     MTM):     THRU,
    (OTO,     REORG):   THRU, (OTO,     SHUFFLE): BRK,
    (OTM,     OTO):     THRU, (OTM,     OTM):     THRU, (OTM,     MTM):     DEP,
    (OTM,     REORG):   DEP,  (OTM,     SHUFFLE): BRK,
    (MTM,     OTO):     THRU, (MTM,     OTM):     DEP,  (MTM,     MTM):     DEP,
    (MTM,     REORG):   DEP,  (MTM,     SHUFFLE): BRK,
    (REORG,   OTO):     THRU, (REORG,   OTM):     THRU, (REORG,   MTM):     DEP,
    (REORG,   REORG):   THRU, (REORG,   SHUFFLE): BRK,
    (SHUFFLE, OTO):     DEP,  (SHUFFLE, OTM):     BRK,  (SHUFFLE, MTM):     BRK,
    (SHUFFLE, REORG):   BRK,  (SHUFFLE, SHUFFLE): BRK,
}


def test_mapping_check_full_5x5_matrix():
    """Every cell of the DNNFusion 5x5 matrix must match the paper Table 3."""
    assert len(EXPECTED_MATRIX) == 25, "the test table itself must cover all 25 cells"
    for (producer, consumer), expected in EXPECTED_MATRIX.items():
        result = dnnfusion_mapping_check(producer, consumer)
        assert result["name"] == expected, (
            f"mapping_check({producer}, {consumer}) returned {result['name']}, "
            f"expected {expected}"
        )


def test_mapping_check_phase1_allow_only_thru():
    """Phase 1 accepts a fusion only when the relation is 'thru'. 'dep' is
    intentionally rejected at this milestone (the latency oracle that would
    decide it is Phase 2 work)."""
    for (producer, consumer), expected in EXPECTED_MATRIX.items():
        result = dnnfusion_mapping_check(producer, consumer)
        assert result["phase1_allow"] is (expected == THRU), (
            f"phase1_allow({producer}, {consumer}) inconsistent with relation "
            f"{expected!r}"
        )


def test_mapping_check_break_collapses():
    """If either side is 'break' (i.e. the op was not classified at all),
    the relation must be 'break' regardless of the other side. This guards
    against the matrix lookup silently treating kBreak as a valid mapping
    type and indexing past the table."""
    for label in DNNFUSION_MAPPING_TYPES:
        result = dnnfusion_mapping_check(BREAK, label)
        assert result["name"] == BRK, f"break -> {label} should be break, got {result['name']}"
        result = dnnfusion_mapping_check(label, BREAK)
        assert result["name"] == BRK, f"{label} -> break should be break, got {result['name']}"


def test_mapping_check_accepts_int_and_string_args():
    """Python wrapper takes either a label string or the enum integer; both
    paths must agree."""
    for i, label in enumerate(DNNFUSION_MAPPING_TYPES):
        result_str = dnnfusion_mapping_check(label, label)
        result_int = dnnfusion_mapping_check(i, i)
        assert result_str == result_int, (
            f"string vs int args disagree for ({label}, {label}): "
            f"{result_str} vs {result_int}"
        )


def test_mapping_check_rejects_unknown_label():
    """An unknown string label must error out instead of silently being
    coerced to some default."""
    with pytest.raises(ValueError):
        dnnfusion_mapping_check("not_a_mapping_type", OTO)


# ---------------------------------------------------------------------------
# Edge walker on a synthetic IR module
# ---------------------------------------------------------------------------


def _edges_by_pair(rows):
    """Index edge-walker rows by (producer_name, consumer_name) for assertion
    convenience. Used by the edge-walker tests to look up the relation on a
    specific (producer, consumer) pair."""
    return {(row["producer"], row["consumer"]): row for row in rows}


def test_analyze_fuse_edges_chain():
    """End-to-end check of the edge walker against a hand-traced graph.

    Layout (post-LegalizeOps style):
        x -> relu (OtO)  -+
                          +-> matmul (MtM) -> sigmoid (OtO) -> transpose (Reorg)
        y -> bcast (OtM) -+

    Expected edges (focus on binding -> binding):
        relu     -> matmul    : OtO   -> MtM    = thru
        bcast    -> matmul    : OtM   -> MtM    = dep
        matmul   -> sigmoid   : MtM   -> OtO    = thru
        sigmoid  -> transpose : OtO   -> Reorg  = thru

    Parameter -> binding edges show up too, with producer="<param>" and
    relation="break" (we never fuse "across a parameter")."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def relu(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def bcast(A: T.Buffer((1, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 1, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((4, 4), "float32"),
            B: T.Buffer((4, 4), "float32"),
            C: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def sigmoid(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def transpose(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 2, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((4, 4), "float32"), y: R.Tensor((1, 4), "float32")):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                b = R.call_tir(cls.bcast, (y,), out_sinfo=R.Tensor((4, 4), "float32"))
                c = R.call_tir(cls.matmul, (a, b), out_sinfo=R.Tensor((4, 4), "float32"))
                d = R.call_tir(cls.sigmoid, (c,), out_sinfo=R.Tensor((4, 4), "float32"))
                e = R.call_tir(cls.transpose, (d,), out_sinfo=R.Tensor((4, 4), "float32"))
                R.output(e)
            return e

    rows = analyze_dnnfusion_fuse_edges(M)
    edges = _edges_by_pair(rows)

    # Each binding-to-binding edge of interest must exist exactly once and
    # match the mapping-type table.
    expected_edges = {
        ("a", "c"): (OTO,   MTM,   THRU),  # relu     -> matmul
        ("b", "c"): (OTM,   MTM,   DEP),   # bcast    -> matmul
        ("c", "d"): (MTM,   OTO,   THRU),  # matmul   -> sigmoid
        ("d", "e"): (OTO,   REORG, THRU),  # sigmoid  -> transpose
    }
    for pair, (prod_mt, cons_mt, rel) in expected_edges.items():
        assert pair in edges, f"edge {pair} missing from analyzer output"
        row = edges[pair]
        assert row["producer_mapping_name"] == prod_mt
        assert row["consumer_mapping_name"] == cons_mt
        assert row["relation_name"] == rel
        assert row["phase1_allow"] is (rel == THRU)


def test_analyze_fuse_edges_includes_parameter_inputs():
    """Edges originating at a function parameter (no Relax binding) are
    surfaced with producer='<param>' and relation='break'. This is what the
    M4 partitioner will rely on to know it must stop walking backwards once
    it hits an input. Test target: relu's input edge."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def relu(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 0, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((4, 4), "float32")):
            cls = M
            with R.dataflow():
                a = R.call_tir(cls.relu, (x,), out_sinfo=R.Tensor((4, 4), "float32"))
                R.output(a)
            return a

    rows = analyze_dnnfusion_fuse_edges(M)
    # Exactly one edge: x (parameter) -> a (relu binding).
    param_edges = [r for r in rows if r["producer"] == "<param>"]
    assert len(param_edges) == 1, f"expected one param edge, got {param_edges}"
    edge = param_edges[0]
    assert edge["consumer"] == "a"
    # Parameters have no op_pattern so they collapse to 'break' regardless of
    # the consumer side.
    assert edge["producer_mapping_name"] == BREAK
    assert edge["consumer_mapping_name"] == OTO
    assert edge["relation_name"] == BRK
    assert edge["phase1_allow"] is False


def test_analyze_fuse_edges_phase1_dep_rejection():
    """The matrix calls OtM -> MtM 'dep', and Phase 1 must reject it. This
    test isolates exactly that edge so a regression in the Phase 1
    decision wrapper would surface here directly (rather than buried in the
    chain test above)."""

    @I.ir_module
    class M:
        @T.prim_func(private=True)
        def bcast(A: T.Buffer((1, 4), "float32"), B: T.Buffer((4, 4), "float32")):
            T.func_attr({"op_pattern": 1, "tir.noalias": True})
            T.evaluate(0)

        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((4, 4), "float32"),
            B: T.Buffer((4, 4), "float32"),
            C: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"op_pattern": 4, "tir.noalias": True})
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((4, 4), "float32"), y: R.Tensor((1, 4), "float32")):
            cls = M
            with R.dataflow():
                b = R.call_tir(cls.bcast, (y,), out_sinfo=R.Tensor((4, 4), "float32"))
                c = R.call_tir(cls.matmul, (x, b), out_sinfo=R.Tensor((4, 4), "float32"))
                R.output(c)
            return c

    rows = analyze_dnnfusion_fuse_edges(M)
    edges = _edges_by_pair(rows)
    assert ("b", "c") in edges, f"bcast->matmul edge missing in {edges.keys()}"
    edge = edges[("b", "c")]
    assert edge["producer_mapping_name"] == OTM
    assert edge["consumer_mapping_name"] == MTM
    assert edge["relation_name"] == DEP
    # Phase 1 rejects 'dep' even though Phase 2 will profile it.
    assert edge["phase1_allow"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
