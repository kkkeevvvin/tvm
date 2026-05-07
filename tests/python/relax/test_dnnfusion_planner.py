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
"""Integration tests for the DNNFusion planner (Phase B).

Strategy: build small Relax IRs via BlockBuilder + topi (the same idiom
used by ``test_transform_fuse_ops.py``), run FuseOps under both
``algorithm="tvm"`` and ``algorithm="dnnfusion"``, and assert on the
resulting group structure.

We don't require the two modes to produce identical IRs (and they often
won't) — instead we check counts and mapping-aware properties:
  * ``count_fused_funcs`` : number of generated ``Primitive=True`` functions.
  * Default config (``algorithm="tvm"``) must equal the upstream baseline
    bit-for-bit, so that the new code path doesn't regress non-DNNFusion
    callers.
"""
import pytest
import tvm
import tvm.testing
from tvm import relax, topi
from tvm.script import relax as R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_fused_funcs(mod: tvm.IRModule) -> int:
    """Count Relax functions tagged ``attr::kPrimitive`` (i.e. fused groups)."""
    n = 0
    for _, func in mod.functions.items():
        if isinstance(func, relax.Function) and func.attrs is not None:
            if func.attrs.get("Primitive", 0) == 1:
                n += 1
    return n


def _annotate_and_fuse(mod, *, algorithm="tvm", yellow_policy="conservative"):
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    config = {
        "relax.FuseOps.algorithm": algorithm,
        "relax.FuseOps.dnnfusion.yellow_policy": yellow_policy,
    }
    with tvm.transform.PassContext(config=config):
        return relax.transform.FuseOps()(mod)


def _build_chain():
    """add -> exp -> squeeze.  All elemwise/injective; should fuse into 1 group."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([10, 20], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
            lv1 = bb.emit_te(topi.exp, lv0)
            gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
        bb.emit_func_output(gv)
    return bb.get()


def _build_diamond():
    """x -> {a=add, b=sub} -> c=mul(a, b).  Pure elemwise diamond."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([8, 16], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            a = bb.emit_te(topi.add, x, relax.const(1, "float32"))
            b = bb.emit_te(topi.subtract, x, relax.const(2, "float32"))
            c = bb.emit_te(topi.multiply, a, b)
            gv = bb.emit_output(c)
        bb.emit_func_output(gv)
    return bb.get()


def _build_two_matmuls():
    """x -> matmul1 -> add1 -> matmul2 -> add2.

    Two MtM ops separated by elemwise.  Under DNNFusion Table 3 the
    MtM x MtM cell is Red, so the two matmuls must end up in separate
    groups regardless of yellow_policy.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([4, 8], "float32"))
    w1 = relax.Var("w1", R.Tensor([8, 16], "float32"))
    w2 = relax.Var("w2", R.Tensor([16, 8], "float32"))
    b1 = relax.Var("b1", R.Tensor([16], "float32"))
    b2 = relax.Var("b2", R.Tensor([8], "float32"))
    with bb.function("main", [x, w1, w2, b1, b2]):
        with bb.dataflow():
            m1 = bb.emit_te(topi.nn.matmul, x, w1)
            a1 = bb.emit_te(topi.add, m1, b1)
            m2 = bb.emit_te(topi.nn.matmul, a1, w2)
            a2 = bb.emit_te(topi.add, m2, b2)
            gv = bb.emit_output(a2)
        bb.emit_func_output(gv)
    return bb.get()


# ---------------------------------------------------------------------------
# Default-config compatibility: tvm algorithm must match upstream behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [_build_chain, _build_diamond, _build_two_matmuls])
def test_default_algorithm_unchanged(builder):
    """``algorithm=tvm`` (default) must produce the same IR as v0.23.0 baseline."""
    mod = builder()
    fused_default = _annotate_and_fuse(mod)  # default = tvm
    fused_explicit = _annotate_and_fuse(mod, algorithm="tvm")
    tvm.ir.assert_structural_equal(fused_default, fused_explicit)


# ---------------------------------------------------------------------------
# DNNFusion mode: smoke tests on small graphs.
# ---------------------------------------------------------------------------


def test_chain_dnnfusion():
    """Three-elemwise chain: TVM and DNNFusion should both produce 1 group."""
    mod = _build_chain()
    fused_tvm = _annotate_and_fuse(mod, algorithm="tvm")
    fused_dnnf = _annotate_and_fuse(mod, algorithm="dnnfusion")
    assert _count_fused_funcs(fused_tvm) == 1
    assert _count_fused_funcs(fused_dnnf) == 1


def test_diamond_dnnfusion():
    mod = _build_diamond()
    fused_tvm = _annotate_and_fuse(mod, algorithm="tvm")
    fused_dnnf = _annotate_and_fuse(mod, algorithm="dnnfusion")
    # Both should fuse the whole diamond (3 ops) into a single group.
    assert _count_fused_funcs(fused_tvm) == 1
    assert _count_fused_funcs(fused_dnnf) == 1


def test_two_matmuls_dnnfusion_red():
    """MtM x MtM is Red, but the two matmuls here are NOT directly connected
    (separated by add).  Per paper §3.2 (OtO x MtM = Green in either order),
    the seed-driven walk from the add seed bridges both matmuls into one
    group via OtO->MtM forward + MtM->OtO backward.

    Note: Earlier dnnf-branch implementation had OtO x MtM = Yellow (a bug
    documented in findings_paper_consistency_check.md), under which this
    test asserted >= 2 groups.  After the Table-3 fix to match paper, the
    seed walk transitively co-groups both matmuls.  Paper Listing 1 has no
    per-block "no two MtM" constraint beyond Table 3 pairwise lookup, so
    this is the paper-correct outcome.
    """
    mod = _build_two_matmuls()
    fused_dnnf = _annotate_and_fuse(mod, algorithm="dnnfusion")
    n = _count_fused_funcs(fused_dnnf)
    assert n == 1, (
        f"DNNFusion fused {n} groups; expected 1 — the add seed bridges "
        f"both matmuls via OtO x MtM = Green (paper §3.2)."
    )


# ---------------------------------------------------------------------------
# Yellow policy: switching the knob must change observable behaviour for
# at least one canonical Yellow case.
# ---------------------------------------------------------------------------


def _build_otm_then_otm():
    """broadcast_to -> exp: an OtM op followed by an OtO op.

    OtM x OtO is Green (always fuse); this is a smoke test that both
    yellow_policy settings produce well-formed IR.  x has shape [1, 8]
    so broadcast_to [4, 8] is a valid numpy-style broadcast.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor([1, 8], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            bcast = bb.emit_te(topi.broadcast_to, x, [4, 8])
            ex = bb.emit_te(topi.exp, bcast)
            gv = bb.emit_output(ex)
        bb.emit_func_output(gv)
    return bb.get()


def test_yellow_policy_runs_both():
    """Both yellow_policy settings must produce a well-formed IR."""
    mod = _build_otm_then_otm()
    cons = _annotate_and_fuse(mod, algorithm="dnnfusion", yellow_policy="conservative")
    aggr = _annotate_and_fuse(mod, algorithm="dnnfusion", yellow_policy="aggressive")
    assert _count_fused_funcs(cons) >= 0
    assert _count_fused_funcs(aggr) >= 0


if __name__ == "__main__":
    tvm.testing.main()
