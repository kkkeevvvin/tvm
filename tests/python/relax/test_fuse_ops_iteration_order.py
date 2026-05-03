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
"""exp2 -- partitioner outer-loop iteration-order coverage.

Verifies that the new ``relax.FuseOps.iteration_order`` PassContext config:
  1. Keeps the default ("topological") path bit-identical to the unconfigured
     baseline (regression guard for v0.23.0 behavior).
  2. Wires "dnnfusion_seed" through ComputeSeedOrder, with the resulting
     IRModule remaining well-formed across hand-crafted graph shapes (chain,
     diamond, multi-branch, tuple) and through the postponed-fusing path
     (plan section 6 risk 1).

The test does *not* assert exact group composition under the seed ordering --
that is what the experiment itself is meant to characterize. The goal here is
to lock down the regression surface and the wiring.

Built using ``@tvm.script.ir_module`` directly rather than BlockBuilder. The
worktree's BlockBuilder is incompatible with the system-installed tvm_ffi
(0.1.10) at v0.23.0; TVMScript has no such dependency.
"""

import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R


# ---------------------------------------------------------------------------
# IR builders (each returns an IRModule)
# ---------------------------------------------------------------------------


def _mod_chain():
    """add -> relu -> conv -> relu -> add (mostly elemwise, one conv)."""

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 8, 16, 16), "float32"),
            w: R.Tensor((8, 8, 3, 3), "float32"),
        ) -> R.Tensor((1, 8, 16, 16), "float32"):
            with R.dataflow():
                lv0 = R.add(x, R.const(1.0, "float32"))
                lv1 = R.nn.relu(lv0)
                lv2 = R.nn.conv2d(lv1, w, padding=(1, 1, 1, 1))
                lv3 = R.nn.relu(lv2)
                gv = R.add(lv3, R.const(2.0, "float32"))
                R.output(gv)
            return gv

    return Mod


def _mod_diamond():
    """x -> conv -> (relu, sigmoid) -> add."""

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 8, 16, 16), "float32"),
            w: R.Tensor((8, 8, 3, 3), "float32"),
        ) -> R.Tensor((1, 8, 16, 16), "float32"):
            with R.dataflow():
                lv0 = R.nn.conv2d(x, w, padding=(1, 1, 1, 1))
                lv1 = R.nn.relu(lv0)
                lv2 = R.sigmoid(lv0)
                gv = R.add(lv1, lv2)
                R.output(gv)
            return gv

    return Mod


def _mod_multi_branch_shared_tail():
    """Two convs feeding a shared elemwise tail."""

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 8, 16, 16), "float32"),
            w1: R.Tensor((8, 8, 3, 3), "float32"),
            w2: R.Tensor((8, 8, 3, 3), "float32"),
        ) -> R.Tensor((1, 8, 16, 16), "float32"):
            with R.dataflow():
                lv0 = R.nn.conv2d(x, w1, padding=(1, 1, 1, 1))
                lv1 = R.nn.conv2d(x, w2, padding=(1, 1, 1, 1))
                lv2 = R.add(lv0, lv1)
                gv = R.nn.relu(lv2)
                R.output(gv)
            return gv

    return Mod


def _mod_tuple():
    """split -> (relu, relu) -> concat (exercises phase 1 / phase 2)."""

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 8, 16, 16), "float32"),
        ) -> R.Tensor((1, 8, 16, 16), "float32"):
            with R.dataflow():
                tup = R.split(x, 2, axis=1)
                lv0 = tup[0]
                lv1 = tup[1]
                r0 = R.nn.relu(lv0)
                r1 = R.nn.relu(lv1)
                gv = R.concat((r0, r1), axis=1)
                R.output(gv)
            return gv

    return Mod


def _mod_postponed():
    """Two parallel elemwise chains feeding a concat -> elemwise tail. The
    concat consumer is fed by independent groups, exercising the postpone
    map. Drives plan section 6 risk 1."""

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 4, 8, 8), "float32"),
            y: R.Tensor((1, 4, 8, 8), "float32"),
        ) -> R.Tensor((1, 8, 8, 8), "float32"):
            with R.dataflow():
                a0 = R.add(x, R.const(1.0, "float32"))
                a1 = R.nn.relu(a0)
                b0 = R.add(y, R.const(2.0, "float32"))
                b1 = R.nn.relu(b0)
                cat = R.concat((a1, b1), axis=1)
                gv = R.nn.relu(cat)
                R.output(gv)
            return gv

    return Mod


CASES = {
    "chain": _mod_chain,
    "diamond": _mod_diamond,
    "multi_branch_shared_tail": _mod_multi_branch_shared_tail,
    "tuple": _mod_tuple,
    "postponed_fusing": _mod_postponed,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_fuse(mod, order_mode):
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    config = {}
    if order_mode is not None:
        config["relax.FuseOps.iteration_order"] = order_mode
    with tvm.transform.PassContext(config=config):
        return relax.transform.FuseOps()(mod)


def _group_signatures(mod):
    """Stable summary of fused-function signatures: (name, arity)."""
    sigs = []
    for gv, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if func.attrs is None or not func.attrs.get("Primitive", False):
            continue
        sigs.append((gv.name_hint, len(func.params)))
    sigs.sort()
    return tuple(sigs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", list(CASES))
def test_default_matches_topological(case_name):
    """Default config must equal explicit "topological" (regression guard)."""
    mod = CASES[case_name]()
    mod_default = _run_fuse(mod, order_mode=None)
    mod_topo = _run_fuse(mod, order_mode="topological")
    tvm.ir.assert_structural_equal(mod_default, mod_topo)


@pytest.mark.parametrize("case_name", list(CASES))
def test_seed_mode_well_formed(case_name):
    """dnnfusion_seed must produce a well-formed IRModule on every shape."""
    mod = CASES[case_name]()
    mod_seed = _run_fuse(mod, order_mode="dnnfusion_seed")
    assert relax.analysis.well_formed(mod_seed), (
        f"case {case_name!r}: dnnfusion_seed produced non-well-formed IR:\n{mod_seed}"
    )
    sigs = _group_signatures(mod_seed)
    # Each test case has at least one fusable region.
    assert sigs, f"case {case_name!r}: no fused groups under dnnfusion_seed"


def test_postponed_fusing_does_not_leak():
    """Plan section 6 risk 1: under reordered iteration, the postpone map must
    not strand bindings. We verify the IR is well-formed and every GlobalVar
    referenced from main resolves inside the module."""
    mod = _mod_postponed()
    mod_seed = _run_fuse(mod, order_mode="dnnfusion_seed")
    assert relax.analysis.well_formed(mod_seed)

    main = mod_seed["main"]
    referenced = set()

    def _collect(expr):
        if isinstance(expr, relax.Call):
            if isinstance(expr.op, tvm.ir.GlobalVar):
                referenced.add(expr.op.name_hint)
            for arg in expr.args:
                if isinstance(arg, tvm.ir.GlobalVar):
                    referenced.add(arg.name_hint)

    relax.analysis.post_order_visit(main, _collect)
    for name in referenced:
        assert name in mod_seed, (
            f"GlobalVar {name!r} referenced from main but missing in module "
            "(possible postpone-map leak under dnnfusion_seed)"
        )


def test_fuse_trace_emits_under_seed_mode():
    """Step 4 wiring: the trace populates and reset() works."""
    reset = tvm.get_global_func("relax.transform.ResetFuseTrace", allow_missing=True)
    get_trace = tvm.get_global_func("relax.transform.GetLastFuseTrace", allow_missing=True)
    assert reset is not None and get_trace is not None, (
        "FuseTrace FFI hooks must be registered when dnnfusion_seed.cc is built"
    )

    mod = _mod_chain()

    reset()
    _run_fuse(mod, order_mode="dnnfusion_seed")
    seed_trace = list(get_trace())

    reset()
    _run_fuse(mod, order_mode="topological")
    topo_trace = list(get_trace())

    assert seed_trace, "expected at least one fuse-trace row under dnnfusion_seed"
    assert topo_trace, "expected at least one fuse-trace row under topological"

    expected_keys = {"phase", "src_nid", "sink_nid", "group_size_after"}
    for row in seed_trace + topo_trace:
        assert expected_keys <= set(row.keys()), (
            f"trace row missing keys: have {set(row.keys())}, want {expected_keys}"
        )


if __name__ == "__main__":
    tvm.testing.main()
