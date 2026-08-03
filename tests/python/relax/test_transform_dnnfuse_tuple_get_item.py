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

"""RunDNNFuse fuses through TupleGetItem.

A multi-output op (split, or any call_tir with several out_sinfos) reaches its
consumers only through a TupleGetItem binding. Those bindings are not call_tir
to an AnnotateTIROpMappingType-annotated PrimFunc, so they have no Table 2
mapping type of their own; left at the node default of ``kMappingOpaque`` they
would break every fusion chain at the multi-output op -- in GPT-2 that is the
attention QKV split, whose three results each feed a reshape + transpose.

A TupleGetItem carries no computation, so ``GraphCreator::VisitTupleGetItem``
classifies it ``kReorganize`` (the mapping-type counterpart of the
``kInjective`` OpPatternKind it already carried), letting RunDNNFuse expand
straight through it.
"""

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.analysis import well_formed


def _pipeline(mod):
    """The demos' pre-fuse pipeline, minus the passes irrelevant here."""
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.AnnotateTIROpMappingType()(mod)
    return mod


def _split_module():
    """add -> split -> (TupleGetItem -> reshape) x2 -> add.

    The leading add is kOneToOne and the smallest node in the graph, so
    ``FindMinOtO`` seeds on it and the forward walk immediately meets split's
    two TupleGetItems.
    """
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([1, 8], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.add(x, relax.const(1.0, "float32")))
            lv1 = bb.emit(relax.op.split(lv0, 2, axis=1))
            lv2 = bb.emit(relax.TupleGetItem(lv1, 0))
            lv3 = bb.emit(relax.TupleGetItem(lv1, 1))
            lv4 = bb.emit(relax.op.reshape(lv2, [2, 2]))
            lv5 = bb.emit(relax.op.reshape(lv3, [2, 2]))
            gv = bb.emit_output(relax.op.multiply(lv4, lv5))
        bb.emit_func_output(gv)
    return bb.get()


def _fused_group_callees(mod):
    """{fused function name: [callee names]} over the module's fused groups."""
    groups = {}
    for gv, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        if func.attrs is None or not func.attrs.get("Primitive", False):
            continue
        callees = []

        def visit(node):
            if isinstance(node, relax.Call) and isinstance(node.args[0], tvm.ir.GlobalVar):
                callees.append(node.args[0].name_hint)

        relax.analysis.post_order_visit(func, visit)
        groups[gv.name_hint] = callees
    return groups


def test_fuses_through_tuple_get_item():
    """split's consumers join split's group instead of starting a new one."""
    mod = relax.transform.DNNFuseOps()(_pipeline(_split_module()))

    assert well_formed(mod)
    groups = _fused_group_callees(mod)
    assert len(groups) == 1, f"expected a single fused group, got {list(groups)}"
    callees = next(iter(groups.values()))
    assert sum(c.startswith("split") for c in callees) == 1, callees
    # The payoff: both reshapes crossed the TupleGetItem boundary into the group.
    assert sum(c.startswith("reshape") for c in callees) == 2, callees


def _build_pipeline():
    """tvm's default build pipeline minus DispatchSampling / DispatchSortScan.

    Both have a broken ``__init__`` in this build (the apache-tvm-ffi wheel
    shadows the bundled 3rdparty/tvm-ffi and uses ``__slots__`` without
    ``_inst``), and neither is needed here: this module has no sampling or
    sort-scan op. Same workaround the demo drivers use.
    """

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod, _ctx):
        return tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.ComputePrimValue(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
            ]
        )(mod)

    return _pipeline


def test_fused_module_is_numerically_equivalent():
    """Fusing through the TupleGetItem does not change what main computes."""
    before = _pipeline(_split_module())
    after = relax.transform.FuseTIR()(relax.transform.DNNFuseOps()(before))

    target, dev = tvm.target.Target("llvm"), tvm.cpu()
    x_np = np.random.RandomState(0).randn(1, 8).astype("float32")
    outs = []
    for mod in (before, after):
        ex = relax.build(mod, target=target, relax_pipeline=_build_pipeline())
        vm = relax.VirtualMachine(ex, dev)
        outs.append(vm["main"](tvm.runtime.tensor(x_np, dev)).numpy())
    tvm.testing.assert_allclose(outs[1], outs[0], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
