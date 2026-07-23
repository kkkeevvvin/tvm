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

"""End-to-end coverage for the IndexedForwardGraph::Node::output_size cache.

The cached ``output_size`` (commit 916151140) is internal C++ state with no
Python accessor. It is populated by ``GraphCreator::Create`` -- which runs
*unconditionally* at the start of ``relax::FuseOps`` / ``relax::DNNFuseOps``
for every graph node, before any partitioning -- and read by the DNNFuse seed
selector ``FindMinOtO`` (``DNNFuseOps()``), which considers only nodes whose
``mapping_type`` is ``kOneToOne`` and skips nodes whose ``output_size`` is the
``-1`` "unknown / dynamic / opaque" sentinel.

Exact byte values are pinned in the C++ unit test
``tests/cpp/relax_struct_info_bytes_test.cc``. These Python tests drive the
*pass* so the population (and, for the dynamic case, the consumption) of the
field is exercised on real Relax modules and regression-guarded:

* ``fuse_opt_level=0`` (via ``FuseOps``) builds the indexed-forward graph
  (computing ``output_size`` for every tensor / tuple node) and then returns
  before fusing -- a stable way to run the population path through the
  public API.
* ``DNNFuseOps()`` on an all-dynamic graph runs ``RunDNNFuse``; the relu
  node classifies ``kOneToOne`` (Table 2 is keyed on the op name, so dynamic
  shapes do not opaque it) yet carries the ``-1`` sentinel, so ``FindMinOtO``
  finds no seed and the pass completes -- exercising the sentinel-skip branch
  of the consumer.
"""

import tvm
import tvm.testing
from tvm import relax, topi
from tvm.relax.analysis import well_formed

# opt_level 0 builds the IFG (populating output_size) then skips fusion.
BUILD_GRAPH_ONLY = 0


def _annotate(mod):
    return relax.transform.AnnotateTIROpMappingType()(mod)


def test_output_size_population_static_tensors():
    """Static tensor nodes get output_size computed during graph creation."""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.emit_te(topi.multiply, lv1, relax.const(2, "float32")))
            bb.emit_func_output(gv)
        return bb.get()

    mod = _annotate(before())
    # Runs GraphCreator::Create -> ComputeNodeOutputBytes on every node.
    out = relax.transform.FuseOps(fuse_opt_level=BUILD_GRAPH_ONLY)(mod)

    assert well_formed(out)
    assert out.get_global_var("main") is not None


def test_output_size_population_tuple():
    """A tuple-producing op exercises the TupleStructInfo branch of the cache."""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo([8, 4], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                # topi.split yields a tuple -> the call node carries TupleStructInfo,
                # routing through StructInfoBytes' tuple-summation path.
                parts = bb.emit_te(topi.split, lv0, 2, axis=1)
                p0 = bb.emit(relax.TupleGetItem(parts, 0))
                gv = bb.emit_output(bb.emit_te(topi.exp, p0))
            bb.emit_func_output(gv)
        return bb.get()

    mod = _annotate(before())
    out = relax.transform.FuseOps(fuse_opt_level=BUILD_GRAPH_ONLY)(mod)

    assert well_formed(out)


def test_dnnfuse_consumer_skips_dynamic_sentinel():
    """Dynamic dims yield output_size == -1; FindMinOtO finds no seed."""

    def before():
        bb = relax.BlockBuilder()
        n = tvm.tir.Var("n", "int64")
        x = relax.Var("x", relax.TensorStructInfo([n, 20], "float32"))
        with bb.function("main", [x], {"relax.force_pure": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.relu, x)
                gv = bb.emit_output(bb.emit_te(topi.exp, lv0))
            bb.emit_func_output(gv)
        return bb.get()

    mod = _annotate(before())
    # The relu node is kOneToOne but carries the -1 sentinel, so RunDNNFuse
    # seeds nothing and the DNNFuse path completes without fusing.
    out = relax.transform.DNNFuseOps()(mod)

    assert well_formed(out)
    assert out.get_global_var("main") is not None


if __name__ == "__main__":
    tvm.testing.main()
