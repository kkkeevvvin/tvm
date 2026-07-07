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

"""issue#8: zero-divergence test for the name-keyed MappingType lookup table.

Issue #6 measured that deriving DNNFusion Table 2 (\\S3.1) MappingType from TVM's
OpPatternKind is lossy -- most visibly, broadcast elementwise ops (add/subtract/
multiply/divide) are indistinguishable from their same-shape counterparts, since
AnnotateTIROpPattern assigns kElemWise to both. Issue #8 replaces that proxy with
``AnnotateTIROpMappingType``, which classifies each call_tir callee directly from
its operator name (Table 2), with a shape check at the call site to disambiguate
the broadcast-capable elementwise ops.

This test covers exactly the op instances from issue #6 t2's agreement table (the
15 ops -- 16 rows, since add is split into same-shape/broadcast instances -- surveyed
across the all_6 models) and asserts the new pass agrees with Table 2 on every one of
them, i.e. zero divergence (unlike the five "△" rows the OpPatternKind proxy left).
"""

import enum

import pytest

import tvm
import tvm.testing
from tvm import relax, tir


# MappingType values assigned by AnnotateTIROpMappingType
# (mirrors include/tvm/relax/op_attr_types.h).
class MappingType(enum.IntEnum):
    kOneToOne = 0
    kOneToMany = 1
    kManyToMany = 2
    kReorganize = 3
    kShuffle = 4
    kMappingOpaque = 8


def _tensor(shape, dtype="float32"):
    return relax.TensorStructInfo(shape, dtype)


def _prim_func_mapping_types(mod):
    """(gvar name, mapping_type) of every TIR PrimFunc in an annotated module."""
    out = []
    for gv, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            attr = func.attrs.get("mapping_type") if func.attrs is not None else None
            if attr is not None:
                out.append((gv.name_hint, int(attr)))
    return out


def derive_mapping_type(sinfos, build) -> MappingType:
    """Lower a single real relax op to TIR and read back its mapping_type.

    ``build`` receives the input relax Vars and returns the op call expression.
    Every op in this test's coverage legalizes to exactly one call_tir callee
    (verified empirically in issue #6 t1 against the all_6 models), so unlike the
    OpPatternKind proxy test there is no multi-PrimFunc fuse-and-retry fallback --
    that would erase the callee name the classifier depends on.
    """
    bb = relax.BlockBuilder()
    args = [relax.Var(f"x{i}", s) for i, s in enumerate(sinfos)]
    with bb.function("main", args):
        with bb.dataflow():
            out = bb.emit(build(*args))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)
    mod = bb.get()

    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpMappingType()(mod)
    mapping_types = _prim_func_mapping_types(mod)
    assert len(mapping_types) == 1, (
        f"expected exactly one call_tir callee, got {mapping_types}"
    )
    return MappingType(mapping_types[0][1])


_F32 = "float32"

# (name, expected Table 2 type, input StructInfos, build_fn) for the 15 ops (16 rows)
# in issue #6 t1 coverage. build_fn returns the op call.
TABLE2_OPS = [
    (
        "adaptive_avg_pool2d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.adaptive_avg_pool2d(x, output_size=(1, 1)),
    ),
    (
        "add (same-shape)",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 8, 8])],
        relax.op.add,
    ),
    (
        "add (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.add,
    ),
    (
        "concatenate",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 8, 8])],
        lambda x, y: relax.op.concat([x, y], axis=1),
    ),
    (
        "conv2d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8]), _tensor([4, 4, 3, 3])],
        relax.op.nn.conv2d,
    ),
    (
        "divide (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.divide,
    ),
    (
        "matmul",
        MappingType.kManyToMany,
        [_tensor([10, 20]), _tensor([20, 30])],
        relax.op.matmul,
    ),
    (
        "max_pool2d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.max_pool2d(x, pool_size=(2, 2)),
    ),
    (
        "mean",
        MappingType.kManyToMany,
        [_tensor([10, 20])],
        lambda x: relax.op.mean(x, axis=1),
    ),
    (
        "multiply (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.multiply,
    ),
    (
        "relu",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.nn.relu,
    ),
    (
        "reshape",
        MappingType.kReorganize,
        [_tensor([10, 20])],
        lambda x: relax.op.reshape(x, [20, 10]),
    ),
    (
        "silu",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8])],
        relax.op.nn.silu,
    ),
    (
        "subtract (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.subtract,
    ),
    (
        "tir_clip",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        lambda x: relax.op.clip(x, relax.PrimValue(0.0), relax.PrimValue(6.0)),
    ),
    (
        "tir_sigmoid",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.sigmoid,
    ),
]


@pytest.mark.parametrize(
    "name,expected,sinfos,build",
    TABLE2_OPS,
    ids=[op[0] for op in TABLE2_OPS],
)
def test_mapping_type_agrees_with_table2(name, expected, sinfos, build):
    """The name-keyed lookup agrees with Table 2 on every op in issue #6 t1 coverage."""
    derived = derive_mapping_type(sinfos, build)
    assert derived == expected, (
        f"{name}: mapping_type derived {derived.name}, but Table 2 says {expected.name}"
    )


if __name__ == "__main__":
    tvm.testing.main()
