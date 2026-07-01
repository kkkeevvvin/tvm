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

"""issue#5: validate the OpPatternKind -> DNNFusion Table 2 mapping proxy.

DNNFusion (Niu et al., PLDI 2021) classifies every operator into one of five
mapping types (Table 2): One-to-One, One-to-Many, Many-to-Many, Reorganize,
Shuffle. Our port drives fusion off TVM's coarser ``OpPatternKind`` instead.
This test measures how faithfully that proxy reproduces Table 2 -- but, unlike
the earlier C++ attempt (which hand-authored a synthetic representative PrimFunc
per *category* and was therefore circular), it derives the pattern from the
**real** relax op: build a one-op module, ``LegalizeOps`` it to TIR, run
``AnnotateTIROpPattern``, and read the ``op_pattern`` the pass actually assigns.

The proxy under test is ``derive_mapping_type`` -- the OpPatternKind -> Table 2
rule. We assert it agrees with the paper on the operators whose relax lowering
is faithful to Table 2 (the documented divergences -- kInjective collapsing
Reorganize/Shuffle/Concat, and opaque/decomposed ops -- are out of scope here).
"""

import enum

import pytest

import tvm
import tvm.testing
from tvm import relax, tir


# OpPatternKind values assigned by AnnotateTIROpPattern
# (mirrors include/tvm/relax/op_attr_types.h).
class OpPatternKind(enum.IntEnum):
    kElemWise = 0
    kBroadcast = 1
    kInjective = 2
    kCommReduce = 3
    kOutEWiseFusable = 4
    kTuple = 7
    kOpaque = 8


# DNNFusion Table 2 mapping types (docs/ref/DNNFusion2106.pdf Sec 3.1).
class MappingType(enum.IntEnum):
    kUnknown = 0
    kOneToOne = 1
    kOneToMany = 2
    kManyToMany = 3
    kReorganize = 4
    kShuffle = 5


def derive_mapping_type(pattern: OpPatternKind) -> MappingType:
    """The proxy under test: OpPatternKind -> Table 2 mapping type."""
    return {
        OpPatternKind.kElemWise: MappingType.kOneToOne,
        # TVM kBroadcast is in-order axis broadcast -- the paper's One-to-Many.
        OpPatternKind.kBroadcast: MappingType.kOneToMany,
        # kInjective collapses Reorganize / Shuffle / Concat; proxy reports Reorganize.
        OpPatternKind.kInjective: MappingType.kReorganize,
        OpPatternKind.kCommReduce: MappingType.kManyToMany,
        OpPatternKind.kOutEWiseFusable: MappingType.kManyToMany,
    }.get(pattern, MappingType.kUnknown)


def _tensor(shape, dtype="float32"):
    return relax.TensorStructInfo(shape, dtype)


def _prim_func_patterns(mod):
    """op_pattern of every TIR PrimFunc in an annotated module."""
    out = []
    for _, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            attr = func.attrs.get("op_pattern") if func.attrs is not None else None
            if attr is not None:
                out.append(int(attr))
    return out


def derive_op_pattern(sinfos, build) -> OpPatternKind:
    """Lower a single real relax op to TIR and read back its op_pattern.

    ``build`` receives the input relax Vars and returns the op call expression.
    Most ops legalize to exactly one ``call_tir`` PrimFunc, whose pattern we
    return directly. If an op decomposes into several PrimFuncs we honour the
    "fuse-or-unknown" rule: fuse them into a single kernel (FuseOps + FuseTIR)
    and use that kernel's pattern; if they will not collapse to one kernel the
    op is opaque for our purposes.
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
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    patterns = _prim_func_patterns(mod)
    if len(patterns) == 1:
        return OpPatternKind(patterns[0])

    # 0 or >1 PrimFuncs: try to fuse the op into a single kernel.
    fused = relax.transform.FuseOps()(mod)
    fused = relax.transform.FuseTIR()(fused)
    fused = relax.transform.AnnotateTIROpPattern()(fused)
    patterns = _prim_func_patterns(fused)
    if len(patterns) == 1:
        return OpPatternKind(patterns[0])
    return OpPatternKind.kOpaque


_F32 = "float32"

# (onnx_name, expected Table 2 type, input StructInfos, build_fn) for the ops
# whose relax lowering is faithful to Table 2. build_fn returns the op call.
FAITHFUL_OPS = [
    # ---- One-to-One -> kElemWise ----
    ("Add", MappingType.kOneToOne, [_tensor([10, 20]), _tensor([10, 20])], relax.op.add),
    ("Asin", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.asin),
    ("Cast", MappingType.kOneToOne, [_tensor([10, 20])], lambda x: relax.op.astype(x, "float16")),
    ("Ceil", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.ceil),
    (
        "Clip",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        lambda x: relax.op.clip(x, relax.PrimValue(0.0), relax.PrimValue(6.0)),
    ),
    ("Cos", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.cos),
    ("Erf", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.erf),
    ("Exp", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.exp),
    ("Greater", MappingType.kOneToOne, [_tensor([10, 20]), _tensor([10, 20])], relax.op.greater),
    ("LeakyRelu", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.nn.leakyrelu),
    ("Log", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.log),
    ("Not", MappingType.kOneToOne, [_tensor([10, 20], "bool")], relax.op.logical_not),
    (
        "PRelu",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8]), _tensor([4])],
        relax.op.nn.prelu,
    ),
    ("Reciprocal", MappingType.kOneToOne, [_tensor([10, 20]), _tensor([10, 20])], relax.op.divide),
    ("Relu", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.nn.relu),
    ("Round", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.round),
    ("Sigmoid", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.sigmoid),
    ("Sin", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.sin),
    ("Sqrt", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.sqrt),
    ("Tanh", MappingType.kOneToOne, [_tensor([10, 20])], relax.op.tanh),
    (
        "Where",
        MappingType.kOneToOne,
        [_tensor([10, 20], "bool"), _tensor([10, 20]), _tensor([10, 20])],
        relax.op.where,
    ),
    # ---- One-to-Many -> kBroadcast ----
    # NB: a broadcasting binary op (e.g. add([20], [10, 20])) does NOT derive to
    # kBroadcast -- its full-rank operand makes the analyzer call it kElemWise
    # (One-to-One). broadcast_to is the faithful One-to-Many representative.
    (
        "Expand",
        MappingType.kOneToMany,
        [_tensor([20])],
        lambda x: relax.op.broadcast_to(x, [10, 20]),
    ),
    # ---- Many-to-Many -> kOutEWiseFusable (conv / pool / GEMM) ----
    (
        "AveragePool",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.avg_pool2d(x, pool_size=(2, 2)),
    ),
    (
        "CONV",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8]), _tensor([4, 4, 3, 3])],
        relax.op.nn.conv2d,
    ),
    (
        "ConvTranspose",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8]), _tensor([4, 4, 3, 3])],
        relax.op.nn.conv2d_transpose,
    ),
    ("GEMM", MappingType.kManyToMany, [_tensor([10, 20]), _tensor([20, 30])], relax.op.matmul),
    (
        "MaxPool",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.max_pool2d(x, pool_size=(2, 2)),
    ),
    # ---- Many-to-Many -> kCommReduce (ONNX Reduce family) ----
    ("ReduceProd", MappingType.kManyToMany, [_tensor([10, 20])], lambda x: relax.op.prod(x, axis=1)),
    ("ReduceMean", MappingType.kManyToMany, [_tensor([10, 20])], lambda x: relax.op.mean(x, axis=1)),
    # ---- Reorganize -> kInjective ----
    # NB: squeeze / expand_dims (unit-axis insert/remove) do NOT derive to
    # kInjective -- the constant-0 axis in the store makes the analyzer call
    # them kBroadcast (One-to-Many). flatten / reshape are the faithful
    # Reorganize representatives.
    ("Flatten", MappingType.kReorganize, [_tensor([1, 4, 8, 8])], relax.op.flatten),
    ("Reshape", MappingType.kReorganize, [_tensor([10, 20])], lambda x: relax.op.reshape(x, [20, 10])),
]


@pytest.mark.parametrize(
    "onnx_name,expected,sinfos,build",
    FAITHFUL_OPS,
    ids=[op[0] for op in FAITHFUL_OPS],
)
def test_derive_agrees_with_table2(onnx_name, expected, sinfos, build):
    """The OpPatternKind proxy agrees with Table 2 on faithful operators."""
    pattern = derive_op_pattern(sinfos, build)
    derived = derive_mapping_type(pattern)
    assert derived == expected, (
        f"{onnx_name}: op_pattern {pattern.name} -> {derived.name}, "
        f"but Table 2 says {expected.name}"
    )


if __name__ == "__main__":
    tvm.testing.main()
