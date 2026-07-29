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

import enum

import numpy as np
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

_F16 = "float16"

# (name, expected Table 2 type, input StructInfos, build_fn) for every op covered by
# the lookup table: issue #6 t1 (all_6 models), the ops issue #37 found uncovered on
# the paper_6 models, and the ops issue #38 found uncovered on the paleo models.
# build_fn returns the op call.
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
    # Broadcast against a weight (relax.Constant, e.g. a conv bias): the constant
    # operand is ignored in the shape comparison, so the op stays One-to-One along
    # its activation edge.
    (
        "add (broadcast weight)",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.add(x, relax.const(np.zeros((1, 4, 1, 1), _F32))),
    ),
    # BSNH layout: (batch, seq_len, num_heads, head_dim). One PrimFunc computing
    # softmax(Q*K^T/sqrt(d))*V.
    (
        "attention",
        MappingType.kManyToMany,
        [_tensor([1, 4, 2, 8], _F16)] * 3,
        relax.op.nn.attention,
    ),
    (
        "attention_bias",
        MappingType.kManyToMany,
        [_tensor([1, 4, 2, 8], _F16)] * 3 + [_tensor([1, 2, 4, 4], _F16)],
        relax.op.nn.attention,
    ),
    (
        "avg_pool2d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.avg_pool2d(x, pool_size=(2, 2)),
    ),
    (
        "avg_pool3d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8, 8])],
        lambda x: relax.op.nn.avg_pool3d(x, pool_size=(2, 2, 2)),
    ),
    (
        "cast",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        lambda x: relax.op.astype(x, _F16),
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
    # unet's up-sampling path. Default kernel_layout for conv2d_transpose is IOHW.
    (
        "conv2d_transpose",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8]), _tensor([4, 4, 3, 3])],
        relax.op.nn.conv2d_transpose,
    ),
    (
        "conv3d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8, 8]), _tensor([4, 4, 3, 3, 3])],
        relax.op.nn.conv3d,
    ),
    (
        "divide (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.divide,
    ),
    # Not in Table 2 (Gelu is ONNX opset 20, post-paper); a pure scalar function of
    # the element at the same index, so it lands in the Relu/Sigmoid/Tanh row.
    (
        "gelu",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.nn.gelu,
    ),
    # Not in Table 2 (LayerNormalization is ONNX opset 17, post-paper); classified
    # as Many-to-Many by analogy with InstanceNormalization / Reduce / Softmax --
    # the row mean/variance make every output element read the whole normalized row.
    (
        "layer_norm",
        MappingType.kManyToMany,
        [_tensor([1, 197, 1024]), _tensor([1024]), _tensor([1024])],
        lambda x, gamma, beta: relax.op.nn.layer_norm(x, gamma, beta, axes=[-1]),
    ),
    # relax.nn.leakyrelu legalizes to topi.nn.leaky_relu -> PrimFunc "leaky_relu".
    (
        "leaky_relu",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.nn.leakyrelu,
    ),
    (
        "matmul",
        MappingType.kManyToMany,
        [_tensor([10, 20]), _tensor([20, 30])],
        relax.op.matmul,
    ),
    # ONNX Max/Pow: broadcast-capable binaries, resolved by operand shape like add.
    (
        "maximum (same-shape)",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 8, 8])],
        relax.op.maximum,
    ),
    (
        "maximum (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.maximum,
    ),
    (
        "max_pool2d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.max_pool2d(x, pool_size=(2, 2)),
    ),
    (
        "max_pool3d",
        MappingType.kManyToMany,
        [_tensor([1, 4, 8, 8, 8])],
        lambda x: relax.op.nn.max_pool3d(x, pool_size=(2, 2, 2)),
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
        "multiply (broadcast weight)",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.multiply(x, relax.const(np.ones((1, 4, 1, 1), _F32))),
    ),
    # Constant-mode pad; the other pad_modes legalize to reflect_pad/replicate_pad/
    # circular_pad, which the surveys never hit and the table does not cover.
    (
        "pad",
        MappingType.kReorganize,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.nn.pad(x, pad_width=(0, 0, 0, 0, 1, 1, 1, 1)),
    ),
    (
        "power (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.power,
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
    # yolov4's upsample. One input element feeds several output elements.
    (
        "resize2d",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8])],
        lambda x: relax.op.image.resize2d(x, size=(16, 16)),
    ),
    (
        "silu",
        MappingType.kOneToOne,
        [_tensor([1, 4, 8, 8])],
        relax.op.nn.silu,
    ),
    (
        "softmax",
        MappingType.kManyToMany,
        [_tensor([10, 20])],
        lambda x: relax.op.nn.softmax(x, axis=-1),
    ),
    (
        "softplus",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.nn.softplus,
    ),
    (
        "split",
        MappingType.kReorganize,
        [_tensor([10, 20])],
        lambda x: relax.op.split(x, 2, axis=1),
    ),
    (
        "squeeze",
        MappingType.kReorganize,
        [_tensor([1, 10, 20])],
        lambda x: relax.op.squeeze(x, axis=0),
    ),
    (
        "stack",
        MappingType.kOneToOne,
        [_tensor([10, 20]), _tensor([10, 20])],
        lambda x, y: relax.op.stack([x, y], axis=0),
    ),
    (
        "strided_slice",
        MappingType.kReorganize,
        [_tensor([10, 20])],
        lambda x: relax.op.strided_slice(x, axes=[1], begin=[0], end=[10]),
    ),
    (
        "subtract (broadcast)",
        MappingType.kOneToMany,
        [_tensor([1, 4, 8, 8]), _tensor([1, 4, 1, 1])],
        relax.op.subtract,
    ),
    (
        "sum",
        MappingType.kManyToMany,
        [_tensor([10, 20])],
        lambda x: relax.op.sum(x, axis=1),
    ),
    (
        "tir_abs",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.abs,
    ),
    (
        "tir_clip",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        lambda x: relax.op.clip(x, relax.PrimValue(0.0), relax.PrimValue(6.0)),
    ),
    (
        "tir_negative",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.negative,
    ),
    (
        "tir_sigmoid",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.sigmoid,
    ),
    (
        "tir_sqrt",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.sqrt,
    ),
    (
        "tir_tanh",
        MappingType.kOneToOne,
        [_tensor([10, 20])],
        relax.op.tanh,
    ),
    # relax.permute_dims legalizes to topi.transpose -> PrimFunc "transpose".
    (
        "transpose",
        MappingType.kShuffle,
        [_tensor([10, 20])],
        lambda x: relax.op.permute_dims(x, axes=[1, 0]),
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


def test_weight_broadcast_via_reshape_is_one_to_one():
    """alexnet's add[lv2] (conv bias add): from_fx routes the bias Constant through a
    reshape binding, so the add's operand is a weight-derived var rather than a
    Constant. The classifier must trace that back to the weight and keep the add
    One-to-One (the pass runs before FoldConstant collapses the reshape)."""
    bb = relax.BlockBuilder()
    x = relax.Var("x", _tensor([1, 4, 8, 8]))
    bias = relax.const(np.zeros((4,), _F32))
    with bb.function("main", [x]):
        with bb.dataflow():
            reshaped = bb.emit(relax.op.reshape(bias, (1, 4, 1, 1)))
            out = bb.emit(relax.op.add(x, reshaped))
            gv = bb.emit_output(out)
        bb.emit_func_output(gv)

    mod = relax.transform.LegalizeOps()(bb.get())
    mod = relax.transform.AnnotateTIROpMappingType()(mod)
    mod = relax.transform.FoldConstant()(mod)
    types = dict(_prim_func_mapping_types(mod))
    assert types["add"] == MappingType.kOneToOne, (
        f"bias add classified {MappingType(types['add']).name}, expected kOneToOne"
    )
    assert types["reshape"] == MappingType.kReorganize

if __name__ == "__main__":
    tvm.testing.main()
