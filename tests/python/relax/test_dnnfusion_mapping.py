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
"""Unit tests for the DNNFusion operator classifier (Phase A).

Covers:
  * Table 2 explicit op-name lookups for every MappingType bucket.
  * OpPatternKind fallback when op name is unrecognized / empty.
  * Table 3 5x5 fusion legality matrix, verbatim against the paper.
"""
import pytest

from tvm.relax.analysis.dnnfusion import (
    FuseColor,
    MappingType,
    derive_mapping_type,
    fuse_mapping_check,
)

# OpPatternKind values, mirroring include/tvm/relax/op_attr_types.h.
K_ELEMWISE = 0
K_BROADCAST = 1
K_INJECTIVE = 2
K_COMM_REDUCE = 3
K_OUT_EWISE_FUSABLE = 4
K_TUPLE = 7
K_OPAQUE = 8


# ---------------------------------------------------------------------------
# Table 2: explicit op-name -> MappingType.  One representative per bucket
# from each paper category, plus a few that triggered TVM-vs-paper friction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name,expected",
    [
        # OneToOne (paper Table 2: Add, Asin, BatchNorm, Cast, Ceil, Clip,
        # Concat, Cos, Erf, Exp, ..., Where).
        ("relax.add", MappingType.ONE_TO_ONE),
        ("relax.subtract", MappingType.ONE_TO_ONE),
        ("relax.exp", MappingType.ONE_TO_ONE),
        ("relax.sigmoid", MappingType.ONE_TO_ONE),
        ("relax.tanh", MappingType.ONE_TO_ONE),
        ("relax.erf", MappingType.ONE_TO_ONE),
        ("relax.clip", MappingType.ONE_TO_ONE),
        ("relax.where", MappingType.ONE_TO_ONE),
        ("relax.astype", MappingType.ONE_TO_ONE),  # ONNX Cast
        ("relax.concat", MappingType.ONE_TO_ONE),  # paper says OtO; TVM has it as Injective
        ("relax.split", MappingType.ONE_TO_ONE),
        ("relax.strided_slice", MappingType.ONE_TO_ONE),  # ONNX Slice
        ("relax.nn.relu", MappingType.ONE_TO_ONE),
        ("relax.nn.leakyrelu", MappingType.ONE_TO_ONE),
        ("relax.nn.gelu", MappingType.ONE_TO_ONE),
        ("relax.nn.batch_norm", MappingType.ONE_TO_ONE),
        ("relax.greater", MappingType.ONE_TO_ONE),
        # OneToMany (Expand, Gather, Resize, Upsample).
        ("relax.broadcast_to", MappingType.ONE_TO_MANY),
        ("relax.take", MappingType.ONE_TO_MANY),
        ("relax.tile", MappingType.ONE_TO_MANY),
        ("relax.image.resize2d", MappingType.ONE_TO_MANY),
        # ManyToMany (Conv, GEMM, Reduce*, Softmax, Pool).
        ("relax.matmul", MappingType.MANY_TO_MANY),
        ("relax.einsum", MappingType.MANY_TO_MANY),
        ("relax.nn.conv2d", MappingType.MANY_TO_MANY),
        ("relax.nn.conv2d_transpose", MappingType.MANY_TO_MANY),
        ("relax.nn.softmax", MappingType.MANY_TO_MANY),
        ("relax.nn.layer_norm", MappingType.MANY_TO_MANY),
        ("relax.sum", MappingType.MANY_TO_MANY),
        ("relax.mean", MappingType.MANY_TO_MANY),
        ("relax.cumsum", MappingType.MANY_TO_MANY),
        ("relax.nn.max_pool2d", MappingType.MANY_TO_MANY),
        # Reorganize (Reshape, Flatten, Squeeze, Unsqueeze).
        ("relax.reshape", MappingType.REORGANIZE),
        ("relax.flatten", MappingType.REORGANIZE),
        ("relax.squeeze", MappingType.REORGANIZE),
        ("relax.expand_dims", MappingType.REORGANIZE),  # ONNX Unsqueeze
        # Shuffle (Transpose, DepthToSpace, SpaceToDepth).
        ("relax.permute_dims", MappingType.SHUFFLE),
    ],
)
def test_table2_explicit(op_name, expected):
    assert derive_mapping_type(op_name, K_OPAQUE) == expected


# ---------------------------------------------------------------------------
# OpPatternKind fallback: unknown op name -> classify by pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern,expected",
    [
        (K_ELEMWISE, MappingType.ONE_TO_ONE),
        (K_BROADCAST, MappingType.ONE_TO_MANY),
        (K_INJECTIVE, MappingType.REORGANIZE),
        (K_COMM_REDUCE, MappingType.MANY_TO_MANY),
        (K_OUT_EWISE_FUSABLE, MappingType.MANY_TO_MANY),
        (K_TUPLE, MappingType.UNKNOWN),
        (K_OPAQUE, MappingType.UNKNOWN),
    ],
)
def test_pattern_fallback(pattern, expected):
    # Empty op_name forces fallback.
    assert derive_mapping_type("", pattern) == expected
    # Unknown op name also falls back.
    assert derive_mapping_type("relax.does_not_exist", pattern) == expected


def test_explicit_overrides_pattern():
    # Even if the caller hands us a pessimistic kOpaque pattern, an explicit
    # Table 2 entry must win.
    assert derive_mapping_type("relax.add", K_OPAQUE) == MappingType.ONE_TO_ONE
    # And conversely: a paper-OtO that TVM happens to tag kInjective
    # (e.g. relax.concat) should classify as OtO, not Reorganize.
    assert derive_mapping_type("relax.concat", K_INJECTIVE) == MappingType.ONE_TO_ONE


# ---------------------------------------------------------------------------
# Table 3: 5x5 fusion legality matrix (DNNFusion §3.2, verbatim).
# Columns and rows in the order { OtO, OtM, MtM, Reorg, Shuffle }.
# Cells encode (color, result_type).
# ---------------------------------------------------------------------------

_OTO = MappingType.ONE_TO_ONE
_OTM = MappingType.ONE_TO_MANY
_MTM = MappingType.MANY_TO_MANY
_REO = MappingType.REORGANIZE
_SHF = MappingType.SHUFFLE
_R, _Y, _G = FuseColor.RED, FuseColor.YELLOW, FuseColor.GREEN
_UNK = MappingType.UNKNOWN

# Ordered list; index i uses _ROWS[i] as 'first'.
_ROWS = [_OTO, _OTM, _MTM, _REO, _SHF]

# fmt: off
_TABLE3 = {
    # first  second  -> (color, result)
    # Paper §3.2 "OtO with others ... so they are profitable" + Add+GEMM
    # "in either order ... profitable" -> entire OtO row is Green.
    (_OTO, _OTO):  (_G, _OTO),
    (_OTO, _OTM):  (_G, _OTM),
    (_OTO, _MTM):  (_G, _MTM),
    (_OTO, _REO):  (_G, _REO),
    (_OTO, _SHF):  (_G, _SHF),

    (_OTM, _OTO):  (_G, _OTM),
    (_OTM, _OTM):  (_Y, _OTM),
    (_OTM, _MTM):  (_R, _UNK),
    (_OTM, _REO):  (_Y, _OTM),
    (_OTM, _SHF):  (_Y, _OTM),

    (_MTM, _OTO):  (_G, _MTM),
    (_MTM, _OTM):  (_Y, _MTM),
    (_MTM, _MTM):  (_R, _UNK),
    (_MTM, _REO):  (_Y, _MTM),
    (_MTM, _SHF):  (_Y, _MTM),

    (_REO, _OTO):  (_G, _REO),
    (_REO, _OTM):  (_Y, _OTM),
    (_REO, _MTM):  (_Y, _MTM),
    (_REO, _REO):  (_G, _REO),
    (_REO, _SHF):  (_G, _REO),

    (_SHF, _OTO):  (_G, _SHF),
    (_SHF, _OTM):  (_Y, _OTM),
    (_SHF, _MTM):  (_Y, _MTM),
    (_SHF, _REO):  (_G, _REO),
    (_SHF, _SHF):  (_G, _SHF),
}
# fmt: on


@pytest.mark.parametrize("first,second,expected", [(f, s, _TABLE3[(f, s)]) for f in _ROWS for s in _ROWS])
def test_table3_matches_paper(first, second, expected):
    color, result = fuse_mapping_check(first, second)
    assert (color, result) == expected, (
        f"Table 3 mismatch at ({first.name}, {second.name}): "
        f"got ({color.name}, {result.name}), want ({expected[0].name}, {expected[1].name})"
    )


@pytest.mark.parametrize("other", _ROWS + [_UNK])
def test_unknown_is_always_red(other):
    # Every cell that involves Unknown must be Red, regardless of side.
    assert fuse_mapping_check(_UNK, other)[0] == _R
    assert fuse_mapping_check(other, _UNK)[0] == _R


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
