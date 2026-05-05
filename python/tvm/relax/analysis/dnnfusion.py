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
"""DNNFusion (Niu et al., PLDI '21) operator classification helpers.

Thin Python wrappers over the C++ classifier in
``src/relax/analysis/dnnfusion_mapping.{h,cc}``.  The goal is to let driver
scripts and tests query MappingType / Table 3 fusion decisions without
crossing into C++.
"""
import enum
from typing import Tuple

from . import _ffi_api


class MappingType(enum.IntEnum):
    """Five-way DNNFusion §3.1 operator classification."""

    UNKNOWN = 0
    ONE_TO_ONE = 1
    ONE_TO_MANY = 2
    MANY_TO_MANY = 3
    REORGANIZE = 4
    SHUFFLE = 5


class FuseColor(enum.IntEnum):
    """DNNFusion Table 3 fusion-candidate verdict."""

    RED = 0      # illegal or known-unprofitable
    YELLOW = 1   # legal; profitability requires profile lookup
    GREEN = 2    # legal and profitable; fuse without further check


def derive_mapping_type(op_name: str, pattern: int = 8) -> MappingType:
    """Classify an operator by its registered name + OpPatternKind fallback.

    Parameters
    ----------
    op_name : str
        Registered op name, e.g. ``"relax.add"``, ``"relax.nn.conv2d"``.
        Pass an empty string to force the OpPatternKind fallback path.
    pattern : int, optional
        OpPatternKind value (kElemWise=0, kBroadcast=1, kInjective=2,
        kCommReduce=3, kOutEWiseFusable=4, kTuple=7, kOpaque=8).  Defaults
        to ``kOpaque`` (worst case, falls through to UNKNOWN).
    """
    return MappingType(_ffi_api.DnnfusionDeriveMappingType(op_name, int(pattern)))


def fuse_mapping_check(first: MappingType, second: MappingType) -> Tuple[FuseColor, MappingType]:
    """Look up DNNFusion Table 3 for a (first, second) fusion candidate.

    Returns a ``(color, result_type)`` tuple.  ``result_type`` is meaningful
    only when ``color`` is YELLOW or GREEN; under RED it is UNKNOWN.
    """
    color, result = _ffi_api.DnnfusionFuseMappingCheck(int(first), int(second))
    return FuseColor(int(color)), MappingType(int(result))
