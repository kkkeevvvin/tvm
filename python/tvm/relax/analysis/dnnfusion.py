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
"""DNNFusion (Niu et al., PLDI '21) operator classification.

Mirrors include/tvm/relax/dnnfusion_op_class.h for Python callers. The
canonical taxonomy and per-op assignments live in
docs/wiki/concepts/dnnfusion_taxonomy.md.
"""
from enum import IntEnum
from typing import Tuple

from tvm import tir
from tvm.relax.expr import Call

from . import _ffi_api


class DnnfOpClass(IntEnum):
    """DNNFusion mapping-type class (paper Table 2)."""

    ONE_TO_ONE = 0
    ONE_TO_MANY = 1
    MANY_TO_MANY = 2
    REORGANIZE = 3
    SHUFFLE = 4
    UNKNOWN = 5


class DnnfClassSource(IntEnum):
    """Provenance of a classification result."""

    ATTR_TABLE = 0
    TIR_ANNOTATION = 1
    PATTERN_HEURISTIC = 2
    FALLBACK = 3


def classify_dnnf_op(call: Call) -> Tuple[DnnfOpClass, DnnfClassSource, str]:
    """Classify a Relax Call into a DnnfOpClass.

    See ``include/tvm/relax/analysis.h`` ``ClassifyDnnfOp`` for lookup
    priority. Returns ``(class, source, reason)``.
    """
    cls_int, src_int, reason = _ffi_api.ClassifyDnnfOp(call)  # type: ignore
    return DnnfOpClass(int(cls_int)), DnnfClassSource(int(src_int)), str(reason)


def infer_dnnf_class_from_tir(func: tir.PrimFunc) -> DnnfOpClass:
    """Run the OpPatternKind→DnnfOpClass heuristic on a TIR PrimFunc.

    Reads the function's ``op_pattern`` integer attribute (populated by
    ``AnnotateTIROpPattern``); returns ``UNKNOWN`` if the attribute is
    missing.
    """
    return DnnfOpClass(int(_ffi_api.InferDnnfClassFromTir(func)))  # type: ignore


def dnnf_class_from_op_pattern(pattern: int) -> DnnfOpClass:
    """Map an OpPatternKind integer value to the heuristic DnnfOpClass."""
    return DnnfOpClass(int(_ffi_api.DnnfClassFromOpPattern(int(pattern))))  # type: ignore
