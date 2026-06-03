/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/analysis/dnnf_profiler.h
 * \brief In-pass cost oracle for the DNNFusion port: JIT-build and time relax
 *  PrimFuncs (single ops or FuseTIR-merged blocks) on the GPU. Used by
 *  GraphPartitioner's FuseProfit to profit-gate kFuseDepend fusion.
 */

#ifndef TVM_RELAX_ANALYSIS_DNNF_PROFILER_H_
#define TVM_RELAX_ANALYSIS_DNNF_PROFILER_H_

#include <tvm/ir/module.h>
#include <tvm/tir/function.h>

#include <vector>

#include "./graph_partitioner.h"

namespace tvm {
namespace relax {

// Sample counts for the in-pass cost oracle: TimePrimFuncCUDA averages over
// kProfileInputs distinct random input sets, each timed over kProfileRuns device
// runs (after one untimed warmup), repeated kProfileRepeats times. Exposed so
// callers can report them alongside results.
constexpr int kProfileInputs = 5;
constexpr int kProfileRuns = 20;
constexpr int kProfileRepeats = 3;

/*!
 * \brief Build `func` on the GTX 1070, feed it random device inputs, and time it.
 * \param func The PrimFunc to schedule, build, and run.
 * \return Average per-run latency in microseconds, or -1.0 if the build fails or
 *  any param cannot be materialized (non-buffer / dynamic shape).
 */
double TimePrimFuncCUDA(const tir::PrimFunc& func);

/*!
 * \brief Build a kPrimitive module fusing all of `nodes` (producer-before-
 *  consumer order), run FuseTIR, and return the single merged PrimFunc.
 *
 *  Internal edges are wired var-to-var; every other input -- including constant
 *  args -- becomes a tensor param, so the merged kernel is self-contained.
 * \param mod The IRModule the nodes' call_tir bindings live in.
 * \param nodes The nodes to fuse, sorted by IndexedForwardGraph index.
 * \return The merged PrimFunc, or nullopt if any node lacks a call_tir binding
 *  or FuseTIR cannot produce a single PrimFunc.
 */
ffi::Optional<tir::PrimFunc> BuildFusedBlock(
    const IRModule& mod, const std::vector<const IndexedForwardGraph::Node*>& nodes);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNF_PROFILER_H_
