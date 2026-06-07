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

#include "./graph_partitioner.h"

namespace tvm {
namespace relax {

/*!
 * \brief Resolve the PrimFunc backing `node` through its cached call_tir GlobalVar.
 * \param mod The IRModule the node's call_tir binding lives in.
 * \param node The IndexedForwardGraph node to resolve.
 * \return The node's PrimFunc, or nullopt for nodes with no call_tir binding
 *  (gvar == nullptr) or whose gvar resolves to a non-PrimFunc (e.g. a relax
 *  Function) -- either way there is no single PrimFunc to time.
 */
ffi::Optional<tir::PrimFunc> FindPrimFunc(const IRModule& mod,
                                          const IndexedForwardGraph::Node* node);

/*!
 * \brief Schedule, build, and time `func` on the current target's device.
 * \param func The PrimFunc to schedule, build, and run.
 * \return Average per-run latency in microseconds, or -1.0 on failure.
 * \details Dispatches on the current target: a CUDA target routes to
 *  TimePrimFuncCUDA; other targets are not yet supported and return -1.0
 *  (treated as a timing failure by FuseProfit). The target comes from the
 *  enclosing `with target:` scope (the driver wraps FuseOps in it so
 *  Target::Current() is set).
 */
double TimePrimFunc(const tir::PrimFunc& func);

/*!
 * \brief Build `func` (BuildPrimFuncCUDA) on the current target's device, then
 *  time it.
 * \param func The PrimFunc to schedule, build, and run.
 * \return Latency in microseconds, or -1.0 if the build fails. Currently always
 *  -1.0: only the build step is wired up so far (and BuildPrimFuncCUDA is itself
 *  a stub); device-argument materialization and timing are not implemented yet.
 * \details The build target (and hence the profiled device) is taken from the
 *  enclosing `with target:` scope (Target::Current), so the driver must wrap its
 *  FuseOps call in `with TARGET:`; ICHECK-fails if no target is in scope.
 */
double TimePrimFuncCUDA(const tir::PrimFunc& func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNF_PROFILER_H_
