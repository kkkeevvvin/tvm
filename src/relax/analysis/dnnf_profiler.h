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
 * \brief Resolve the PrimFunc backing `node` through node->gvar in `mod`.
 * \param mod The IRModule the node's call_tir binding lives in.
 * \param node The IndexedForwardGraph node to resolve.
 * \return The node's PrimFunc, or nullopt if it has no cached call_tir GlobalVar
 *  (or that GlobalVar resolves to a non-PrimFunc).
 */
ffi::Optional<tir::PrimFunc> FindPrimFunc(const IRModule& mod,
                                          const IndexedForwardGraph::Node* node);

/*!
 * \brief Schedule, build, and time `func` on the GPU.
 * \param func The PrimFunc to schedule, build, and run.
 * \return Average per-run latency in microseconds, or -1.0 on failure.
 *
 *  STUB: device timing not implemented yet; always returns -1.0.
 */
double TimePrimFunc(const tir::PrimFunc& func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_DNNF_PROFILER_H_
