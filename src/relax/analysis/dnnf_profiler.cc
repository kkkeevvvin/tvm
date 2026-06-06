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
 * \file src/relax/analysis/dnnf_profiler.cc
 * \brief In-pass cost oracle for the DNNFusion port: JIT-build and time relax
 *  PrimFuncs on the GPU. Self-contained device-timing utilities with no
 *  dependency on GraphPartitioner state; see dnnf_profiler.h.
 */

#include "./dnnf_profiler.h"

namespace tvm {
namespace relax {

// STUB: resolve the PrimFunc backing `node` through node->gvar in `mod`. Not
// implemented yet -- always returns nullopt so TimeNode reports "no PrimFunc".
ffi::Optional<tir::PrimFunc> FindPrimFunc(const IRModule&, const IndexedForwardGraph::Node*) {
  LOG(INFO) << "  FindPrimFunc stub: PrimFunc lookup disabled";
  return std::nullopt;
}

// STUB: schedule, build, and time `func` on the GPU. Not implemented yet --
// always returns -1.0 (treated as a timing failure by FuseProfit).
double TimePrimFunc(const tir::PrimFunc&) {
  LOG(INFO) << "  TimePrimFunc stub: profiling disabled";
  return -1.0;
}

}  // namespace relax
}  // namespace tvm
