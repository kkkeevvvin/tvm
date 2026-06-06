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

// Resolve the PrimFunc backing `node` through its cached call_tir GlobalVar.
// Returns nullopt for nodes with no call_tir binding (gvar == nullptr) or whose
// gvar resolves to a non-PrimFunc (e.g. a relax Function) -- either way there is
// no single PrimFunc to time.
ffi::Optional<tir::PrimFunc> FindPrimFunc(const IRModule& mod,
                                          const IndexedForwardGraph::Node* node) {
  if (node->gvar == nullptr) return std::nullopt;
  GlobalVar gvar = ffi::GetRef<GlobalVar>(node->gvar);
  // The gvar is cached during IFG construction from a call_tir in `mod`, so it
  // is expected to resolve; guard the lookup anyway to stay arena-safe.
  if (!mod->ContainGlobalVar(gvar->name_hint)) return std::nullopt;
  return mod->Lookup(gvar).as<tir::PrimFunc>();
}

// STUB: schedule, build, and time `func` on the GPU. Not implemented yet --
// always returns -1.0 (treated as a timing failure by FuseProfit).
double TimePrimFunc(const tir::PrimFunc& func) {
  ICHECK(func.defined()) << "TimePrimFunc called with an undefined PrimFunc";
  LOG(INFO) << "  TimePrimFunc:\n" << func;
  LOG(INFO) << "  TimePrimFunc stub: profiling disabled";
  return -1.0;
}

}  // namespace relax
}  // namespace tvm
