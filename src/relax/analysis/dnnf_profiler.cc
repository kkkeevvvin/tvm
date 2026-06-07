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

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace relax {

namespace {

/*!
 * \brief Log the CUDA C source of the first "cuda" submodule (DFS over imports).
 * \param mod The runtime module to search for a "cuda" submodule.
 * \note A GPU build keeps device kernels in an imported submodule, not the root,
 *       so we recurse into imports. Best-effort: silent if none is found.
 */
void DumpCUDASource(const ffi::Module& mod) {
  if (mod->kind() == std::string("cuda")) {
    LOG(INFO) << "  BuildPrimFuncCUDA: generated CUDA source:\n" << mod->InspectSource("");
    return;
  }
  for (const Any& im : mod->imports()) {
    DumpCUDASource(im.cast<ffi::Module>());
  }
}

/*!
 * \brief Build `func` into a callable CUDA kernel, or nullopt if the build fails.
 * \param func The PrimFunc to schedule and build.
 * \param target The CUDA target to build for (also attached to the PrimFunc).
 * \return The built function looked up by name, or nullopt if the build throws.
 * \details Names the PrimFunc and wraps it in a single-function IRModule (with
 *          `target` attached as kTarget); runs DefaultGPUSchedule to give the
 *          otherwise unscheduled func its GPU thread bindings; tir.builds it for
 *          `target`; dumps the generated CUDA source; and returns the built
 *          function by name. Any tvm::Error is caught and turned into nullopt, so
 *          a failing build never aborts the caller.
 * \note We avoid opening a `With<Target>` scope here on purpose: if the build
 *       throws while such a scope is live, the scope's destructor (ExitWithScope)
 *       runs mid-unwind and its ICHECK would std::terminate the process. The
 *       current target instead comes from the caller's outer scope (the driver's
 *       `with TARGET:` around FuseOps).
 */
ffi::Optional<ffi::Function> BuildPrimFuncCUDA(const tir::PrimFunc& func, const Target& target) {
  try {
    const ffi::String func_name = "tir_function";
    tir::PrimFunc named = WithAttr(func, tvm::attr::kGlobalSymbol, func_name);
    named = WithAttr(named, tvm::attr::kTarget, target);
    GlobalVar gv(func_name);
    ffi::Map<GlobalVar, BaseFunc> funcs;
    funcs.Set(gv, named);
    IRModule m(funcs);
    // Apply thread bindings via the DefaultGPUSchedule heuristic.
    m = tir::transform::DefaultGPUSchedule()(m);
    tir::PrimFunc scheduled = Downcast<tir::PrimFunc>(m->Lookup(gv));
    LOG(INFO) << "  BuildPrimFuncCUDA: scheduled PrimFunc:\n" << scheduled;
    // "tir.build" is the Python `build` function in python/tvm/tir/build.py
    // (registered there via tvm.register_global_func("tir.build", build)). It
    // wraps the PrimFunc in an IRModule, binds the target, runs the TIR lowering
    // pipeline, splits host/device modules, and codegens each to a runtime
    // Module (CUDA device code imported into the host module).
    const auto build = tvm::ffi::Function::GetGlobalRequired("tir.build");
    ffi::Module rt_module = build(scheduled, target).cast<ffi::Module>();
    DumpCUDASource(rt_module);
    return rt_module->GetFunction(func_name).value();
  } catch (const tvm::Error& err) {
    LOG(INFO) << "  BuildPrimFuncCUDA: build failed: " << err.what();
    return std::nullopt;
  }
}

}  // namespace

ffi::Optional<tir::PrimFunc> FindPrimFunc(const IRModule& mod,
                                          const IndexedForwardGraph::Node* node) {
  if (node->gvar == nullptr) return std::nullopt;
  GlobalVar gvar = ffi::GetRef<GlobalVar>(node->gvar);
  // The gvar is cached during IFG construction from a call_tir in `mod`, so it
  // is expected to resolve; guard the lookup anyway to stay arena-safe.
  if (!mod->ContainGlobalVar(gvar->name_hint)) return std::nullopt;
  return mod->Lookup(gvar).as<tir::PrimFunc>();
}

double TimePrimFunc(const tir::PrimFunc& func) {
  ICHECK(func.defined()) << "TimePrimFunc called with an undefined PrimFunc";
  LOG(INFO) << "  TimePrimFunc:\n" << func;
  Target target = Target::Current(/*allow_not_defined=*/true);
  if (target.defined() && target->kind->name == "cuda") {
    return TimePrimFuncCUDA(func);
  }
  LOG(INFO) << "  TimePrimFunc: no CUDA target in scope; profiling disabled";
  return -1.0;
}

double TimePrimFuncCUDA(const tir::PrimFunc& func) {
  ICHECK(func.defined()) << "TimePrimFuncCUDA called with an undefined PrimFunc";
  Target target = Target::Current(/*allow_not_defined=*/true);
  ICHECK(target.defined())
      << "TimePrimFuncCUDA requires a target in the current context: wrap the "
         "FuseOps call in `with target:` so Target::Current() is set.";

  ffi::Optional<ffi::Function> kernel = BuildPrimFuncCUDA(func, target);
  if (!kernel) return -1.0;

  // TODO: materialize device args and time the kernel.
  LOG(INFO) << "  TimePrimFuncCUDA: kernel built; timing not implemented yet";
  return -1.0;
}

}  // namespace relax
}  // namespace tvm
