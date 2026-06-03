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
 * \brief In-pass cost oracle for the DNNFusion port: schedule, JIT-build, and
 *  time relax PrimFuncs on the GPU. Self-contained device-timing utilities with
 *  no dependency on GraphPartitioner state; see dnnf_profiler.h.
 */

#include "./dnnf_profiler.h"

#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>
#include <tvm/tir/transform.h>

#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

namespace {

// MetaSchedule trials used to schedule each profiled PrimFunc. The cost oracle
// tunes each kernel briefly (rather than the DefaultGPUSchedule heuristic) so
// the timed kernel reflects a tuned schedule.
constexpr int kProfileTuneTrials = 4;

// When false, skip MetaSchedule tuning entirely and schedule every profiled
// kernel with the DefaultGPUSchedule heuristic (used for fast smoke tests). Set
// true to tune each profiled kernel with kProfileTuneTrials MS trials.
constexpr bool kUseMetaSchedule = true;

// Schedule `func` and build it on `target`, returning the callable device
// kernel; nullopt if scheduling or build fails. An unscheduled PrimFunc has no
// thread bindings, so it must be scheduled before tir.build. Scheduling reads
// Target::Current(), so build stays under the same target scope that drives the
// host/device split. The PrimFunc is MetaSchedule-tuned (kProfileTuneTrials
// trials) via the `relax.dnnf.MetaScheduleSchedulePrimFunc` Python helper, with
// a DefaultGPUSchedule fallback when MS is unavailable or finds no schedule.
ffi::Optional<ffi::Function> BuildPrimFuncGPU(const tir::PrimFunc& func, const Target& target) {
  try {
    tvm::With<Target> target_scope(target);
    tir::PrimFunc named = WithAttr(func, tvm::attr::kGlobalSymbol, ffi::String("tir_function"));
    GlobalVar gv("tir_function");
    ffi::Map<GlobalVar, BaseFunc> funcs;
    funcs.Set(gv, named);
    IRModule m(funcs);

    tir::PrimFunc scheduled;
    ffi::Optional<tir::PrimFunc> tuned;
    if (kUseMetaSchedule) {
      if (auto ms_schedule =
              ffi::Function::GetGlobal("relax.dnnf.MetaScheduleSchedulePrimFunc")) {
        ffi::Any ret = (*ms_schedule)(named, target, kProfileTuneTrials);
        tuned = ret.try_cast<tir::PrimFunc>();
      }
    }
    if (tuned) {
      scheduled = tuned.value();
    } else {
      // No tuned schedule (MS disabled for smoke test, MS not imported, or no
      // valid record in the trial budget): fall back to DefaultGPUSchedule.
      LOG(INFO) << "  MetaSchedule disabled/unavailable; DefaultGPUSchedule fallback";
      m = tir::transform::DefaultGPUSchedule()(m);
      scheduled = Downcast<tir::PrimFunc>(m->Lookup(gv));
    }

    const auto build = tvm::ffi::Function::GetGlobalRequired("tir.build");
    ffi::Module rt_module = build(scheduled, target).cast<ffi::Module>();
    return rt_module->GetFunction("tir_function").value();
  } catch (const tvm::Error& err) {
    LOG(INFO) << "  build failed: " << err.what();
    return std::nullopt;
  }
}

// Materialize one device tensor per buffer param of `func` (inputs + outputs, in
// param order): allocate on `cpu_dev`, random-fill float32 buffers with values in
// [-1, 1], then copy to `cuda_dev` (kernels can't be fed host pointers, and we
// can't write GPU memory directly). nullopt if any param is not a buffer or has a
// dynamic shape.
ffi::Optional<std::vector<runtime::Tensor>> MakeRandomDeviceArgs(const tir::PrimFunc& func,
                                                                 DLDevice cuda_dev,
                                                                 DLDevice cpu_dev) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<runtime::Tensor> args;
  for (const tir::Var& param : func->params) {
    ffi::Optional<tir::Buffer> opt_buf = func->buffer_map.Get(param);
    if (!opt_buf) {
      LOG(INFO) << "  param " << param << " is not a buffer; skip profiling";
      return std::nullopt;
    }
    tir::Buffer buf = opt_buf.value();
    std::vector<int64_t> shape;
    for (const PrimExpr& dim : buf->shape) {
      const auto* imm = dim.as<IntImmNode>();
      if (imm == nullptr) {
        LOG(INFO) << "  param " << param << " has a dynamic shape; skip profiling";
        return std::nullopt;
      }
      shape.push_back(imm->value);
    }
    runtime::Tensor host_t = runtime::Tensor::Empty(ffi::Shape(shape), buf->dtype, cpu_dev);
    if (buf->dtype == DataType::Float(32)) {
      int64_t numel = 1;
      for (int64_t d : shape) numel *= d;
      float* p = static_cast<float*>(host_t->data);
      for (int64_t i = 0; i < numel; ++i) p[i] = dist(rng);
    }
    args.push_back(host_t.CopyTo(cuda_dev));  // CopyTo triggers a TVMSynchronize
  }
  return args;
}

// Run `kernel` on `args` once untimed (warmup, to absorb cold-cache / first-
// dispatch PTX-JIT cost), then time batches of kProfileRuns launches with the
// device timer. On CUDA this uses cudaEvent elapsed time, avoiding host launch
// and StreamSync overhead in the cost model.
double TimeKernel(const ffi::Function& kernel, const std::vector<runtime::Tensor>& args,
                  DLDevice cuda_dev) {
  // Pack args once (AnyView is non-owning, so `args` must outlive the calls).
  std::vector<ffi::AnyView> packed(args.size());
  for (size_t i = 0; i < args.size(); ++i) packed[i] = args[i];

  runtime::DeviceAPI* dev_api = runtime::DeviceAPI::Get(cuda_dev);

  {
    ffi::Any ret;
    kernel.CallPacked(ffi::PackedArgs(packed.data(), packed.size()), &ret);
    dev_api->StreamSync(cuda_dev, nullptr);
  }

  double total_us = 0.0;
  for (int repeat = 0; repeat < kProfileRepeats; ++repeat) {
    runtime::Timer timer = runtime::Timer::Start(cuda_dev);
    for (int r = 0; r < kProfileRuns; ++r) {
      ffi::Any ret;
      kernel.CallPacked(ffi::PackedArgs(packed.data(), packed.size()), &ret);
    }
    timer->Stop();
    total_us += static_cast<double>(timer->SyncAndGetElapsedNanos()) / 1000.0 / kProfileRuns;
  }
  return total_us / kProfileRepeats;
}

// Find the call_tir Call bound to `var` in any relax function of `mod`.
ffi::Optional<Call> FindCallTIR(const IRModule& mod, const Object* var) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  for (const auto& kv : mod->functions) {
    const auto* func = kv.second.as<FunctionNode>();
    if (func == nullptr) continue;
    for (const BindingBlock& block : func->body->blocks) {
      for (const Binding& binding : block->bindings) {
        const auto* vb = binding.as<VarBindingNode>();
        if (vb == nullptr || vb->var.get() != var) continue;
        const auto* call = vb->value.as<CallNode>();
        if (call != nullptr && call->op.same_as(call_tir_op)) {
          return ffi::GetRef<Call>(call);
        }
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

// Construct the inner `fused_block` relax Function over `nodes` (given in
// producer-before-consumer order, i.e. sorted by IndexedForwardGraph index): a
// kPrimitive dataflow block chaining every node's call_tir. Each callee PrimFunc
// is registered in `bb`; an input that is the output of another node in the set
// is wired var-to-var (the internal edges), while every other input -- including
// constants -- becomes a tensor param, so the kernel is self-contained. Each
// node output not consumed inside the set becomes a function output (a single
// var, or a Tuple if several). Returns nullopt if any node lacks a call_tir
// binding.
ffi::Optional<Function> MakeFusedBlockFunc(
    BlockBuilder bb, const IRModule& mod,
    const std::vector<const IndexedForwardGraph::Node*>& nodes) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");

  // Resolve every node's call_tir up front so we can bail before mutating `bb`.
  std::vector<Call> calls;
  calls.reserve(nodes.size());
  for (const auto* n : nodes) {
    ffi::Optional<Call> c = FindCallTIR(mod, n->ref);
    if (!c) return std::nullopt;
    calls.push_back(c.value());
  }

  ffi::Array<Var> params;
  int pidx = 0;
  std::unordered_map<const Object*, Var> produced;  // node->ref -> emitted var
  std::unordered_set<const Object*> consumed;        // outputs used within the set

  bb->BeginDataflowBlock();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Call& call = calls[i];
    auto pf = Downcast<tir::PrimFunc>(mod->Lookup(Downcast<GlobalVar>(call->args[0])));
    GlobalVar callee = bb->AddFunction(pf, "k" + std::to_string(i));
    ffi::Array<Expr> args;
    for (const Expr& a : Downcast<Tuple>(call->args[1])->fields) {
      auto it = produced.find(a.get());
      if (it != produced.end()) {
        consumed.insert(a.get());  // internal edge: another node's output feeds this op
        args.push_back(it->second);
      } else {
        Var param("p" + std::to_string(pidx++), GetStructInfo(a));
        params.push_back(param);
        args.push_back(param);
      }
    }
    Call inner(call_tir_op, {callee, Tuple(args)}, Attrs(), call->sinfo_args);
    produced[nodes[i]->ref] = bb->Emit(inner);
  }

  // Any node whose output is not consumed by another node in the set escapes.
  // The highest-index node is never consumed internally, so there is >= 1.
  ffi::Array<Expr> outs;
  for (const auto* n : nodes) {
    if (!consumed.count(n->ref)) outs.push_back(produced[n->ref]);
  }
  Expr out_expr;
  if (outs.size() == 1) {
    out_expr = outs[0];
  } else {
    out_expr = Tuple(outs);
  }
  Var out = bb->EmitOutput(out_expr);
  BindingBlock blk = bb->EndBlock();

  Expr body = bb->Normalize(out);
  body = bb->Normalize(SeqExpr({blk}, body));
  ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set(attr::kPrimitive, true);
  return Function(params, body, /*ret_struct_info=*/std::nullopt, /*is_pure=*/true, DictAttrs(attrs));
}

// Append a public `main` to `bb` that forwards fresh params straight to
// `callee_gv`. FuseTIR ends with DeadCodeElimination, which would otherwise reap
// the merged kernel since it has no caller in this standalone module.
void AppendMainCaller(BlockBuilder bb, const GlobalVar& callee_gv, const ffi::Array<Var>& params) {
  ffi::Array<Var> main_params;
  ffi::Array<Expr> main_args;
  for (const Var& p : params) {
    Var mp(p->name_hint() + "_in", GetStructInfo(p));
    main_params.push_back(mp);
    main_args.push_back(mp);
  }
  bb->BeginDataflowBlock();
  Var main_out = bb->EmitOutput(Call(callee_gv, main_args));
  BindingBlock main_blk = bb->EndBlock();
  Expr main_body = bb->Normalize(SeqExpr({main_blk}, bb->Normalize(main_out)));
  ffi::Map<ffi::String, ffi::Any> main_attrs;
  main_attrs.Set(tvm::attr::kGlobalSymbol, ffi::String("main"));
  Function main_fn(main_params, main_body, /*ret_struct_info=*/std::nullopt, /*is_pure=*/true,
                   DictAttrs(main_attrs));
  bb->AddFunction(main_fn, "main");
}

// Run FuseTIR over bb's context module and return `target_gv`'s merged PrimFunc;
// nullopt if FuseTIR throws or the result is not a PrimFunc.
ffi::Optional<tir::PrimFunc> FuseTIRAndExtract(BlockBuilder bb, const GlobalVar& target_gv) {
  IRModule scratch = bb->GetContextIRModule();
  try {
    scratch = transform::FuseTIR()(scratch);
  } catch (const tvm::Error& err) {
    LOG(INFO) << "  FuseTIR failed: " << err.what();
    return std::nullopt;
  }
  ffi::Optional<BaseFunc> merged =
      scratch->functions.Get(scratch->GetGlobalVar(target_gv->name_hint));
  if (merged) {
    if (const auto* pf = merged.value().as<tir::PrimFuncNode>()) {
      return ffi::GetRef<tir::PrimFunc>(pf);
    }
  }
  LOG(INFO) << "  fused result is not a PrimFunc";
  return std::nullopt;
}

}  // namespace

// Build `func` on the GTX 1070 (BuildPrimFuncGPU), feed it random device inputs
// (MakeRandomDeviceArgs), and time it (TimeKernel); -1 if the build fails or any
// param cannot be materialized.
double TimePrimFuncCUDA(const tir::PrimFunc& func) {
  Target target("nvidia/geforce-gtx-1070");
  DLDevice cuda_dev = {kDLCUDA, 0};
  DLDevice cpu_dev = {kDLCPU, 0};

  LOG(INFO) << "  PrimFunc to build:\n" << func;
  ffi::Optional<ffi::Function> kernel = BuildPrimFuncGPU(func, target);
  if (!kernel) return -1.0;
  ffi::Optional<std::vector<runtime::Tensor>> args = MakeRandomDeviceArgs(func, cuda_dev, cpu_dev);
  if (!args) return -1.0;
  return TimeKernel(kernel.value(), args.value(), cuda_dev);
}

// Build a kPrimitive module fusing all of `nodes` (producer-before-consumer
// order), run FuseTIR, and return the merged PrimFunc. Internal edges are wired
// var-to-var; every other input -- including constant args -- becomes a tensor
// param, so the merged kernel is self-contained. Returns nullopt if any node
// lacks a call_tir binding or FuseTIR cannot produce a single PrimFunc.
ffi::Optional<tir::PrimFunc> BuildFusedBlock(
    const IRModule& mod, const std::vector<const IndexedForwardGraph::Node*>& nodes) {
  BlockBuilder bb = BlockBuilder::Create(std::nullopt);
  ffi::Optional<Function> fused = MakeFusedBlockFunc(bb, mod, nodes);
  if (!fused) return std::nullopt;
  GlobalVar fused_gv = bb->AddFunction(fused.value(), "fused_block");
  AppendMainCaller(bb, fused_gv, fused.value()->params);
  return FuseTIRAndExtract(bb, fused_gv);
}

}  // namespace relax
}  // namespace tvm
