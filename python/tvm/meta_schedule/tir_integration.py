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
"""MetaSchedule-TIR integration"""
from typing import List, Mapping, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal
from tvm_ffi import register_global_func

# isort: on
from tvm import ir, tir
from tvm.target import Target
from tvm.tir.expr import IntImm

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext, _normalize_mod
from .utils import fork_seed


def tune_tir(  # pylint: disable=too-many-locals
    mod: Union[ir.IRModule, tir.PrimFunc],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    *,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
    module_equality: str = "structural",
    special_space: Optional[Mapping[str, SpaceGenerator.SpaceGeneratorType]] = None,
    post_optimization: Optional[bool] = False,
) -> Database:
    """Tune a TIR function or an IRModule of TIR functions.

    Parameters
    ----------
    mod : Union[ir.IRModule, tir.PrimFunc]
        The TIR IRModule to tune.
    target : Union[str, Target]
        The target to tune for.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The seed for the random number generator.
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
    special_space : Optional[Mapping[str, SpaceGenerator.SpaceGeneratorType]]
        A mapping from task name to a special space generator for that task.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    if isinstance(mod, tir.PrimFunc):
        mod = _normalize_mod(mod)

    named_tasks: List[Tuple[str, tir.PrimFunc]] = []
    for gv, func in mod.functions_items():  # pylint: disable=invalid-name
        if isinstance(func, tir.PrimFunc):
            named_tasks.append((gv.name_hint, func))
    named_tasks.sort(key=lambda x: x[0])

    task_names = [x for x, _ in named_tasks]
    tasks: List[TuneContext] = []
    for task_name, task_func, logger, rand_state in zip(
        task_names,
        [x for _, x in named_tasks],
        get_loggers_from_work_dir(work_dir, task_names),
        fork_seed(seed, n=len(named_tasks)),
    ):
        if special_space and task_name in special_space:
            task_space = special_space[task_name]
        else:
            task_space = space
        if task_space is None:
            continue
        tasks.append(
            TuneContext(
                mod=task_func,
                target=target,
                space_generator=task_space,
                search_strategy=strategy,
                task_name=task_name,
                rand_state=rand_state,
                num_threads=num_tuning_cores,
                logger=logger,
            ).clone()
        )
    return tune_tasks(
        tasks=tasks,
        task_weights=[1.0] * len(tasks),
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        module_equality=module_equality,
        post_optimization=post_optimization,
    )


@register_global_func("tvm.meta_schedule.tune_tir")
def _tune_tir(
    mod: Union[ir.IRModule, tir.PrimFunc],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    *,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "round-robin",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
) -> Database:
    """Interface with tuning api to tune a TIR program.

    Parameters
    ----------
    mod : Union[ir.IRModule, tir.PrimFunc]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The seed for the random number generator.

    Returns
    -------
    ret_mod : IRModule
        IRModule
    """
    if isinstance(max_trials_global, IntImm):
        max_trials_global = int(max_trials_global)
    tune_tir(
        mod,
        target,
        work_dir,
        max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        space=space,
        strategy=strategy,
        num_tuning_cores=num_tuning_cores,
        seed=seed,
    )
    # Return original IRModule
    # This pass only makes optimization decision
    return mod


def compile_tir(
    database: Database,
    mod: Union[ir.IRModule, tir.PrimFunc],
    target: Union[Target, str],
) -> tir.Schedule:
    """Compile a TIR to tir.Schedule, according to the records in the database.

    Parameters
    ----------
    database : Database
        The database of tuning records.
    mod : Union[ir.IRModule, tir.PrimFunc]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.

    Returns
    -------
    sch : tir.Schedule
        The best schedule found in the database.
    """
    mod = _normalize_mod(mod)
    if not isinstance(target, Target):
        target = Target(target)
    return database.query_schedule(mod, target, workload_name="main")


@register_global_func("relax.dnnf.MetaScheduleSchedulePrimFunc")
def _ms_schedule_primfunc(
    func: tir.PrimFunc,
    target: Union[str, Target],
    max_trials: int,
) -> Optional[tir.PrimFunc]:
    """Tune a single PrimFunc with MetaSchedule and return the tuned PrimFunc.

    Called from the in-pass cost oracle (``BuildPrimFuncGPU`` in
    ``graph_partitioner.cc``) as a replacement for ``DefaultGPUSchedule``: tune
    ``func`` for ``max_trials`` trials, then apply the best record. Returns
    ``None`` (so C++ can fall back to ``DefaultGPUSchedule``) if no valid
    schedule is found within the budget.

    The tuning database is persisted to a stable work directory so it survives
    the run and can be inspected / reused: ``$DNNF_MS_WORKDIR`` (default
    ``./dnnf_ms_workdir``), with a per-kernel subdirectory keyed by the
    PrimFunc's structural hash so structurally-identical kernels share a
    database.

    Parameters
    ----------
    func : tir.PrimFunc
        The PrimFunc to tune.
    target : Union[str, Target]
        The target to tune for.
    max_trials : int
        The MetaSchedule trial budget.

    Returns
    -------
    tuned : Optional[tir.PrimFunc]
        The tuned PrimFunc, or ``None`` if tuning found no schedule.
    """
    import os  # pylint: disable=import-outside-toplevel

    if isinstance(max_trials, IntImm):
        max_trials = int(max_trials)
    base_dir = os.environ.get("DNNF_MS_WORKDIR", os.path.join(os.getcwd(), "dnnf_ms_workdir"))
    work_dir = os.path.join(base_dir, str(ir.structural_hash(func)))
    os.makedirs(work_dir, exist_ok=True)
    database = tune_tir(
        func,
        target,
        work_dir,
        max_trials_global=max_trials,
        num_trials_per_iter=max_trials,
    )
    sch = compile_tir(database, func, target)
    if sch is None:
        return None
    return sch.mod["main"]


def _dlight_has_thread_binding(func: tir.PrimFunc) -> bool:
    """True iff some For loop in `func` carries a GPU thread binding -- i.e. the
    func is really GPU-scheduled, not just claimed by a rule that returned it
    untouched. dlight rules sometimes "claim" a PrimFunc (return a length-1
    space) without binding any loop; committing such a func and marking it
    is_scheduled would hide it from the DefaultGPUSchedule net and then trip
    VerifyMemory ("directly accessed by host memory") at codegen."""
    found = [False]
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda s: found.__setitem__(
            0, found[0] or (isinstance(s, tir.For) and s.thread_binding is not None)
        ),
    )
    return found[0]


@register_global_func("relax.dnnf.DlightSchedulePrimFunc")
def _dlight_schedule_primfunc(
    func: tir.PrimFunc,
    target: Union[str, Target],
) -> tir.PrimFunc:
    """Schedule a single PrimFunc with dlight's GPU rules, DefaultGPUSchedule net.

    Called from the in-pass cost oracle (``BuildPrimFuncGPU`` in
    ``dnnf_profiler.cc``) as the ``kDlight`` scheduling mode: a deterministic
    middle ground between ``DefaultGPUSchedule`` (cheap heuristic) and
    MetaSchedule (tuned but slow). Returns a *fully* scheduled PrimFunc -- dlight
    where a GPU rule genuinely thread-binds it, DefaultGPUSchedule otherwise --
    so the caller never has to schedule it again.

    This mirrors ``expr/dlight_compile.py``'s proven ``_apply_dlight`` recipe:

    1. Try the dlight GPU rules (Matmul / GEMV / Reduction / GeneralReduction /
       Transpose / Fallback) on the func. A rule that *raises* (e.g. GEMV's
       normalize asserts on conv PrimFuncs) is skipped; a rule that "claims" the
       func but produces no thread binding (``_dlight_has_thread_binding``) is
       rejected -- only a genuinely thread-bound schedule is accepted, then
       marked ``tir.is_scheduled``.
    2. Run ``DefaultGPUSchedule`` over the module. It skips the already-scheduled
       func and binds whatever dlight could not -- the same catch-all the
       TVM/MLC pipeline uses for conv-heavy (non-LLM) models. This keeps conv
       PrimFuncs (which no dlight rule binds well, and which an aggressive
       matmul-style tiling would over-allocate past the device shared-memory
       limit -> ptxas failure) on the safe DefaultGPUSchedule path.

    Drives dlight's ``ScheduleRule`` objects directly rather than the
    ``dlight.ApplyDefaultSchedule`` module pass, whose FFI ``__init__`` (sets
    ``_inst``) is shadowed by the apache-tvm-ffi PyPI wheel here -- the same
    shadowing CLAUDE.md documents for ``BlockBuilder`` / MetaSchedule's
    ``TVMDerivedObject``. The ``ScheduleRule`` objects and their ``apply``
    methods are plain Python and are unaffected.

    Parameters
    ----------
    func : tir.PrimFunc
        The PrimFunc to schedule (carries global_symbol "tir_function" and the
        kTarget attr set by BuildPrimFuncGPU).
    target : Union[str, Target]
        The target to schedule for.

    Returns
    -------
    scheduled : tir.PrimFunc
        The fully scheduled PrimFunc (global_symbol preserved).
    """
    # pylint: disable=import-outside-toplevel
    import tvm
    from tvm import dlight as dl
    from tvm.dlight.base.transform import _get_target, _is_scheduled

    if not isinstance(target, Target):
        target = Target(target)
    rules = (
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Transpose(),
        dl.gpu.Fallback(),
    )

    def _first_bound_schedule(f, tgt):
        for rule in rules:
            try:
                # `tunable` is positional: dlight GPU rules name the 3rd param
                # `_` and reject it as a keyword (`tunable=False` -> TypeError).
                space = rule.apply(f, tgt, False)
            except Exception:  # pylint: disable=broad-except
                continue  # rule can't handle this func; skip it
            if space is None:
                continue
            # A non-tunable rule returns a bare tir.Schedule or a (ffi.Array, not
            # list/tuple) sequence of them -- duck-type the length, don't isinstance.
            if isinstance(space, tir.Schedule):
                space = [space]
            try:
                n = len(space)
            except TypeError:
                continue
            if n == 1:
                scheduled = space[0].mod["main"]
                if _dlight_has_thread_binding(scheduled):
                    return scheduled.with_attr("tir.is_scheduled", True)
        return None

    # global_symbol stays "tir_function" so the built runtime module exposes the
    # kernel under the name BuildPrimFuncGPU looks up.
    gv = ir.GlobalVar("tir_function")
    mod = ir.IRModule({gv: func})
    with target:
        f = mod[gv]
        if isinstance(f, tir.PrimFunc) and not _is_scheduled(f):
            sch = _first_bound_schedule(f, _get_target(f))
            if sch is not None:
                mod[gv] = sch
        # DefaultGPUSchedule binds whatever dlight left unscheduled.
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    return mod[gv]
