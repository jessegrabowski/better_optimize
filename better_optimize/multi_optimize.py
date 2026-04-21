import contextlib
import logging
import multiprocessing as mp
import pickle
import sys
import threading

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, NamedTuple

import numpy as np
import pandas as pd

from rich.console import Console
from rich.progress import TaskID
from rich.table import Table
from scipy.optimize import OptimizeResult
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from better_optimize.constants import MINIMIZE_MODE_KWARGS, ROOT_MODE_KWARGS
from better_optimize.utilities import ToggleableProgress
from better_optimize.wrapper import build_progress_bar

_log = logging.getLogger(__name__)

_QUEUE_POLL_INTERVAL = 0.05  # seconds between progress-queue polls in the listener thread

InitStrategy = Literal["uniform", "normal", "sobol", "lhs"] | Callable


def generate_starts(
    x0: np.ndarray,
    n_runs: int,
    strategy: InitStrategy,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    init_scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Create ``n_runs`` starting points from ``x0`` according to ``strategy``."""
    n_params = x0.shape[0]

    # str is callable in Python, so we must exclude it explicitly
    if callable(strategy) and not isinstance(strategy, str):
        return [np.asarray(row) for row in strategy(x0, n_runs, rng)]

    if strategy == "normal":
        return [x0 + init_scale * rng.standard_normal(n_params) for _ in range(n_runs)]

    if bounds is None:
        raise ValueError(f"strategy={strategy!r} requires bounds")
    low = np.broadcast_to(bounds[0], n_params)
    high = np.broadcast_to(bounds[1], n_params)

    if strategy == "uniform":
        return [rng.uniform(low, high) for _ in range(n_runs)]

    samplers = {"sobol": qmc.Sobol, "lhs": qmc.LatinHypercube}
    if strategy not in samplers:
        raise ValueError(f"Unknown init strategy: {strategy!r}")

    raw = samplers[strategy](d=n_params, seed=rng).random(n_runs)
    scaled = qmc.scale(raw, low, high)
    return [scaled[i] for i in range(n_runs)]


class ProgressProxy:
    """Mimics the ``ToggleableProgress`` interface but serializes calls over a ``Queue``."""

    def __init__(self, queue: mp.Queue, task_id):
        self._queue = queue
        self._task_id = task_id

    def update(self, task_id, **kwargs):
        self._queue.put(("update", task_id, kwargs))

    def add_task(self, *args, **kwargs):
        return self._task_id

    def reset(self, task_id, **kwargs):
        self._queue.put(("reset", task_id, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _drain_progress_queue(
    queue: mp.Queue,
    progress: ToggleableProgress,
    stop: threading.Event,
) -> None:
    """Drain ``queue`` into ``progress`` until ``stop`` is set.  Runs on a daemon thread."""
    while not stop.is_set():
        try:
            action, task_id, kwargs = queue.get(timeout=_QUEUE_POLL_INTERVAL)
        except Exception:
            continue
        getattr(progress, action)(task_id, **kwargs)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Translate joblib-style n_jobs into an actual core count."""
    import os

    if n_jobs < 0:
        # joblib convention: -1 → cpu_count, -2 → cpu_count-1, …
        return max(1, os.cpu_count() + 1 + n_jobs)
    return max(1, n_jobs)


class BlasConfig(NamedTuple):
    """Result of :func:`setup_blas_cores`."""

    limiter: Callable[[], contextlib.AbstractContextManager]
    per_worker: int | None


def setup_blas_cores(
    blas_cores: int | None | str,
    n_jobs: int,
    mp_ctx: mp.context.BaseContext | None,
) -> BlasConfig:
    """Resolve the BLAS thread budget into a main-process limiter and a per-worker quota.

    Returns
    -------
    BlasConfig
        ``limiter`` – callable returning a context manager that caps total BLAS threads.
        ``per_worker`` – per-worker thread count, or ``None`` (unlimited).
    """
    n_jobs = _resolve_n_jobs(n_jobs)

    if isinstance(blas_cores, str):
        if blas_cores != "auto":
            raise ValueError(f"blas_cores must be int, 'auto', or None — got {blas_cores!r}")
        blas_cores = n_jobs

    if blas_cores is None:
        return BlasConfig(contextlib.nullcontext, None)

    per_worker = max(1, blas_cores // n_jobs)

    # Forked children inherit the parent's thread-pool state, so a
    # main-process limiter would be redundant (and potentially racy).
    is_fork = mp_ctx is not None and mp_ctx.get_start_method() == "fork"
    if is_fork:
        return BlasConfig(contextlib.nullcontext, per_worker)

    def joined_limiter():
        return threadpool_limits(limits=blas_cores)

    return BlasConfig(joined_limiter, per_worker)


def _is_pickle_error(exc: Exception) -> bool:
    if isinstance(exc, pickle.PickleError):
        return True
    return isinstance(exc, AttributeError) and str(exc).startswith("Can't pickle")


def _run_single(
    run_index: int,
    x0: np.ndarray,
    solver: Callable[..., OptimizeResult],
    solver_kwargs: dict,
    progress: ToggleableProgress | ProgressProxy,
    task_id,
    blas_cores_per_worker: int | None = None,
) -> OptimizeResult:
    """Execute one optimization run, optionally under a per-worker BLAS thread limit."""
    limiter = (
        threadpool_limits(limits=blas_cores_per_worker)
        if blas_cores_per_worker is not None
        else contextlib.nullcontext()
    )
    try:
        with limiter:
            result = solver(x0=x0, progressbar=progress, progress_task=task_id, **solver_kwargs)
    except Exception as exc:
        _log.warning(
            "Run %d failed with %s: %s",
            run_index,
            type(exc).__name__,
            exc,
        )
        result = OptimizeResult(
            x=x0,
            fun=np.inf,
            success=False,
            message=f"Solver raised {type(exc).__name__}: {exc}",
        )

    result.run_index = run_index
    result.x0 = x0
    return result


@dataclass
class MultiStartResult:
    """Collected results from a multi-start optimization campaign."""

    results: list[OptimizeResult]
    sort_key: Callable[[OptimizeResult], float] = field(
        default_factory=lambda: lambda res: float(res.fun)
    )

    def ranked(self, ascending: bool = True) -> list[OptimizeResult]:
        return sorted(self.results, key=self.sort_key, reverse=not ascending)

    @property
    def best(self) -> OptimizeResult:
        return min(self.results, key=self.sort_key)

    @property
    def x_best(self) -> np.ndarray:
        return self.best.x

    @property
    def fun_best(self) -> float:
        return self.sort_key(self.best)

    def top_k(self, k: int = 5) -> list[OptimizeResult]:
        return self.ranked()[:k]

    @property
    def success_rate(self) -> float:
        return sum(r.success for r in self.results) / len(self.results)

    def _format_result_row(self, rank: int, res: OptimizeResult) -> tuple[str, ...]:
        """Format a single ``OptimizeResult`` into a table row tuple."""
        grad_str = (
            f"{np.linalg.norm(res.jac):.6e}" if hasattr(res, "jac") and res.jac is not None else ""
        )
        return (
            str(rank),
            str(getattr(res, "run_index", "?")),
            f"{self.sort_key(res):.8e}",
            grad_str,
            "✓" if res.success else "✗",
            str(getattr(res, "nit", "—")),
            str(getattr(res, "message", "")),
        )

    _SUMMARY_COLUMNS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("Rank", "right"),
        ("Run", "right"),
        ("f(x*)", "right"),
        ("||grad||", "right"),
        ("Success", "center"),
        ("Iterations", "right"),
        ("Message", "left"),
    )

    def summary(self, top_k: int | None = None) -> None:
        table = Table(title="Multi-Start Optimization Results", show_lines=True)
        for col, justify in self._SUMMARY_COLUMNS:
            table.add_column(col, justify=justify)

        display = self.ranked()[:top_k]

        for rank, res in enumerate(display, 1):
            table.add_row(*self._format_result_row(rank, res))
        Console(file=sys.stderr).print(table)

    def _result_to_dict(self, rank: int, res: OptimizeResult) -> dict:
        """Convert a single ``OptimizeResult`` into a flat dict for DataFrame export."""
        return {
            "rank": rank,
            "run_index": getattr(res, "run_index", None),
            "fun": self.sort_key(res),
            "success": res.success,
            "nit": getattr(res, "nit", None),
            "x": res.x,
            "message": str(getattr(res, "message", "")),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self._result_to_dict(rank, res) for rank, res in enumerate(self.ranked(), 1)]
        )


class _MultiStart:
    """Implementation behind :func:`multistart`. Not part of the public API."""

    def __init__(
        self,
        solver: Callable[..., OptimizeResult],
        x0: np.ndarray | list[np.ndarray],
        solver_kwargs: dict | None = None,
        sort_key: Callable[[OptimizeResult], float] | None = None,
        n_runs: int = 16,
        init_strategy: InitStrategy = "normal",
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        init_scale: float = 1.0,
        backend: Literal["sequential", "loky", "threading"] = "sequential",
        n_jobs: int = -1,
        seed: int | None = None,
        progressbar: bool = True,
        mp_ctx: mp.context.BaseContext | None = None,
        blas_cores: int | None | Literal["auto"] = "auto",
    ):
        self._solver = solver
        self._solver_kwargs = solver_kwargs or {}
        self._sort_key = sort_key or (lambda res: float(res.fun))
        self._backend = backend
        self._n_jobs = n_jobs
        self._progressbar = progressbar
        self._mp_ctx = mp_ctx
        self._blas_cores = blas_cores

        if isinstance(x0, list):
            self._starting_points = [np.asarray(xi) for xi in x0]
        else:
            self._starting_points = generate_starts(
                np.asarray(x0),
                n_runs,
                init_strategy,
                bounds,
                init_scale,
                np.random.default_rng(seed),
            )

    def run(self) -> MultiStartResult:
        if self._backend == "sequential":
            return self._run_sequential()

        try:
            return self._run_parallel()
        except (pickle.PickleError, AttributeError) as exc:
            if not _is_pickle_error(exc):
                raise
            _log.warning("Could not pickle objective — falling back to sequential execution.")
            return self._run_sequential()

    def _run_sequential(self) -> MultiStartResult:
        blas_config = setup_blas_cores(self._blas_cores, n_jobs=1, mp_ctx=self._mp_ctx)
        progress = self._build_progress()
        task_ids = self._register_tasks(progress)

        with blas_config.limiter(), progress:
            results = [
                _run_single(
                    run_index=i,
                    x0=x0_i,
                    solver=self._solver,
                    solver_kwargs=self._solver_kwargs,
                    progress=progress,
                    task_id=task_id,
                )
                for i, (x0_i, task_id) in enumerate(zip(self._starting_points, task_ids))
            ]
        return MultiStartResult(results=results, sort_key=self._sort_key)

    def _run_parallel(self) -> MultiStartResult:
        from joblib import Parallel, delayed

        blas_config = setup_blas_cores(
            self._blas_cores,
            n_jobs=self._n_jobs,
            mp_ctx=self._mp_ctx,
        )
        progress = self._build_progress()
        task_ids = self._register_tasks(progress)

        manager = mp.Manager()
        queue = manager.Queue()
        stop = threading.Event()
        listener = threading.Thread(
            target=_drain_progress_queue,
            args=(queue, progress, stop),
            daemon=True,
        )

        with blas_config.limiter(), progress:
            listener.start()
            try:
                results = Parallel(n_jobs=self._n_jobs, backend=self._backend)(
                    delayed(_run_single)(
                        run_index=i,
                        x0=x0_i,
                        solver=self._solver,
                        solver_kwargs=self._solver_kwargs,
                        progress=ProgressProxy(queue, task_id),
                        task_id=task_id,
                        blas_cores_per_worker=blas_config.per_worker,
                    )
                    for i, (x0_i, task_id) in enumerate(zip(self._starting_points, task_ids))
                )
            finally:
                stop.set()
                listener.join(timeout=1.0)
                manager.shutdown()

        return MultiStartResult(results=results, sort_key=self._sort_key)

    _SOLVER_LABELS = {
        "minimize": "Minimizing",
        "root": "Finding Roots",
        "basinhopping": "Basinhopping",
    }

    @property
    def _task_description(self) -> str:
        name = getattr(self._solver, "__name__", "")
        return self._SOLVER_LABELS.get(name, name.title() or "Optimizing")

    @property
    def _is_root(self) -> bool:
        return getattr(self._solver, "__name__", "") == "root"

    def _build_progress(self) -> ToggleableProgress:
        method = self._solver_kwargs.get("method", "")

        if self._is_root:
            mode_info = ROOT_MODE_KWARGS.get(method, {})
            use_jac = mode_info.get("uses_jac", False) or "jac" in self._solver_kwargs
            use_rayleigh = False
        else:
            mode_info = MINIMIZE_MODE_KWARGS.get(method, {})
            use_jac = mode_info.get("uses_grad", False) or "jac" in self._solver_kwargs
            has_hess = "hess" in self._solver_kwargs or "hessp" in self._solver_kwargs
            use_rayleigh = use_jac and has_hess

        return build_progress_bar(
            description=self._task_description,
            progressbar=self._progressbar,
            root=self._is_root,
            use_jac=use_jac,
            use_rayleigh=use_rayleigh,
        )

    def _register_tasks(self, progress: ToggleableProgress) -> list[TaskID | None]:
        # Include all possible fields so columns never hit a missing key.
        # Unused fields (e.g. rayleigh when no curvature column) are silently ignored.
        return [
            progress.add_task(
                description=self._task_description,
                total=None,
                f_value=0.0,
                grad_norm=0.0,
                rayleigh=0.0,
            )
            for _ in self._starting_points
        ]


def multi_optimize(
    solver: Callable[..., OptimizeResult],
    x0: np.ndarray | list[np.ndarray],
    solver_kwargs: dict | None = None,
    sort_key: Callable[[OptimizeResult], float] | None = None,
    n_runs: int = 16,
    init_strategy: InitStrategy = "normal",
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    init_scale: float = 1.0,
    backend: Literal["sequential", "loky", "threading"] = "loky",
    n_jobs: int = -1,
    seed: int | None = None,
    progressbar: bool = True,
    mp_ctx: mp.context.BaseContext | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
) -> MultiStartResult:
    """
    Run a solver from many starting points and return the collected results.

    Parameters
    ----------
    solver: Callable
        Any solver with signature (x0, **kwargs) -> OptimizeResult, such as minimize or root
    x0: np.ndarray or list of np.ndarray
        A single starting point used as the center for init_strategy, or an explicit list of
        starting points. When a list is provided, n_runs and init_strategy are ignored.
    solver_kwargs: dict, optional
        Keyword arguments forwarded verbatim to solver on every call
    sort_key: Callable, optional
        Function mapping an OptimizeResult to a float used for ranking. Defaults to
        lambda res: res.fun, which ranks results by their optimized objective value.
    n_runs: int
        Number of random restarts. Ignored when x0 is a list.
    init_strategy: str or Callable
        How to generate starting points from x0. One of "uniform", "normal", "sobol", "lhs",
        or a callable with signature (x0, n_runs, rng) → list of arrays.
    bounds: tuple, optional
        (low, high) arrays or scalars, required by bounded init strategies
    init_scale: float
        Standard deviation for the "normal" init strategy
    backend: str
        Execution backend. One of "sequential", "loky", or "threading".
    n_jobs: int
        Number of parallel workers. -1 means all cores.
    seed: int, optional
        Random seed for reproducible starting points
    progressbar: bool
        Whether to display per-run progress bars
    mp_ctx: multiprocessing context, optional
        Explicit multiprocessing context (spawn, fork, or forkserver)
    blas_cores: int, str, or None
        Total BLAS/OpenMP thread budget. "auto" splits evenly across workers, None is unlimited.

    Returns
    -------
    result: MultiStartResult
        Collected results with ranking, summary, and export utilities
    """
    return _MultiStart(
        solver=solver,
        x0=x0,
        solver_kwargs=solver_kwargs,
        sort_key=sort_key,
        n_runs=n_runs,
        init_strategy=init_strategy,
        bounds=bounds,
        init_scale=init_scale,
        backend=backend,
        n_jobs=n_jobs,
        seed=seed,
        progressbar=progressbar,
        mp_ctx=mp_ctx,
        blas_cores=blas_cores,
    ).run()


__all__ = ["multi_optimize", "MultiStartResult", "generate_starts"]
