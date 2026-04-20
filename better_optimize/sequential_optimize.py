import inspect
import logging
import sys

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from rich.console import Console
from rich.progress import TaskID
from rich.table import Table
from scipy.optimize import OptimizeResult

from better_optimize.utilities import LRUCache1, check_f_is_fused_minimize
from better_optimize.wrapper import build_progress_bar

_log = logging.getLogger(__name__)

FailurePolicy = Literal["stop", "continue"]

_X0_MISSING = object()

_DRIVER_RESERVED_KEYS = frozenset({"solver", "name", "x0"})


@dataclass
class SequentialResult:
    """Collected results from a sequence of optimization stages.

    Mirrors the shape of :class:`better_optimize.multi_optimize.MultiStartResult` but with
    stage-chain semantics rather than independent-restart semantics.
    """

    best: OptimizeResult
    stage_results: list[OptimizeResult]
    best_stage: int
    success: bool
    message: str
    nit: int
    nfev: int
    sort_key: Callable[[OptimizeResult], float] = field(
        default_factory=lambda: lambda res: float(res.fun)
    )

    @property
    def x(self) -> np.ndarray:
        return self.best.x

    @property
    def fun(self) -> float:
        return float(self.best.fun)

    def summary(self) -> None:
        table = Table(title="Sequential Optimization Stages", show_lines=True)
        for col, justify in _SUMMARY_COLUMNS:
            table.add_column(col, justify=justify)
        for i, res in enumerate(self.stage_results):
            marker = "★" if i == self.best_stage else ""
            table.add_row(
                str(i),
                str(getattr(res, "solver_name", "")),
                str(getattr(res, "classification", "")),
                f"{float(getattr(res, 'fun', float('inf'))):.8e}",
                "✓" if getattr(res, "success", False) else "✗",
                str(getattr(res, "nit", "—")),
                marker,
                str(getattr(res, "message", "")),
            )
        Console(file=sys.stderr).print(table)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, res in enumerate(self.stage_results):
            rows.append(
                {
                    "stage": i,
                    "solver": getattr(res, "solver_name", ""),
                    "classification": getattr(res, "classification", ""),
                    "fun": float(getattr(res, "fun", float("inf"))),
                    "success": bool(getattr(res, "success", False)),
                    "nit": getattr(res, "nit", None),
                    "best_stage": i == self.best_stage,
                    "message": str(getattr(res, "message", "")),
                }
            )
        return pd.DataFrame(rows)


_SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Stage", "right"),
    ("Solver", "left"),
    ("Classification", "left"),
    ("f(x)", "right"),
    ("Success", "center"),
    ("Iter", "right"),
    ("Best", "center"),
    ("Message", "left"),
)


def _accepts_kwarg(solver: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(solver)
    except (ValueError, TypeError):
        return True
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _objective_kwarg_name(solver: Callable) -> str:
    """Return the parameter name this solver uses for the objective function.

    Our own wrappers use ``f``; scipy-style and basinhopping use ``func``. This
    lets sequential_optimize forward the cached objective under whichever name
    the stage solver actually accepts.
    """
    try:
        sig = inspect.signature(solver)
    except (ValueError, TypeError):
        return "f"
    params = sig.parameters
    if "f" in params:
        return "f"
    if "func" in params:
        return "func"
    return "f"


def _validate_stages(stages: list[dict[str, Any]], x0: np.ndarray | None) -> None:
    if not stages:
        raise ValueError("stages must be a non-empty list")
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(f"stage {i}: expected dict, got {type(stage).__name__}")
        solver = stage.get("solver")
        if solver is None:
            raise ValueError(f"stage {i}: missing required 'solver' key")
        if not callable(solver):
            raise TypeError(f"stage {i}: 'solver' must be callable")
        if "f" in stage or "func" in stage:
            raise ValueError(
                f"stage {i}: the objective is a top-level argument of "
                "sequential_optimize; do not set 'f' or 'func' per-stage"
            )

        stage_x0 = stage.get("x0", _X0_MISSING)
        if stage_x0 is _X0_MISSING:
            will_receive_x0 = (i > 0) or (x0 is not None)
        else:
            will_receive_x0 = stage_x0 is not None

        if will_receive_x0 and not _accepts_kwarg(solver, "x0"):
            raise ValueError(
                f"stage {i}: solver {solver!r} does not accept 'x0'; "
                'suppress x0 forwarding for this stage by setting "x0": None'
            )


def _classify(res: OptimizeResult, best_fun: float | None, f_cached: LRUCache1) -> str:
    """Return 'hard', 'soft', or 'usable'."""
    if getattr(res, "_bo_crashed", False):
        return "hard"

    x = np.asarray(getattr(res, "x", None), dtype=np.float64)
    if x.ndim == 0 or not np.all(np.isfinite(x)):
        return "hard"

    fun_attr = getattr(res, "fun", None)
    try:
        fun_val = float(fun_attr)
    except (TypeError, ValueError):
        fun_val = float("nan")

    if not np.isfinite(fun_val):
        try:
            recovered = float(f_cached.value(x))
        except Exception:
            return "hard"
        if not np.isfinite(recovered):
            return "hard"
        res.fun = recovered
        fun_val = recovered

    if best_fun is not None and fun_val > best_fun:
        return "soft"
    return "usable"


def sequential_optimize(
    f: Callable,
    x0: Sequence[float] | np.ndarray | None,
    stages: list[dict[str, Any]],
    args: tuple = (),
    progressbar: bool = True,
    verbose: bool = False,
    on_failure: FailurePolicy = "stop",
) -> SequentialResult:
    """Run a sequence of optimizers, chaining each stage's best-so-far into the next's x0.

    Each stage is a dict containing a ``"solver"`` key (callable) plus any kwargs the solver
    accepts. Stage output x is forwarded to the next stage's ``x0`` unless the stage dict
    explicitly sets ``"x0"``. Set ``"x0": None`` to suppress forwarding. An ``LRUCache1``
    wraps ``f`` at the driver level so cross-stage cache hits fire at stage handoffs.

    Parameters
    ----------
    f : Callable
        Objective function. May return a scalar, ``(f, grad)``, or ``(f, grad, hess)``.
    x0 : array-like or None
        Initial guess for the first stage (unless the stage overrides via its own ``"x0"``).
    stages : list of dict
        Each dict must contain ``"solver": Callable`` and may include ``"name": str`` plus
        any kwargs the solver accepts (e.g. ``"method"``, ``"bounds"``).
    args : tuple
        Extra positional args forwarded identically to every stage's objective call.
    progressbar : bool
        If True, renders a shared rich progress bar with one outer task for the chain plus
        per-stage tasks.
    verbose : bool
        Forwarded to stages that support it.
    on_failure : {"stop", "continue"}
        Policy for hard failures (exceptions or NaN/Inf in a stage's result). ``"stop"``
        terminates the chain and returns best-so-far. ``"continue"`` proceeds to the next
        stage using best-so-far as x0. Soft failures (finite regression) always continue.

    Returns
    -------
    SequentialResult
    """
    x0_arr = np.asarray(x0, dtype=np.float64) if x0 is not None else None

    _validate_stages(stages, x0_arr)

    if x0_arr is not None:
        has_grad, has_hess = check_f_is_fused_minimize(f, x0_arr, args or ())
    else:
        has_grad, has_hess = False, False
    f_returns_list = has_grad or has_hess
    f_cached = LRUCache1(
        f,
        f_returns_list=f_returns_list,
        copy_x=False,
        dtype=x0_arr.dtype if x0_arr is not None else None,
    )

    progress = build_progress_bar(
        description="Sequential",
        progressbar=progressbar,
        root=False,
        use_jac=True,
        use_hess=False,
    )
    outer_task = progress.add_task(
        description="Sequential",
        total=len(stages),
        f_value=float("inf"),
        grad_norm=0.0,
        hess_norm=0.0,
    )
    stage_tasks: list[TaskID] = []
    for i, stage in enumerate(stages):
        solver = stage["solver"]
        name = stage.get("name") or getattr(solver, "__name__", f"stage_{i}")
        stage_tasks.append(
            progress.add_task(
                description=name,
                total=None,
                f_value=float("inf"),
                grad_norm=0.0,
                hess_norm=0.0,
            )
        )

    stage_results: list[OptimizeResult] = []
    best_res: OptimizeResult | None = None
    best_stage: int = -1
    best_fun: float | None = None
    current_x: np.ndarray | None = x0_arr
    hard_failure_seen = False

    with progress:
        for i, stage in enumerate(stages):
            solver = stage["solver"]
            name = stage.get("name") or getattr(solver, "__name__", f"stage_{i}")
            stage_kwargs = {k: v for k, v in stage.items() if k not in _DRIVER_RESERVED_KEYS}

            stage_x0_override = stage.get("x0", _X0_MISSING)
            if stage_x0_override is _X0_MISSING:
                x_to_pass = current_x
            elif stage_x0_override is None:
                x_to_pass = None
            else:
                x_to_pass = np.asarray(stage_x0_override, dtype=np.float64)

            obj_kwarg = _objective_kwarg_name(solver)
            call_kwargs: dict[str, Any] = {obj_kwarg: f_cached, **stage_kwargs}
            if args and _accepts_kwarg(solver, "args"):
                call_kwargs.setdefault("args", args)
            if x_to_pass is not None and _accepts_kwarg(solver, "x0"):
                call_kwargs["x0"] = x_to_pass
            if _accepts_kwarg(solver, "progressbar"):
                call_kwargs.setdefault("progressbar", progress)
            if _accepts_kwarg(solver, "progress_task"):
                call_kwargs.setdefault("progress_task", stage_tasks[i])
            if verbose and _accepts_kwarg(solver, "verbose"):
                call_kwargs.setdefault("verbose", True)

            try:
                res = solver(**call_kwargs)
            except KeyboardInterrupt:
                res = OptimizeResult(
                    x=_fallback_x(current_x),
                    fun=float("inf"),
                    success=False,
                    message=f"stage {i} ({name}) interrupted by user",
                )
                res._bo_crashed = True
                hard_failure_seen = True
            except Exception as e:
                _log.warning(
                    "sequential_optimize: stage %d (%s) raised %s: %s",
                    i,
                    name,
                    type(e).__name__,
                    e,
                )
                res = OptimizeResult(
                    x=_fallback_x(current_x),
                    fun=float("inf"),
                    success=False,
                    message=f"stage {i} ({name}) raised {type(e).__name__}: {e}",
                )
                res._bo_crashed = True
                hard_failure_seen = True

            res.solver_name = name
            res.stage_index = i

            classification = _classify(res, best_fun, f_cached)
            res.classification = classification
            stage_results.append(res)

            if classification == "usable":
                best_res = res
                best_stage = i
                best_fun = float(res.fun)
                current_x = np.asarray(res.x, dtype=np.float64)
            elif classification == "hard":
                hard_failure_seen = True

            progress.update(
                outer_task,
                advance=1,
                f_value=best_fun if best_fun is not None else float("inf"),
            )

            if classification == "hard" and on_failure == "stop":
                break

    if best_res is None:
        final_x = _fallback_x(current_x)
        final_fun = float("inf")
        if final_x.size > 0:
            try:
                final_fun = float(f_cached.value(final_x))
            except Exception:
                final_fun = float("inf")
        best_res = OptimizeResult(
            x=final_x,
            fun=final_fun,
            success=False,
            message="no stage produced a usable result",
        )

    success = (not hard_failure_seen) and best_stage >= 0
    total_nit = sum(int(getattr(r, "nit", 0) or 0) for r in stage_results)
    total_nfev = sum(int(getattr(r, "nfev", 0) or 0) for r in stage_results)

    msg_parts = [f"{len(stage_results)} stage(s) run"]
    if best_stage >= 0:
        msg_parts.append(f"best from stage {best_stage}")
    else:
        msg_parts.append("no usable stage")
    if hard_failure_seen:
        msg_parts.append("hard failure(s) encountered")

    return SequentialResult(
        best=best_res,
        stage_results=stage_results,
        best_stage=best_stage,
        success=success,
        message="; ".join(msg_parts),
        nit=total_nit,
        nfev=total_nfev,
    )


def _fallback_x(current_x: np.ndarray | None) -> np.ndarray:
    if current_x is None:
        return np.array([])
    return np.asarray(current_x, dtype=np.float64)


__all__ = ["FailurePolicy", "SequentialResult", "sequential_optimize"]
