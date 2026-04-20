import contextlib
import logging

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from rich.progress import TaskID
from scipy.optimize import OptimizeResult
from scipy.optimize import differential_evolution as sp_differential_evolution

from better_optimize.constants import DE_INIT_OPTIONS, DE_STRATEGY_OPTIONS
from better_optimize.utilities import LRUCache1, check_f_is_fused_minimize
from better_optimize.wrapper import build_progress_bar

_log = logging.getLogger(__name__)


def _nullcontext():
    return contextlib.nullcontext()


def _bounds_to_array(bounds) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"bounds must be a sequence of (low, high) pairs or an (n, 2) array; "
            f"got shape {arr.shape}"
        )
    if np.any(arr[:, 0] > arr[:, 1]):
        raise ValueError("bounds low > high for at least one dimension")
    return arr


def _sample_x_from_bounds(bounds_arr: np.ndarray) -> np.ndarray:
    """Generate one midpoint within bounds for fused-function sniffing."""
    return 0.5 * (bounds_arr[:, 0] + bounds_arr[:, 1])


def _default_maxiter(d: int) -> int:
    return max(1000, 200 * d)


def differential_evolution(
    f: Callable,
    bounds: Sequence[tuple[float, float]] | np.ndarray,
    args: tuple = (),
    strategy: str | Callable = "best1bin",
    maxiter: int | None = None,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    callback: Callable | None = None,
    init: str | np.ndarray = "sobol",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int | Callable = 1,
    constraints: tuple = (),
    x0: Sequence[float] | np.ndarray | None = None,
    integrality: Sequence[bool] | np.ndarray | None = None,
    vectorized: bool = False,
    progressbar: bool = True,
    progress_task: TaskID | None = None,
    verbose: bool = False,
    rng: int | np.random.Generator | None = None,
) -> OptimizeResult:
    """Differential evolution with progress bar, graceful failure, and fused-function support.

    See scipy's :func:`scipy.optimize.differential_evolution` documentation for parameter
    semantics. Polish customization (method choice, tolerance) is not exposed here by design —
    chain a dedicated ``minimize`` stage via :func:`sequential_optimize` instead.

    Parameters
    ----------
    f : Callable
        Scalar objective, or fused ``(f, grad)`` / ``(f, grad, hess)``.
    bounds : sequence of (low, high) or (d, 2) array
        Search region per dimension. Required.
    progressbar : bool or ToggleableProgress
        If True, show a live rich progress bar with best-so-far each generation. If a
        ``ToggleableProgress`` instance is passed (e.g. from ``sequential_optimize``), reuse
        it instead of creating a new one.
    progress_task : TaskID, optional
        When ``sequential_optimize`` pre-registers a task on the shared progress and passes
        the id here, that task is updated instead of creating a new one. Ignored for
        standalone calls.
    rng : int, numpy.random.Generator, or None
        Seed / generator for reproducibility. Forwarded to scipy as the ``rng`` kwarg.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The usual scipy result. On internal failure, ``success=False`` and ``fun=np.inf``.
    """
    bounds_arr = _bounds_to_array(bounds)
    d = bounds_arr.shape[0]

    if isinstance(strategy, str) and strategy not in DE_STRATEGY_OPTIONS:
        raise ValueError(
            f"strategy must be one of {DE_STRATEGY_OPTIONS} or a callable; got {strategy!r}"
        )
    if isinstance(init, str) and init not in DE_INIT_OPTIONS:
        raise ValueError(f"init must be one of {DE_INIT_OPTIONS} or an array; got {init!r}")

    if maxiter is None:
        maxiter = _default_maxiter(d)

    sample_x = (
        np.asarray(x0, dtype=np.float64) if x0 is not None else _sample_x_from_bounds(bounds_arr)
    )

    has_fused_f_and_grad, has_fused_f_grad_hess = check_f_is_fused_minimize(f, sample_x, args or ())
    f_returns_list = has_fused_f_and_grad or has_fused_f_grad_hess
    f_cached = LRUCache1(
        f,
        f_returns_list=f_returns_list,
        copy_x=False,
        dtype=sample_x.dtype.name if sample_x.dtype != object else None,
    )

    owns_progress = not isinstance(progressbar, object.__class__) and isinstance(progressbar, bool)
    if isinstance(progressbar, bool):
        progress = build_progress_bar(
            description="Differential Evolution",
            progressbar=progressbar,
            root=False,
            use_jac=False,
            use_hess=False,
        )
        owns_progress = True
    else:
        progress = progressbar
        owns_progress = False

    if progress_task is None:
        de_task = progress.add_task(
            description="Differential Evolution",
            total=maxiter,
            f_value=float("inf"),
            grad_norm=0.0,
            hess_norm=0.0,
        )
    else:
        de_task = progress_task
        progress.update(de_task, total=maxiter, f_value=float("inf"))

    iter_counter = [0]
    best_fun_seen = [float("inf")]

    def progress_callback(intermediate_result):
        """Drive progress bar + forward to user's callback.

        scipy passes an OptimizeResult-like object since scipy 1.13 (or (xk, convergence)
        in older versions). Support both.
        """
        if hasattr(intermediate_result, "x"):
            xk = np.asarray(intermediate_result.x, dtype=np.float64)
            convergence = float(getattr(intermediate_result, "convergence", 0.0))
        else:  # legacy tuple-style
            xk = np.asarray(intermediate_result, dtype=np.float64)
            convergence = 0.0

        try:
            f_val = float(f_cached.value(xk))
        except Exception:
            f_val = best_fun_seen[0]

        if f_val < best_fun_seen[0]:
            best_fun_seen[0] = f_val

        iter_counter[0] += 1
        progress.update(de_task, advance=1, f_value=best_fun_seen[0])

        if callback is not None:
            user_stop = callback(xk, convergence=convergence)
            if user_stop:
                raise StopIteration("user callback returned True")
        return False

    scipy_kwargs: dict[str, Any] = {
        "args": args,
        "strategy": strategy,
        "maxiter": maxiter,
        "popsize": popsize,
        "tol": tol,
        "mutation": mutation,
        "recombination": recombination,
        "rng": rng,
        "callback": progress_callback,
        "disp": False,
        "polish": False,
        "init": init,
        "atol": atol,
        "updating": updating,
        "workers": workers,
        "constraints": constraints,
        "integrality": integrality,
        "vectorized": vectorized,
    }
    if x0 is not None:
        scipy_kwargs["x0"] = np.asarray(x0, dtype=np.float64)

    fallback_x = np.asarray(x0, dtype=np.float64) if x0 is not None else sample_x

    progress_ctx = progress if owns_progress else _nullcontext()
    with progress_ctx:
        try:
            res = sp_differential_evolution(f_cached.value, bounds_arr, **scipy_kwargs)
        except (KeyboardInterrupt, StopIteration) as e:
            _log.warning("differential_evolution interrupted: %s", e)
            res = OptimizeResult(
                x=fallback_x,
                fun=best_fun_seen[0],
                success=False,
                message=f"interrupted: {type(e).__name__}",
                nit=iter_counter[0],
                nfev=f_cached.cache_misses,
            )
        except (FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as e:
            _log.warning("differential_evolution raised %s: %s", type(e).__name__, e)
            res = OptimizeResult(
                x=fallback_x,
                fun=float("inf"),
                success=False,
                message=f"differential_evolution raised {type(e).__name__}: {e}",
                nit=iter_counter[0],
                nfev=f_cached.cache_misses,
            )
        else:
            progress.update(de_task, completed=maxiter, f_value=float(res.fun), refresh=True)

    if verbose:
        _log.info(
            "differential_evolution finished: fun=%.6e, nit=%d, success=%s",
            float(getattr(res, "fun", float("inf"))),
            int(getattr(res, "nit", 0) or 0),
            bool(getattr(res, "success", False)),
        )

    return res


__all__ = ["differential_evolution"]
