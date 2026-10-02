import logging

from collections.abc import Callable
from functools import partial

import numpy as np

from rich.progress import Progress, TaskID
from scipy.optimize import OptimizeResult
from scipy.optimize import root as sp_root

from better_optimize.configuration import (
    MinimizeConfig,
    OptimizerConfig,
    RootConfig,
    root_config_from_kwargs,
)
from better_optimize.constants import ROOT_METHODS_WITHOUT_CALLBACK, root_method
from better_optimize.utilities import (
    LRUCache1,
    check_f_is_fused_root,
    validate_provided_functions_root,
)
from better_optimize.wrapper import (
    ObjectiveWrapper,
    _compose_callback,
    optimizer_early_stopping_wrapper,
)

_log = logging.getLogger(__name__)


def root(
    f: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    x0: np.ndarray,
    method: root_method | RootConfig,
    jac: Callable[..., np.ndarray] | None = None,
    progressbar: bool | Progress = True,
    progress_task: TaskID | None = None,
    progressbar_update_interval: int = 1,
    verbose: bool = False,
    args: tuple | None = None,
    callback: Callable[..., bool | None] | None = None,
    **optimizer_kwargs,
) -> OptimizeResult:
    """
    Solve a minimization problem using the scipy.optimize.minimize function.

    Parameters
    ----------
    x0: np.ndarray
        The initial values of the parameters to optimize
    f: Callable
        The objective function to minimize
    args: tuple, optional
        Additional arguments to pass to the objective function. Additional arguments are also passed to the gradient
        and Hessian functions, if provided
    jac: Callable, optional
        The gradient of the objective function
    hess: Callable, optional
        The Hessian of the objective function
    hessp: Callable, optional
        The Hessian-vector product of the objective function
    method: str
        The optimization method to use
    progressbar: bool
        Whether to display a progress bar
    progressbar_update_interval: int
        The interval at which the progress bar is updated. If progressbar is False, this parameter is ignored.
    verbose: bool
        If True, warnings about the provided configuration are displayed. These warnings are intended to help users
        understand potential configuration issues that may affect the optimization process, but can be safely ignored.
    callback: Callable, optional
        Function called after each iteration as ``callback(res)``, where ``res`` is an
        ``OptimizeResult`` carrying the current ``res.x``, ``res.fun`` (the residual vector), and
        ``res.nit``. The return value is ignored; raise ``StopOptimization`` to stop early. The
        direct solvers ``hybr`` and ``lm`` ignore the callback and warn. Default None.
    optimizer_kwargs
        Additional keyword arguments to pass to the optimizer

    Returns
    -------
    optimizer_result: OptimizeResult
        Optimization result

    """
    if isinstance(method, OptimizerConfig) and not isinstance(method, RootConfig):
        driver = "minimize" if isinstance(method, MinimizeConfig) else "the driver that runs it"
        raise TypeError(
            f"{type(method).__name__} configures {method.method_name}, which root does not "
            f"run. Pass it to {driver}."
        )

    n_vars = len(x0)
    config = root_config_from_kwargs(method, optimizer_kwargs)

    has_fused_f_and_grad = check_f_is_fused_root(f, x0, args)
    has_jac = validate_provided_functions_root(config, jac, has_fused_f_and_grad, verbose=verbose)

    f_cached = LRUCache1(f, f_returns_list=has_fused_f_and_grad, copy_x=True, dtype=x0.dtype)

    maxiter = config.evaluation_budget(n_vars)

    objective = ObjectiveWrapper(
        maxeval=maxiter,
        f=f_cached.value_and_grad if has_fused_f_and_grad else f_cached.value,
        jac=jac,
        args=args,
        progressbar=progressbar,
        progressbar_update_interval=progressbar_update_interval,
        has_fused_f_and_grad=has_fused_f_and_grad,
        root=True,
        task=progress_task,
    )

    if callback is not None and config.method_name in ROOT_METHODS_WITHOUT_CALLBACK:
        _log.warning(
            f"Method {config.method_name} does not support callbacks; the provided callback "
            "will be ignored."
        )
        callback = None

    # Passing callback=None matches SciPy's default, so the direct methods never trigger SciPy's
    # "does not accept callback" warning.
    root_callback = (
        _compose_callback(objective.callback, objective.callback_result, callback)
        if callback is not None
        else None
    )

    f_optim = partial(
        sp_root,
        fun=objective,
        x0=x0,
        method=config.method_name,
        jac=True if has_jac else None,
        callback=root_callback,
        options=config.optimizer_kwargs(n=n_vars),
    )

    optimizer_result = optimizer_early_stopping_wrapper(f_optim)
    return optimizer_result


__all__ = ["root"]
