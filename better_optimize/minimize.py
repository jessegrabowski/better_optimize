from collections.abc import Callable
from functools import partial

import numpy as np

from rich.progress import Progress, TaskID
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize
from scipy.sparse.linalg import LinearOperator

from better_optimize.constants import minimize_method
from better_optimize.utilities import (
    LRUCache1,
    check_f_is_fused_minimize,
    determine_maxiter,
    determine_tolerance,
    kwargs_to_options,
    validate_provided_functions_minimize,
)
from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper


def minimize(
    f: Callable[..., float | tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: minimize_method,
    jac: Callable[..., np.ndarray] | None = None,
    hess: Callable[..., np.ndarray | LinearOperator] | None = None,
    hessp: Callable[..., np.ndarray] | None = None,
    progressbar: bool | Progress = True,
    progress_task: TaskID | None = None,
    progressbar_update_interval: int = 1,
    verbose: bool = False,
    args: tuple | None = None,
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
    optimizer_kwargs
        Additional keyword arguments to pass to the optimizer

    Returns
    -------
    optimizer_result: OptimizeResult
        Optimization result

    """
    has_fused_f_and_grad, has_fused_f_grad_hess = check_f_is_fused_minimize(f, x0, args)

    use_grad, use_hess, use_hessp = validate_provided_functions_minimize(
        method, jac, hess, hessp, has_fused_f_and_grad, has_fused_f_grad_hess, verbose=verbose
    )

    f_returns_list = has_fused_f_grad_hess or has_fused_f_grad_hess
    f_cached = LRUCache1(f, f_returns_list=f_returns_list, copy_x=False, dtype=x0.dtype)

    if has_fused_f_grad_hess:
        hess = f_cached.hess

    options = optimizer_kwargs.pop("options", {})
    optimizer_kwargs["options"] = options

    optimizer_kwargs = kwargs_to_options(optimizer_kwargs, method)
    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method, len(x0))
    optimizer_kwargs = determine_tolerance(optimizer_kwargs, method)

    # Test hessian function -- if it returns a LinearOperator, it can't be used inside the wrapper
    args = () if args is None else args
    use_hess = use_hess and not isinstance(hess(x0, *args), LinearOperator)

    objective = ObjectiveWrapper(
        maxeval=maxiter,
        f=f_cached.value_and_grad if has_fused_f_and_grad else f_cached.value,
        jac=jac,
        hess=hess if use_hess else None,
        args=args,
        progressbar=progressbar,
        progressbar_update_interval=progressbar_update_interval,
        has_fused_f_and_grad=has_fused_f_and_grad,
        root=False,
        task=progress_task,
    )

    f_optim = partial(
        sp_minimize,
        fun=objective,
        x0=x0,
        method=method,
        jac=True if has_fused_f_and_grad or jac is not None else None,
        hess=None if not use_hess else lambda x: hess(x, *args),
        hessp=None if not use_hessp else lambda x, p: hessp(x, p, *args),
        callback=objective.callback,
        **optimizer_kwargs,
    )

    optimizer_result = optimizer_early_stopping_wrapper(f_optim)
    return optimizer_result


__all__ = ["minimize"]
