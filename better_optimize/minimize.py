from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize

from better_optimize.utilities import validate_provided_functions, determine_maxiter, determine_tolerance, check_f_is_fused
from better_optimize.wrapper import CostFuncWrapper, optimzer_early_stopping_wrapper
from better_optimize.constants import minimize_method

from functools import partial


def minimize(
    f: Callable[..., float | tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: minimize_method,
    jac: Callable[..., np.ndarray] | None = None,
    hess: Callable[..., np.ndarray] | None = None,
    hessp: Callable[..., np.ndarray] | None = None,
    progressbar: bool = True,
    verbose: bool=  True,
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
    verbose: bool
        Whether to display verbose output
    optimizer_kwargs
        Additional keyword arguments to pass to the optimizer

    Returns
    -------
    optimizer_result: OptimizeResult
        Optimization result

    """
    has_fused_f_and_grad = check_f_is_fused(f, x0, args)
    validate_provided_functions(method, jac, hess, hessp ,has_fused_f_and_grad, verbose=verbose)
    print(has_fused_f_and_grad)

    options = optimizer_kwargs.pop("options", {})
    optimizer_kwargs['options'] = options

    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method)
    optimizer_kwargs = determine_tolerance(optimizer_kwargs, method)

    objective = CostFuncWrapper(
        maxeval=maxiter,
        f=f,
        jac=jac,
        hess=hess,
        args=args,
        progressbar=progressbar,
        has_fused_f_and_grad=has_fused_f_and_grad,
    )

    f_optim = partial(
        sp_minimize,
        fun=objective,
        x0=x0,
        method=method,
        jac=True if has_fused_f_and_grad or jac is not None else None,
        hess=hess,
        hessp=hessp,
        callback=objective.callback,
        **optimizer_kwargs,
    )

    optimizer_result = optimzer_early_stopping_wrapper(f_optim)
    return optimizer_result