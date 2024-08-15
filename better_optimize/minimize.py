from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize

from better_optimize.utilities import validate_provided_functions, determine_maxiter
from better_optimize.wrapper import CostFuncWrapper, optimzer_early_stopping_wrapper
from better_optimize.constants import minimize_method

from functools import partial


def minimize(
    method: minimize_method,
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    f_grad: Callable[[np.ndarray], np.ndarray] | None = None,
    f_hess: Callable[[np.ndarray], np.ndarray] | None = None,
    f_hessp: Callable[[np.ndarray], np.ndarray] | None = None,
    progressbar: bool = True,
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
    f_grad: Callable, optional
        The gradient of the objective function
    f_hess: Callable, optional
        The Hessian of the objective function
    f_hessp: Callable, optional
        The Hessian-vector product of the objective function
    method: str
        The optimization method to use
    progressbar: bool
        Whether to display a progress bar
    optimizer_kwargs: dict
        Additional keyword arguments to pass to the optimizer

    Returns
    -------
    optimizer_result: OptimizeResult
        Optimization result

    """
    validate_provided_functions(method, f_grad, f_hess, f_hessp)

    options = optimizer_kwargs.pop("options", {})
    optimizer_kwargs['options'] = options

    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs)

    objective = CostFuncWrapper(
        maxeval=maxiter,
        f=f,
        f_jac=f_grad,
        f_hess=f_hess,
        progressbar=progressbar,
    )

    f_optim = partial(
        sp_minimize,
        fun=objective,
        x0=x0,
        args=args,
        method=method,
        jac=f_grad is not None or None,
        hess=f_hess,
        hessp=f_hessp,
        callback=objective.callback,
        **optimizer_kwargs,
    )

    optimizer_result = optimzer_early_stopping_wrapper(f_optim)
    return optimizer_result