from collections.abc import Callable
from functools import partial

import numpy as np

from scipy.optimize import OptimizeResult
from scipy.optimize import root as sp_root

from better_optimize.constants import root_method
from better_optimize.utilities import (
    check_f_is_fused,
    determine_maxiter,
    determine_tolerance,
    kwargs_to_jac_options,
    kwargs_to_options,
    validate_provided_functions_root,
)
from better_optimize.wrapper import ObjectiveWrapper, optimzer_early_stopping_wrapper


def root(
    f: Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]],
    x0: np.ndarray,
    method: root_method,
    jac: Callable[..., np.ndarray] | None = None,
    progressbar: bool = True,
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
    has_fused_f_and_grad = check_f_is_fused(f, x0, args)
    validate_provided_functions_root(
        method, jac, has_fused_f_and_grad, has_fused_f_and_grad, verbose=verbose
    )

    options = optimizer_kwargs.pop("options", {})
    optimizer_kwargs["options"] = options
    optimizer_kwargs = kwargs_to_options(optimizer_kwargs, method)
    optimizer_kwargs = kwargs_to_jac_options(optimizer_kwargs, method)

    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method, len(x0))
    optimizer_kwargs = determine_tolerance(optimizer_kwargs, method)

    objective = ObjectiveWrapper(
        maxeval=maxiter,
        f=f,
        jac=jac,
        args=args,
        progressbar=progressbar,
        has_fused_f_and_grad=has_fused_f_and_grad,
        root=True,
    )

    f_optim = partial(
        sp_root,
        fun=objective,
        x0=x0,
        method=method,
        jac=True if has_fused_f_and_grad or jac is not None else None,
        **optimizer_kwargs,
    )

    optimizer_result = optimzer_early_stopping_wrapper(f_optim)
    return optimizer_result
