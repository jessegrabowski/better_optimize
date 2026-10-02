from collections.abc import Callable
from functools import partial

import numpy as np

from rich.progress import Progress, TaskID
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize
from scipy.sparse.linalg import LinearOperator

from better_optimize.configuration import (
    OptimizerConfig,
    SolverProblem,
    config_from_kwargs,
)
from better_optimize.constants import minimize_method
from better_optimize.utilities import (
    LRUCache1,
    check_f_is_fused_minimize,
    validate_provided_functions_minimize,
)
from better_optimize.wrapper import (
    ObjectiveWrapper,
    _compose_callback,
    optimizer_early_stopping_wrapper,
)


def minimize(
    f: Callable[..., float | tuple[float, np.ndarray]],
    x0: np.ndarray,
    method: minimize_method | OptimizerConfig,
    jac: Callable[..., np.ndarray] | None = None,
    hess: Callable[..., np.ndarray | LinearOperator] | None = None,
    hessp: Callable[..., np.ndarray] | None = None,
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
    method: str or OptimizerConfig
        The optimization method to use, either by name or as a configuration carrying its
        options. A configuration cannot be combined with options passed as keywords.
    progressbar: bool
        Whether to display a progress bar
    progressbar_update_interval: int
        The interval at which the progress bar is updated. If progressbar is False, this parameter is ignored.
    verbose: bool
        If True, warnings about the provided configuration are displayed. These warnings are intended to help users
        understand potential configuration issues that may affect the optimization process, but can be safely ignored.
    callback: Callable, optional
        Function called after each iteration as ``callback(res)``, where ``res`` is an
        ``OptimizeResult`` carrying the current ``res.x``, ``res.fun``, and ``res.nit`` (plus
        ``res.jac`` when a gradient is available). The return value is ignored; raise
        ``StopOptimization`` to stop early. Default None.
    optimizer_kwargs
        Options for the chosen method, plus ``bounds`` and ``constraints``, which describe
        the problem and reach scipy directly. An option the method does not accept raises
        ``TypeError``.

    Returns
    -------
    optimizer_result: OptimizeResult
        Optimization result

    """
    n_vars = len(x0)
    config, solver_kwargs = config_from_kwargs(method, optimizer_kwargs)

    solver = config.solver_function()
    if solver is not None:
        return solver(
            **config.build_solver_kwargs(
                SolverProblem(
                    f=f,
                    x0=x0,
                    jac=jac,
                    hess=hess,
                    hessp=hessp,
                    args=() if args is None else args,
                    callback=callback,
                    progressbar=progressbar,
                    progress_task=progress_task,
                    progressbar_update_interval=progressbar_update_interval,
                    verbose=verbose,
                    solver_kwargs=solver_kwargs,
                )
            )
        )

    has_fused_f_and_grad, has_fused_f_grad_hess = check_f_is_fused_minimize(f, x0, args)

    use_grad, use_hess, use_hessp = validate_provided_functions_minimize(
        config, jac, hess, hessp, has_fused_f_and_grad, has_fused_f_grad_hess, verbose=verbose
    )

    f_returns_list = has_fused_f_and_grad or has_fused_f_grad_hess
    f_cached = LRUCache1(f, f_returns_list=f_returns_list, copy_x=False, dtype=x0.dtype)

    if has_fused_f_grad_hess:
        hess = f_cached.hess

    # Test hessian function -- if it returns a LinearOperator, it can't be used inside the wrapper
    args = () if args is None else args
    use_hess = use_hess and not isinstance(hess(x0, *args), LinearOperator)

    objective = ObjectiveWrapper(
        maxeval=config.evaluation_budget(n_vars),
        f=f_cached.value_and_grad if has_fused_f_and_grad else f_cached.value,
        jac=jac,
        hess=hess if use_hess else None,
        hessp=hessp if use_hessp else None,
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
        method=config.method_name,
        jac=True if has_fused_f_and_grad or jac is not None else None,
        hess=None if not use_hess else lambda x: hess(x, *args),
        hessp=None if not use_hessp else lambda x, p: hessp(x, p, *args),
        callback=_compose_callback(objective.callback, objective.callback_result, callback),
        options=config.optimizer_kwargs(n=n_vars),
        **solver_kwargs,
    )

    optimizer_result = optimizer_early_stopping_wrapper(f_optim)
    return optimizer_result


__all__ = ["minimize"]
