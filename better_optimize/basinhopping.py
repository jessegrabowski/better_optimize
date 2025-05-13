from collections.abc import Callable, Sequence

import numpy as np

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from scipy.optimize import OptimizeResult
from scipy.optimize._basinhopping import (
    AdaptiveStepsize,
    BasinHoppingRunner,
    Metropolis,
    MinimizerWrapper,
    RandomDisplacement,
    Storage,
    check_random_state,
)

from better_optimize.minimize import minimize
from better_optimize.utilities import ToggleableProgress, check_f_is_fused


def initialize_progress_bar(progressbar, use_jac=True, use_hess=False):
    description = "Process"
    name_column = TextColumn("{task.fields[name]}", table_column=Column(description, ratio=2))
    bar_column = BarColumn(bar_width=None, table_column=Column("", ratio=2))
    time_column = TimeElapsedColumn(table_column=Column("Elapsed", ratio=1))
    n_iters = MofNCompleteColumn(table_column=Column("Iteration"))

    step_size = TextColumn("{task.fields[step_size]:0.3f}", table_column=Column("Step", ratio=1))
    target_accept = TextColumn(
        "{task.fields[target_accept]:0.3f}", table_column=Column("Target", ratio=1)
    )

    objective_name = "Objective"
    obj_column = TextColumn(
        "{task.fields[f_value]:0.5f}", table_column=Column(objective_name, ratio=1)
    )

    columns = [name_column, bar_column, time_column, n_iters, step_size, target_accept, obj_column]

    if use_jac:
        columns += [
            TextColumn("{task.fields[grad_norm]:0.5f}", table_column=Column("||grad||", ratio=1))
        ]
    if use_hess:
        columns += [
            TextColumn("{task.fields[hess_norm]:0.8f}", table_column=Column("||hess||", ratio=1))
        ]

    return ToggleableProgress(
        *columns, expand=False, disable=not progressbar, console=Console(width=100), transient=True
    )


class AllowFailureStorage(Storage):
    """
    Subclass of Storage that allows the minimizer to fail, but still updates the global minimum
    if the new point is better than the current global minimum.
    """

    def update(self, minres):
        if minres.fun < self.minres.fun:
            self._add(minres)
            return True
        else:
            return False


class AllowFailureBasinHoppingRunner(BasinHoppingRunner):
    """
    A subclass of BasinHoppingRunner that allows the minimizer to fail, but still updates the
    global minimum if the new point is better than the current global minimum.
    """

    def __init__(
        self, x0, minimizer, step_taking, accept_tests, accept_on_minimizer_fail=False, disp=False
    ):
        super().__init__(x0, minimizer, step_taking, accept_tests, disp=disp)

        if accept_on_minimizer_fail:
            self.storage = AllowFailureStorage(self.storage.minres)


def basinhopping(
    func: Callable[..., float | np.ndarray],
    x0: Sequence[float],
    niter: int = 100,
    T: float = 1.0,
    stepsize: float = 0.5,
    minimizer_kwargs: dict | None = None,
    take_step: Callable | None = None,
    accept_test: Callable | None = None,
    accept_on_minimizer_fail: bool = False,
    callback: Callable | None = None,
    interval: int = 50,
    progressbar: bool = True,
    verbose: bool = False,
    niter_success: int | None = None,
    rng: int | float | np.random.Generator | None = None,
    *,
    target_accept_rate: float = 0.5,
    stepwise_factor: float = 0.9,
) -> OptimizeResult:
    """
    Perform a global optimization using the basin-hopping algorithm. For details, see scipy.optimize.basinhopping.

    Parameters
    ----------
    func: Callable
        Scalar function to optimize
    x0: np.ndarray
        Initial values
    niter: int
        Number of basin-hopping iterations
    T: float
        Temperature parameter for the accept/reject criterion
    stepsize: float
        Initial step size for use in the random displacement
    minimizer_kwargs: dict, optional
        Extra keyword arguments to pass to the minimizer
    take_step: Callable, optional
        Custom step-taking routine
    accept_test: Callable, optional
        Custom accept/reject routine
    accept_on_minimizer_fail: bool
        Accept the new point even if the minimizer fails. Will also update the global minimum if the point is
        better than the current global minimum, even if the minimizer fails.
    callback: Callable, optional
        User-supplied function to call after each iteration
    interval: int
        Interval for how often to update the step size
    progressbar: bool
        Whether to display a progress bar
    verbose: bool
        Whether to print verbose output. Ignored if progressbar is True.
    niter_success: int, optional
        Stop if the global minimum candidate remains the same for this many iterations
    rng: np.random.RandomState, optional
        Random number generator
    target_accept_rate: float
        Target acceptance rate for the adaptive step size
    stepwise_factor: float
        Factor to adjust the step size

    Returns
    -------
    res: OptimizerResult
        Result of optimization
    """
    if target_accept_rate <= 0.0 or target_accept_rate >= 1.0:
        raise ValueError("target_accept_rate has to be in range (0, 1)")
    if stepwise_factor <= 0.0 or stepwise_factor >= 1.0:
        raise ValueError("stepwise_factor has to be in range (0, 1)")

    x0 = np.array(x0)

    # set up the np.random generator
    rng = check_random_state(rng)

    # set up minimizer
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()

    has_fused_f_and_grad = check_f_is_fused(func, x0, minimizer_kwargs.get("args", ()))
    if has_fused_f_and_grad:
        minimizer_kwargs.pop("jac", None)

    use_jac = has_fused_f_and_grad or minimizer_kwargs.get("jac", False)
    progress = initialize_progress_bar(progressbar, use_jac=use_jac, use_hess=False)

    if progressbar:
        verbose = False

    bh_task = progress.add_task(
        description="Basinhopping",
        name="Basinhopping",
        total=niter,
        target_accept=target_accept_rate,
        step_size=stepsize,
        f_value=0.0,
        grad_norm=0.0,
        hess_norm=0.0,
    )

    minimize_task = progress.add_task(
        description="Minimize",
        name="Minimize",
        step_size=0.0,
        target_accept=0.0,
        f_value=0.0,
        grad_norm=0.0,
        hess_norm=0.0,
    )

    wrapped_minimizer = MinimizerWrapper(
        minimize,
        func,
        progressbar=progress,
        progress_task=minimize_task,
        verbose=verbose,
        **minimizer_kwargs,
    )

    # set up step-taking algorithm
    if take_step is not None:
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        # if take_step.stepsize exists then use AdaptiveStepsize to control
        # take_step.stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(
                take_step,
                interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=verbose,
            )
        else:
            take_step_wrapped = take_step
    else:
        # use default
        displace = RandomDisplacement(stepsize=stepsize, rng=rng)
        take_step_wrapped = AdaptiveStepsize(
            displace,
            interval=interval,
            accept_rate=target_accept_rate,
            factor=stepwise_factor,
            verbose=verbose,
        )

    # set up accept tests
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests = [accept_test]

    # use default
    metropolis = Metropolis(T, rng=rng)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = niter + 2

    bh = AllowFailureBasinHoppingRunner(
        x0,
        wrapped_minimizer,
        take_step_wrapped,
        accept_tests,
        accept_on_minimizer_fail=accept_on_minimizer_fail,
        disp=verbose,
    )

    # The wrapped minimizer is called once during construction of
    # BasinHoppingRunner, so run the callback
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # start main iteration loop
    count, i = 0, 0
    message = ["requested number of basinhopping iterations completed" " successfully"]

    grad_norm_at_min = 0.0
    if use_jac:
        _, grad_val = func(bh.x)
        grad_norm_at_min = np.linalg.norm(grad_val)

    with progress:
        for i in range(niter):
            if progressbar:
                progress.reset(minimize_task, visible=True)
            new_global_min = bh.one_cycle()
            if new_global_min and progressbar:
                grad_norm_at_min = progress.tasks[minimize_task].fields["grad_norm"]

            progress.update(
                bh_task,
                advance=1,
                step_size=take_step_wrapped.takestep.stepsize,
                target_accept=take_step_wrapped.target_accept_rate,
                f_value=bh.storage.get_lowest().fun,
                grad_norm=grad_norm_at_min,
            )

            if callable(callback):
                # should we pass a copy of x?
                val = callback(bh.xtrial, bh.energy_trial, bh.accept)
                if val is not None:
                    if val:
                        message = ["callback function requested stop early by" "returning True"]
                        break

            count += 1
            if new_global_min:
                count = 0
            elif count > niter_success:
                message = ["success condition satisfied"]
                break

    # Hack -- jupyter wants transient = True, or else a bunch of newlines get inserted. But when transient = True,
    # the progress disappears in the terminal. Setting False now makes both work.
    if hasattr(progress, "live") and progress.live.transient:
        progress.live.transient = False

    with progress:
        progress.update(
            bh_task,
            total=i + 1,
            completed=i + 1,
            refresh=True,
            f_value=bh.storage.get_lowest().fun,
            grad_norm=grad_norm_at_min,
        )
    # prepare return object
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    res.success = res.lowest_optimization_result.success

    return res


__all__ = ["basinhopping"]
