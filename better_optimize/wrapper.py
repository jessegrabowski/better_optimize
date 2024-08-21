import logging

from collections.abc import Callable

import numpy as np

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

_log = logging.getLogger(__name__)


class ObjectiveWrapper:
    def __init__(
        self,
        f: Callable[[np.ndarray | float, ...], float | np.ndarray | tuple[float, np.ndarray]],
        jac: Callable[[np.ndarray | float, ...], np.ndarray] | None = None,
        hess: Callable[[np.ndarray | float, ...], np.ndarray] | None = None,
        has_fused_f_and_grad: bool = False,
        args: tuple | None = None,
        maxeval: int = 5000,
        progressbar: bool = True,
        update_every: int = 10,
    ):
        self.n_eval = 0
        self.maxeval = maxeval
        self.args = args if args is not None else ()
        self.f = lambda x: f(x, *self.args)
        self.use_jac = False
        self.use_hess = False
        self.has_fused_f_and_grad = has_fused_f_and_grad

        self.progress = None
        self.task = None

        self.update_every = update_every
        self.interrupted = False
        self.desc = "f = {task.fields[f_value]:,.5g}"

        if jac is not None or has_fused_f_and_grad:
            self.desc += ", ||grad|| = {task.fields[grad_norm]:,.5g}"
            self.use_jac = True
            self.f_jac = lambda x: jac(x, *self.args)

        if hess is not None:
            self.desc += ", ||hess|| = {task.fields[hess_norm]:,.5g}"
            self.use_hess = True
            self.f_hess = lambda x: hess(x, *self.args)

        self.previous_x = None
        self.progressbar = progressbar
        if progressbar:
            self.progress = self.initialize_progress_bar()
            self.progress.start()

    def step(self, x):
        grad = None
        hess = None
        if self.has_fused_f_and_grad:
            value, grad = self.f(x)
        else:
            value = self.f(x)

        if self.use_jac and not self.has_fused_f_and_grad:
            grad = self.f_jac(x)
        if self.use_hess:
            hess = self.f_hess(x)

        if np.all(np.isfinite(x)):
            self.previous_x = x

        if self.n_eval % self.update_every == 0:
            self.update_progressbar(value, grad, hess)

        if self.n_eval > self.maxeval:
            self.update_progressbar(value, grad, hess)
            self.interrupted = True

        self.n_eval += 1

        if self.use_hess:
            return value, grad, hess
        elif self.use_jac:
            return value, grad
        else:
            return value

    def __call__(self, x):
        try:
            return self.step(x)
        except (KeyboardInterrupt, StopIteration):
            self.interrupted = True
            return self.step(x)

    def callback(self, *args):
        if self.interrupted:
            raise StopIteration

    def update_progressbar(
        self, value: float, grad: np.float64 = None, hess: np.float64 = None, completed=False
    ) -> None:
        if not self.progressbar:
            return

        if isinstance(value, np.ndarray):
            value = (value**2).sum()
        elif isinstance(value, list | tuple):
            value = sum([x**2 for x in value])

        value_dict = {"f_value": value}
        if grad is not None:
            grad_norm = np.linalg.norm(grad)
            value_dict["grad_norm"] = grad_norm
        if hess is not None:
            hess_norm = np.linalg.norm(hess)
            value_dict["hess_norm"] = hess_norm

        if self.n_eval == 0:
            self.task = self.progress.add_task(
                "Optimizing", total=self.maxeval, refresh=True, **value_dict
            )

        if not completed:
            self.progress.update(self.task, advance=self.update_every, **value_dict)
        else:
            self.progress.update(
                self.task, total=self.n_eval, completed=self.n_eval, refresh=True, **value_dict
            )
            self.progress.stop()

    def initialize_progress_bar(self):
        text_column = TextColumn("{task.description}", table_column=Column(ratio=1))
        bar_column = BarColumn(bar_width=None, table_column=Column(ratio=2))
        time_column = TimeElapsedColumn()
        m_of_n = MofNCompleteColumn()
        spinner = SpinnerColumn()

        stat_column = TextColumn(self.desc, table_column=Column(ratio=1))

        return Progress(
            text_column, spinner, bar_column, time_column, m_of_n, stat_column, expand=False
        )


def optimzer_early_stopping_wrapper(f_optim):
    objective = f_optim.keywords["fun"]
    progressbar = objective.progressbar

    try:
        # Do the optimization. This calls either optimize.root or optimize.minimize;  all arguments are pre-configured
        # at this point.
        res = f_optim()
    except Exception as e:
        # Teardown the progress bar if necessary, then forward the error
        if progressbar:
            objective.progress.stop()
        raise e

    x_final = res.x

    if progressbar:
        # Evaluate the objective function one last time to get the final value, gradient, and hessian
        # and update the progress bar to show the true final state
        outputs = objective.step(x_final)
        if not objective.use_jac and not objective.use_hess:
            value = outputs
            grad = None
            hess = None
        elif objective.use_jac and not objective.use_hess:
            value, grad = outputs
            hess = None
        else:
            value, grad, hess = outputs

        objective.update_progressbar(value, grad, hess, completed=True)
        objective.progress.stop()

    return res
