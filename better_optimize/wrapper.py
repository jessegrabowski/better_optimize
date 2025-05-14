import logging

from collections.abc import Callable
from functools import partial

import numpy as np

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from scipy.optimize import OptimizeResult

from better_optimize.utilities import ToggleableProgress

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
        progressbar: bool | Progress = True,
        progressbar_update_interval: int = 1,
        root=False,
        task: TaskID | None = None,
    ):
        self.n_eval = 0
        self.maxeval = maxeval
        self.args = args if args is not None else ()
        self.f = lambda x: f(x, *self.args)
        self.use_jac = False
        self.use_hess = False
        self.has_fused_f_and_grad = has_fused_f_and_grad
        self.root = root

        self.progress = None
        self.task = task

        self.update_every = progressbar_update_interval
        self.interrupted = False

        if jac is not None or has_fused_f_and_grad:
            self.use_jac = True
            self.f_jac = lambda x: jac(x, *self.args)

        if hess is not None:
            self.use_hess = True
            self.f_hess = lambda x: hess(x, *self.args)

        self.previous_x = None
        if isinstance(progressbar, bool):
            self.progressbar = progressbar
            self.progress = self.initialize_progress_bar()
        else:
            self.progressbar = True
            self.progress = progressbar

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
        if self.root and self.interrupted:
            # Certain optimizers allow callbacks, which can be used to interrupt the optimization process gracefully.
            # Others don't. For those that don't, we have to call the callback ourselves and give the user something
            # back ourselves.
            self.callback()

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

        if self.n_eval == 0 and self.task is None:
            verb = "Minimizing" if not self.root else "Finding Roots"
            self.task = self.progress.add_task(verb, total=self.maxeval, refresh=True, **value_dict)

        if not completed:
            self.progress.update(self.task, advance=self.update_every, **value_dict)
        else:
            self.progress.update(
                self.task, total=self.n_eval, completed=self.n_eval, refresh=True, **value_dict
            )

    def initialize_progress_bar(self):
        # text_column = TextColumn("{task.description}", table_column=Column(ratio=1))
        description = "Minimizing" if not self.root else "Finding Roots"
        bar_column = BarColumn(bar_width=None, table_column=Column(description, ratio=1))
        time_column = TimeElapsedColumn(table_column=Column("Elapsed", ratio=1))
        n_iters = MofNCompleteColumn(table_column=Column("Iteration"))

        objective_name = "Objective" if not self.root else "Residual"
        obj_column = TextColumn(
            "{task.fields[f_value]:0.8f}", table_column=Column(objective_name, ratio=1)
        )

        columns = [bar_column, time_column, n_iters, obj_column]

        if self.use_jac:
            grad_name = "||grad||" if not self.root else "||jac||"
            columns += [
                TextColumn("{task.fields[grad_norm]:0.8f}", table_column=Column(grad_name, ratio=1))
            ]
        if self.use_hess:
            columns += [
                TextColumn(
                    "{task.fields[hess_norm]:0.8f}", table_column=Column("||hess||", ratio=1)
                )
            ]

        return ToggleableProgress(
            *columns, expand=False, disable=not self.progressbar, console=Console(width=100)
        )


def optimizer_early_stopping_wrapper(f_optim: partial):
    objective = f_optim.keywords["fun"]

    with objective.progress:
        try:
            # Do the optimization. This calls either optimize.root or optimize.minimize;
            # all arguments are pre-configured at this point.
            res = f_optim()
            final_value = res.x
        except (KeyboardInterrupt, StopIteration):
            # Teardown the progress bar if necessary, then forward the error
            final_value = objective.previous_x
            res = OptimizeResult(
                x=final_value,
                fun=objective.previous_x,
                success=False,
                message="`StopIteration` or `KeyboardInterrupt` raised -- optimization stopped "
                "prematurely.",
            )
        except Exception as e:
            raise e

        outputs = objective.step(final_value)

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

    return res
