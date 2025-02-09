import logging

from collections.abc import Callable, Iterable
from copy import deepcopy

import numpy as np

from rich.box import SIMPLE_HEAD
from rich.progress import Progress, Task
from rich.table import Column, Table

from better_optimize.constants import (
    MINIMIZE_MODE_KWARGS,
    ROOT_MODE_KWARGS,
    TOLERANCES,
    minimize_method,
    root_method,
)

_log = logging.getLogger(__name__)


class ToggleableProgress(Progress):
    """
    Copied from PyMC: https://github.com/pymc-devs/pymc/blob/5352798ee0d36ed566e651466e54634b1b9a06c8/pymc/util.py#L545
    A child of Progress that allows to disable progress bars and its container.

    The implementation simply checks an `is_enabled` flag and generates the progress bar only if
    it's `True`.
    """

    def __init__(self, *args, **kwargs):
        self.is_enabled = kwargs.get("disable", None) is not True
        if self.is_enabled:
            super().__init__(*args, **kwargs)

    def __enter__(self):
        """Enter the context manager."""
        if self.is_enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.is_enabled:
            super().__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, *args, **kwargs):
        if self.is_enabled:
            return super().add_task(*args, **kwargs)
        return None

    def advance(self, task_id, advance=1) -> None:
        if self.is_enabled:
            super().advance(task_id, advance)
        return None

    def update(
        self,
        task_id,
        *,
        total=None,
        completed=None,
        advance=None,
        description=None,
        visible=None,
        refresh=False,
        **fields,
    ):
        if self.is_enabled:
            super().update(
                task_id,
                total=total,
                completed=completed,
                advance=advance,
                description=description,
                visible=visible,
                refresh=refresh,
                **fields,
            )
        return None

    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        """Get a table to render the Progress display.

        Unlike the parent method, this one returns a full table (not a grid), allowing for column headings.

        Parameters
        ----------
        tasks: Iterable[Task]
            An iterable of Task instances, one per row of the table.

        Returns
        -------
        table: Table
            A table instance.
        """

        def call_column(column, task):
            # Subclass rich.BarColumn and add a callback method to dynamically update the display
            if hasattr(column, "callbacks"):
                column.callbacks(task)

            return column(task)

        table_columns = (
            (
                Column(no_wrap=True)
                if isinstance(_column, str)
                else _column.get_table_column().copy()
            )
            for _column in self.columns
        )

        table = Table(
            *table_columns,
            padding=(0, 1),
            expand=self.expand,
            show_header=True,
            show_edge=True,
            box=SIMPLE_HEAD,
        )

        for task in tasks:
            if task.visible:
                table.add_row(
                    *(
                        (
                            column.format(task=task)
                            if isinstance(column, str)
                            else call_column(column, task)
                        )
                        for column in self.columns
                    )
                )

        return table


def get_option_kwargs(method: minimize_method | root_method):
    if method in MINIMIZE_MODE_KWARGS.keys():
        options_kwargs = MINIMIZE_MODE_KWARGS[method]
    elif method in ROOT_MODE_KWARGS.keys():
        options_kwargs = ROOT_MODE_KWARGS[method]
    else:
        raise ValueError(f"Unknown method: {method}")

    return options_kwargs


def validate_provided_functions_minimize(
    method: minimize_method,
    f_grad: Callable[[np.ndarray], np.ndarray] | None,
    f_hess: Callable[[np.ndarray], np.ndarray] | None,
    f_hessp: Callable[[np.ndarray], np.ndarray] | None,
    has_fused_f_and_grad: bool,
    verbose=True,
) -> None:
    has_grad, has_hess, has_hessp = map(lambda f: f is not None, [f_grad, f_hess, f_hessp])
    uses_grad, uses_hess, uses_hessp, *_ = MINIMIZE_MODE_KWARGS[method].values()

    if has_fused_f_and_grad and has_grad:
        _log.warning(
            "Objective function returns a tuple (interpreted as (loss, gradient), but a gradient function was "
            "also provided. The gradient function will be ignored."
        )

    elif has_fused_f_and_grad:
        has_grad = True

    if method not in MINIMIZE_MODE_KWARGS:
        raise ValueError(
            f"Method {method} not recognized. Must be one of {list(MINIMIZE_MODE_KWARGS.keys())}"
        )

    if has_hess and has_hessp:
        raise ValueError(
            "Cannot ask for Hessian and Hessian-vector product at the same time. For all algorithms "
            "except trust-exact and dogleg, use_hessp is preferred."
        )

    if has_grad and not uses_grad and verbose:
        _log.warning(
            f"Gradient provided but not used by method {method}. Gradients will still be evaluated at each "
            f"optimzer step and the norm will be reported as a diagnositc. For large problems, this may be "
            f"computationally intensive."
        )

    if (has_hess and not uses_hess) or (has_hessp and not uses_hessp) and verbose:
        _log.warning(
            f"Hessian provided but not used by method {method}. The full hessian will still be evaluated at "
            f"each optimzer step and the norm will be reported as a diagnositc. For large problems, this may "
            f"be computationally intensive."
        )

    if has_hess and not has_hessp and uses_hessp and uses_hess and verbose:
        _log.warning(
            f"You provided a function to compute the full hessian, but method {method} allows the use of a "
            f"hessian-vector product instead. Consider passing hessp instead -- this may be significantly "
            f"more efficient."
        )


def validate_provided_functions_root(
    method: root_method, f, jac, has_fused_f_and_grad: bool, verbose: bool = True
):
    has_jac = jac is not None
    info_dict = get_option_kwargs(method)
    uses_jac = info_dict["uses_jac"]

    if has_fused_f_and_grad and has_jac and verbose:
        _log.warning(
            "Objective function returns a tuple (interpreted as (loss, gradient), but a gradient function was "
            "also provided. The gradient function will be ignored."
        )

    elif has_fused_f_and_grad:
        has_jac = True

    if has_jac and not uses_jac and verbose:
        _log.warning(
            f"Gradient provided but not used by method {method}. Gradients will still be evaluated at each "
            f"optimzer step and the norm will be reported as a diagnositc. For large problems, this may be "
            f"computationally intensive."
        )


def check_f_is_fused(f, x0, args):
    args = () if args is None else args
    output = f(x0, *args)
    if not isinstance(output, tuple | list):
        return False

    # If the output is a tuple, it should be length 2 (returning the value and the grad).If not, something is wrong
    if len(output) != 2:
        raise ValueError(
            "Objective function should return either a scalar loss or a two-tuple of (loss, gradient)"
        )
    return True


def determine_maxiter(
    optimizer_kwargs: dict, method: minimize_method | root_method, n_vars
) -> tuple[int, dict]:
    MAXITER_KWARGS = ["maxiter", "maxfun", "maxfev"]
    method_info = get_option_kwargs(method)
    maxiter_kwargs = [x for x in method_info["valid_options"] if x in MAXITER_KWARGS]
    maxiter_possibilities = [
        optimizer_kwargs.pop("maxiter", None),
        *(optimizer_kwargs["options"].get(kwarg) for kwarg in maxiter_kwargs),
    ]
    if any(maxiter_possibilities):
        maxiter = max([x for x in maxiter_possibilities if x is not None])
    else:
        maxiter = method_info["f_maxiter_default"](n_vars)

    for kwarg in maxiter_kwargs:
        if kwarg not in optimizer_kwargs["options"]:
            optimizer_kwargs["options"][kwarg] = maxiter

    return maxiter, optimizer_kwargs


def kwargs_to_options(optimizer_kwargs: dict, method: minimize_method | root_method) -> dict:
    optimizer_kwargs = deepcopy(optimizer_kwargs)

    NEVER_AUTO_PROMOTE = ["bounds", "tol", "jac_options"]
    option_kwargs = get_option_kwargs(method)["valid_options"]

    provided_kwargs = list(optimizer_kwargs.keys())
    options = optimizer_kwargs.get("options", {})

    for kwarg in option_kwargs:
        if kwarg in provided_kwargs and kwarg not in NEVER_AUTO_PROMOTE:
            options[kwarg] = optimizer_kwargs.pop(kwarg)

    optimizer_kwargs["options"] = options
    return optimizer_kwargs


def kwargs_to_jac_options(optimizer_kwargs: dict, method: root_method) -> dict:
    provided_kwargs = list(optimizer_kwargs.keys())
    method_jac_kwargs = get_option_kwargs(method).get("jac_options", None)

    if method_jac_kwargs is None:
        return optimizer_kwargs

    optimizer_kwargs = deepcopy(optimizer_kwargs)

    if "options" not in optimizer_kwargs:
        optimizer_kwargs["options"] = {}

    if (
        any(kwarg in method_jac_kwargs for kwarg in provided_kwargs)
        and "jac_options" not in optimizer_kwargs["options"]
    ):
        optimizer_kwargs["options"]["jac_options"] = {}

    for kwarg in provided_kwargs:
        if kwarg in method_jac_kwargs:
            optimizer_kwargs["options"]["jac_options"][kwarg] = optimizer_kwargs.pop(kwarg)

    return optimizer_kwargs


def determine_tolerance(optimizer_kwargs: dict, method: minimize_method | root_method) -> dict:
    tol = optimizer_kwargs.pop("tol", 1e-8)
    optimizer_kwargs["tol"] = tol
    method_options = get_option_kwargs(method)["valid_options"]
    method_tolerances = [tol_type for tol_type in method_options if tol_type in TOLERANCES]

    for tol_name in method_tolerances:
        if tol_name not in optimizer_kwargs["options"]:
            optimizer_kwargs["options"][tol_name] = tol

    return optimizer_kwargs
