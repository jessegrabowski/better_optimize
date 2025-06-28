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
    has_fused_f_grad_hess: bool,
    verbose=True,
) -> tuple[bool, bool, bool]:
    has_grad, has_hess, has_hessp = map(lambda f: f is not None, [f_grad, f_hess, f_hessp])
    uses_grad, uses_hess, uses_hessp, *_ = MINIMIZE_MODE_KWARGS[method].values()

    # Handle fused outputs first
    if has_fused_f_grad_hess:
        if verbose and (has_grad or has_hess):
            _log.warning(
                "Objective function returns a tuple (interpreted as (loss, gradient, hessian)), "
                "but a gradient or hessian function was also provided. The gradient and hessian functions will be ignored."
            )
        if verbose and has_hessp:
            _log.warning(
                "Objective function returns a tuple (interpreted as (loss, gradient, hessian)), "
                "but a hessian-vector product function was also provided. The hessian-vector product function will be ignored."
            )
        # Triple-fused disables external grad/hess/hessp
        has_grad = True
        has_hess = True
        has_hessp = False

    elif has_fused_f_and_grad:
        if verbose and has_grad:
            _log.warning(
                "Objective function returns a tuple (interpreted as (loss, gradient)), "
                "but a gradient function was also provided. The gradient function will be ignored."
            )
        has_grad = True  # fused (loss, grad) disables external grad

    if method not in MINIMIZE_MODE_KWARGS:
        raise ValueError(
            f"Method {method} not recognized. Must be one of {list(MINIMIZE_MODE_KWARGS.keys())}"
        )

    if has_hess and has_hessp:
        raise ValueError(
            "Cannot ask for Hessian and Hessian-vector product at the same time. For all algorithms "
            "except trust-exact and dogleg, use_hessp is preferred."
        )

    if verbose and (has_grad and not uses_grad):
        _log.warning(
            f"Gradient provided but not used by method {method}. Gradients will still be evaluated at each "
            f"optimizer step and the norm will be reported as a diagnostic. For large problems, this may be "
            f"computationally intensive."
        )

    if verbose and ((has_hess and not uses_hess) or (has_hessp and not uses_hessp)):
        _log.warning(
            f"Hessian or Hessian-vector product provided but not used by method {method}. "
            f"The full Hessian or Hessian-vector product will still be evaluated at each optimizer step and "
            f"the norm will be reported as a diagnostic. For large problems, this may be computationally intensive."
        )

    if verbose and (has_hess and not has_hessp and uses_hessp and uses_hess):
        _log.warning(
            f"You provided a function to compute the full Hessian, but method {method} allows the use of a "
            f"Hessian-vector product instead. Consider passing hessp instead -- this may be significantly "
            f"more efficient."
        )

    return has_grad, has_hess, has_hessp


def validate_provided_functions_root(
    method: root_method, f, jac, has_fused_f_and_grad: bool, verbose: bool = True
) -> bool:
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

    return has_jac


def check_f_is_fused_minimize(f, x0, args) -> tuple[bool, bool]:
    """
    Check if the minimize objective function returns fused outputs (value, grad[, hess]).
    Returns (is_fused, has_hess).
    """
    args = () if args is None else args
    output = f(x0, *args)

    if not isinstance(output, tuple | list):
        return (False, False)

    if len(output) == 2:
        value, grad = output
        hess = None
        ret_val = (True, False)
    elif len(output) == 3:
        value, grad, hess = output
        ret_val = (True, True)
    else:
        raise ValueError(
            "Objective function should return either a scalar loss, a two-tuple of (loss, gradient), "
            "or three-tuple of (loss, gradient, hessian)."
        )

    if not (
        np.isscalar(value)
        or (hasattr(value, "shape") and (value.shape == () or value.shape == (1,)))
    ):
        raise ValueError(
            "First value returned by the objective function must be a scalar (float or 0-d array)."
        )

    if not (hasattr(grad, "ndim") and grad.ndim == 1):
        raise ValueError(
            "Second value returned by the objective function must be a 1d array representing the gradient."
        )

    if hess is not None:
        if not (hasattr(hess, "ndim") and hess.ndim == 2):
            raise ValueError(
                "Third value returned by the objective function must be a 2d array representing the Hessian."
            )

    return ret_val


def check_f_is_fused_root(f, x0, args) -> bool:
    """
    Check if the root objective function returns fused outputs (value[, jac]), and returns True if it does.
    """
    args = () if args is None else args
    output = f(x0, *args)

    if not isinstance(output, tuple | list):
        return False

    if len(output) == 2:
        value, jac = output
        ret_val = True
    elif len(output) == 1:
        value = output[0]
        jac = None
        ret_val = True
    else:
        raise ValueError(
            "Objective function should return either a 1d array or a two-tuple of (value, jacobian)."
        )

    if not (hasattr(value, "ndim") and value.ndim == 1):
        raise ValueError("First value returned by the objective function must be a 1d array.")

    if jac is not None:
        if not (hasattr(jac, "ndim") and jac.ndim == 2):
            raise ValueError(
                "Second value returned by the objective function must be a 2d array representing the Jacobian."
            )

    return ret_val


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


class LRUCache1:
    """
    Simple LRU cache with a memory size of 1.

    This cache is only usable for a function that takes a single input `x` and returns a single output. The
    function can also take any number of additional arguments `*args`, but these are assumed to be constant
    between function calls.

    The purpose of this cache is to allow for Hessian computation to be reused when calling scipy.optimize functions.
    It is very often the case that some sub-computations are repeated between the objective, gradient, and hessian
    functions, but by default scipy only allows for the objective and gradient to be fused.

    By using this cache, all 3 functions can be fused, which can significantly speed up the optimization process for
    expensive functions.
    """

    def __init__(
        self, fn, f_returns_list: bool = False, copy_x: bool = False, dtype: str | None = None
    ):
        self.fn = fn
        self.last_x = None
        self.last_result = None
        self.copy_x = copy_x
        self.f_returns_list = f_returns_list

        # Scipy does not respect dtypes *at all*, so we have to force it ourselves.
        self.dtype = dtype

        self.cache_hits = 0
        self.cache_misses = 0

        self.value_calls = 0
        self.grad_calls = 0
        self.value_and_grad_calls = 0
        self.hess_calls = 0

    def __call__(self, x, *args):
        """
        Call the cached function with the given input `x` and additional arguments `*args`.

        If the input `x` is the same as the last input, return the cached result. Otherwise update the cache with the
        new input and result.
        """
        x = x.astype(self.dtype)

        if self.last_result is None or not (x == self.last_x).all():
            self.cache_misses += 1

            # scipy.optimize.root changes x in place, so the cache has to copy it, otherwise we get false
            # cache hits and optimization always fails.
            if self.copy_x:
                x = x.copy()
            self.last_x = x

            result = self.fn(x, *args)
            self.last_result = result

            return result

        else:
            self.cache_hits += 1
            return self.last_result

    def value(self, x, *args):
        self.value_calls += 1
        if not self.f_returns_list:
            return self(x, *args)
        else:
            return self(x, *args)[0]

    def grad(self, x, *args):
        self.grad_calls += 1
        return self(x, *args)[1]

    def value_and_grad(self, x, *args):
        self.value_and_grad_calls += 1
        return self(x, *args)[:2]

    def hess(self, x, *args):
        self.hess_calls += 1
        return self(x, *args)[-1]

    def report(self):
        _log.info(f"Value and Grad calls: {self.value_and_grad_calls}")
        _log.info(f"Hess Calls: {self.hess_calls}")
        _log.info(f"Hits: {self.cache_hits}")
        _log.info(f"Misses: {self.cache_misses}")

    def clear_cache(self):
        self.last_x = None
        self.last_result = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.value_calls = 0
        self.grad_calls = 0
        self.value_and_grad_calls = 0
        self.hess_calls = 0
