import logging

from collections.abc import Callable
from copy import deepcopy

import numpy as np

from better_optimize.constants import (
    MINIMIZE_MODE_KWARGS,
    ROOT_MODE_KWARGS,
    TOLERANCES,
    minimize_method,
    root_method,
)

_log = logging.getLogger(__name__)


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
    uses_grad, uses_hess, uses_hessp, _ = MINIMIZE_MODE_KWARGS[method].values()

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

    if has_hess and not has_hessp and uses_hessp and uses_hess:
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

    if has_fused_f_and_grad and has_jac:
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
    optimizer_kwargs: dict, method: minimize_method | root_method
) -> tuple[int, dict]:
    MAXITER_KWARGS = ["maxiter", "maxfun", "maxfev"]
    maxiter = optimizer_kwargs.get("maxiter", 5000)
    optimizer_kwargs["options"].update({"maxiter": maxiter})
    maxiter_kwargs = [x for x in get_option_kwargs(method)["valid_options"] if x in MAXITER_KWARGS]

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
