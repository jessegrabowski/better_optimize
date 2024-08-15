from typing import Callable
import numpy as np
from better_optimize.constants import MODE_KWARGS, minimize_method
import logging

_log = logging.getLogger(__name__)


def validate_provided_functions(
        method: str,
        f_grad: Callable[[np.ndarray], np.ndarray] | None,
        f_hess: Callable[[np.ndarray], np.ndarray] | None,
        f_hessp: Callable[[np.ndarray], np.ndarray] | None,
        verbose=True
) -> None:
    has_grad, has_hess, has_hessp = map(lambda f: f is not None, [f_grad, f_hess, f_hessp])
    uses_grad, uses_hess, uses_hessp = MODE_KWARGS[method].values()

    if method not in MODE_KWARGS:
        raise ValueError(
            f"Method {method} not recognized. Must be one of {list(MODE_KWARGS.keys())}"
        )

    if has_hess and has_hessp:
        raise ValueError(
            "Cannot ask for Hessian and Hessian-vector product at the same time. For all algorithms "
            "except trust-exact and dogleg, use_hessp is preferred."
        )

    if has_grad and not uses_grad and verbose:
        _log.warning(f"Gradient provided but not used by method {method}. Gradients will still be evaluated at each "
                     f"optimzer step and the norm will be reported as a diagnositc. For large problems, this may be "
                     f"computationally intensive.")

    if (has_hess and not uses_hess) or (has_hessp and not uses_hessp) and verbose:
        _log.warning(f"Hessian provided but not used by method {method}. The full hessian will still be evaluated at "
                     f"each optimzer step and the norm will be reported as a diagnositc. For large problems, this may "
                     f"be computationally intensive.")

    if has_hess and not has_hessp and uses_hessp and uses_hess:
        _log.warning(f"You provided a function to compute the full hessian, but method {method} allows the use of a "
                     f"hessian-vector product instead. Consider passing hessp instead -- this may be significantly "
                     f"more efficient.")


def determine_maxiter(optimizer_kwargs: dict, method: minimize_method) -> tuple[int, dict]:
    maxiter = optimizer_kwargs.pop("maxiter", 5000)
    if "options" not in optimizer_kwargs:
        optimizer_kwargs["options"] = {}
    optimizer_kwargs["options"].update({"maxiter": maxiter})

    if method == "L-BFGS-B":
        optimizer_kwargs["options"].update({"maxfun": maxiter})

    return maxiter, optimizer_kwargs
