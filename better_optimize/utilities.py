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
        has_fused_f_and_grad: bool,
        verbose=True
) -> None:

    has_grad, has_hess, has_hessp = map(lambda f: f is not None, [f_grad, f_hess, f_hessp])
    uses_grad, uses_hess, uses_hessp = MODE_KWARGS[method].values()

    if has_fused_f_and_grad and has_grad:
        _log.warning("Objective function returns a tuple (interpreted as (loss, gradient), but a gradient function was "
                     "also provided. The gradient function will be ignored.")

    elif has_fused_f_and_grad:
        has_grad = True

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


def check_f_is_fused(f, x0, args):
    args = () if args is None else args
    output = f(x0, *args)
    if not isinstance(output, tuple | list):
        return False

    # If the output is a tuple, it should be length 2 (returning the value and the grad).If not, something is wrong
    if len(output) != 2:
        raise ValueError('Objective function should return either a scalar loss or a two-tuple of (loss, gradient)')
    return True

def determine_maxiter(optimizer_kwargs: dict, method: minimize_method) -> tuple[int, dict]:
    maxiter = optimizer_kwargs.pop("maxiter", 5000)
    optimizer_kwargs["options"].update({"maxiter": maxiter})

    if method in ["L-BFGS-B"]:
        optimizer_kwargs["options"].update({"maxfun": maxiter})
    if method in ['powell']:
        optimizer_kwargs["options"].update({"maxfev": maxiter})

    return maxiter, optimizer_kwargs


def determine_tolerance(optimizer_kwargs: dict, method: minimize_method) -> dict:
    tol = optimizer_kwargs.pop("tol", 1e-8)
    optimizer_kwargs["tol"]  = tol

    if method in ['nelder-mead', 'powell', 'TNC']:
        if 'xtol' not in optimizer_kwargs['options']:
            optimizer_kwargs["options"].update({"xtol": tol})
    if method in ['nelder-mead', 'powell', 'TNC', 'SLSQP']:
        if 'ftol' not in optimizer_kwargs['options']:
            optimizer_kwargs["options"].update({"ftol": tol})
    if method in ['CG', 'BFGS', 'TNC', 'trust-krylov', 'trust-exact', 'trust-ncg', 'trust-constr']:
        if 'gtol' not in optimizer_kwargs['options']:
            optimizer_kwargs["options"].update({"gtol": tol})

    return optimizer_kwargs