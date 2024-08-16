import pytest

from typing import get_args
from better_optimize.utilities import validate_provided_functions, determine_maxiter, determine_tolerance
from better_optimize.constants import minimize_method, MODE_KWARGS
from contextlib import contextmanager
from itertools import product

methods = get_args(minimize_method)


@contextmanager
def no_op(*args):
    yield


def func_not_none(f):
    return f is not None

@pytest.fixture
def settings():
    # Combinations of f_grad, f_hess, f_hessp
    return product([None, lambda x: x], repeat=3)


@pytest.mark.parametrize('method', methods, ids=methods)
def test_validate_provided_functions_raises_on_two_hess(settings, method: minimize_method):
    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        message = ('Cannot ask for Hessian and Hessian-vector product at the same time. For all algorithms '
                   'except trust-exact and dogleg, use_hessp is preferred.')
        manager = no_op() if not (use_hess and use_hessp) else pytest.raises(ValueError, match=message)
        with manager:
            validate_provided_functions(method, f_grad, f_hess, f_hessp, has_fused_f_and_grad=False, verbose=True)


@pytest.mark.parametrize('method', methods, ids=methods)
def test_validate_provided_functions_raises_on_two_hess(caplog, settings, method: minimize_method):
    uses_grad, uses_hess, uses_hessp = MODE_KWARGS[method].values()

    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        if use_hess and use_hessp:
            # Skip this error case, it's caught in another test
            continue

        validate_provided_functions(method, f_grad, f_hess, f_hessp, has_fused_f_and_grad=False, verbose=True)

        if use_grad and not uses_grad:
            message = f"Gradient provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if (use_hess and not uses_hess) or (use_hessp and not uses_hessp) :
            message = f"Hessian provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if uses_hessp and use_hess and not use_hessp:
            message = (f"You provided a function to compute the full hessian, but method {method} "
                       f"allows the use of a hessian-vector product instead.")
            assert any(message in log_message for log_message in caplog.messages)

        caplog.clear()


@pytest.mark.parametrize('method', methods, ids=methods)
def test_determine_maxiter(method: minimize_method):
    optimizer_kwargs = {'options': {}}
    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method)

    assert maxiter == 5000
    assert optimizer_kwargs['options']['maxiter'] == 5000

    if method == "L-BFGS-B":
        assert optimizer_kwargs['options']['maxfun'] == 5000
    else:
        assert 'maxfun' not in optimizer_kwargs['options']


@pytest.mark.parametrize('method', methods, ids=methods)
def test_determine_tolerance(method: minimize_method):
    optimizer_kwargs = {'options': {}, 'tol':1e-8}
    optimizer_kwargs = determine_tolerance(optimizer_kwargs, method)
    options = optimizer_kwargs['options']

    if method in ['nelder-mead', 'powell', 'TNC']:
        assert 'xtol' in options

    if method in ['nelder-mead', 'powell', 'TNC', 'SLSQP']:
        assert 'ftol' in options

    if method in ['CG', 'BFGS', 'TNC', 'trust-krylov', 'trust-exact', 'trust-ncg', 'trust-constr']:
        assert 'gtol' in options
