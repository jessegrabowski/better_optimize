from contextlib import contextmanager
from itertools import product
from typing import get_args

import numpy as np
import pytest

from better_optimize.configuration import config_for_method
from better_optimize.constants import minimize_method, root_method
from better_optimize.utilities import (
    LRUCache1,
    check_f_is_fused_minimize,
    check_f_is_fused_root,
    validate_provided_functions_minimize,
)

methods = get_args(minimize_method)
root_methods = get_args(root_method)


@contextmanager
def no_op(*args):
    yield


def func_not_none(f):
    return f is not None


@pytest.fixture
def settings():
    # Combinations of f_grad, f_hess, f_hessp
    return product([None, lambda x: x], repeat=3)


@pytest.mark.parametrize("method", methods, ids=methods)
def test_validate_provided_functions_raises_on_two_hess(settings, method: minimize_method):
    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        message = (
            "Cannot ask for Hessian and Hessian-vector product at the same time. For all algorithms "
            "except trust-exact and dogleg, use_hessp is preferred."
        )
        manager = (
            no_op() if not (use_hess and use_hessp) else pytest.raises(ValueError, match=message)
        )
        with manager:
            validate_provided_functions_minimize(
                config_for_method(method),
                f_grad,
                f_hess,
                f_hessp,
                has_fused_f_and_grad=False,
                has_fused_f_grad_hess=False,
                verbose=True,
            )


@pytest.mark.parametrize("method", methods, ids=methods)
def test_validate_provided_functions_warnings(caplog, settings, method: minimize_method):
    config = config_for_method(method)
    uses_grad, uses_hess, uses_hessp = config.uses_grad, config.uses_hess, config.uses_hessp

    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        if use_hess and use_hessp:
            # Skip this error case, it's caught in another test
            continue

        validate_provided_functions_minimize(
            config,
            f_grad,
            f_hess,
            f_hessp,
            has_fused_f_and_grad=False,
            has_fused_f_grad_hess=False,
            verbose=True,
        )

        if use_grad and not uses_grad:
            message = f"Gradient provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if (use_hess and not uses_hess) or (use_hessp and not uses_hessp):
            message = f"Hessian or Hessian-vector product provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if uses_hessp and use_hess and not use_hessp:
            message = (
                f"You provided a function to compute the full Hessian, but method {method} allows the use of a "
                f"Hessian-vector product instead."
            )
            assert any(message in log_message for log_message in caplog.messages)

        caplog.clear()


@pytest.mark.parametrize(
    "output,expected",
    [
        (lambda x: np.sum(x**2), (False, False)),
        (lambda x: (np.sum(x**2), np.ones_like(x)), (True, False)),
        (lambda x: (np.sum(x**2), np.ones_like(x), np.eye(len(x))), (True, True)),
    ],
    ids=["scalar_only", "scalar_and_grad", "scalar_grad_hess"],
)
def test_check_f_is_fused_minimize_valid(output, expected):
    x0 = np.array([1.0, 2.0])
    assert check_f_is_fused_minimize(output, x0, None) == expected


@pytest.mark.parametrize(
    "output",
    [
        lambda x: (np.sum(x**2), 1.0),
        lambda x: (np.sum(x**2), np.ones_like(x), np.ones_like(x)),
        lambda x: (np.ones_like(x), np.ones_like(x)),
        lambda x: (np.sum(x**2), np.ones_like(x), np.eye(len(x)), 123),
    ],
    ids=["grad_not_1d", "hess_not_2d", "value_not_scalar", "tuple_wrong_length"],
)
def test_check_f_is_fused_minimize_invalid(output):
    x0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        check_f_is_fused_minimize(output, x0, None)


@pytest.mark.parametrize(
    "output,expected",
    [
        (lambda x: np.ones_like(x), False),
        (lambda x: (np.ones_like(x), np.eye(len(x))), True),
    ],
)
def test_check_f_is_fused_root_valid(output, expected):
    x0 = np.array([1.0, 2.0])
    assert check_f_is_fused_root(output, x0, None) == expected


@pytest.mark.parametrize(
    "output",
    [
        lambda x: (1.0, np.eye(len(x))),
        lambda x: (np.ones_like(x), np.ones_like(x)),
        lambda x: (np.ones_like(x), np.eye(len(x)), 123),
    ],
    ids=["jac_not_1d", "jac_not_2d", "tuple_wrong_length"],
)
def test_check_f_is_fused_root_invalid(output):
    x0 = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        check_f_is_fused_root(output, x0, None)


class TestLRUCache1:
    def test_basic_behavior(self):
        calls = {"count": 0}

        def f(x):
            calls["count"] += 1
            return (np.sum(x), x + 1, np.eye(len(x)))

        cache = LRUCache1(f)
        x = np.array([1.0, 2.0])

        # First call: cache miss
        result1 = cache(x)
        assert np.allclose(result1[0], 3.0)
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0
        assert calls["count"] == 1

        # Second call with same x: cache hit
        result2 = cache(x)
        assert np.allclose(result2[0], 3.0)
        assert cache.cache_misses == 1
        assert cache.cache_hits == 1
        assert calls["count"] == 1

        # Third call with different x: cache miss
        x2 = np.array([2.0, 3.0])
        result3 = cache(x2)
        assert np.allclose(result3[0], 5.0)
        assert cache.cache_misses == 2
        assert cache.cache_hits == 1
        assert calls["count"] == 2

    def test_value_grad_hess_methods(self):
        def f(x):
            return (np.sum(x), x * 2, np.eye(len(x)))

        cache = LRUCache1(f, f_returns_list=True)
        x = np.array([1.0, 2.0])

        val = cache.value(x)
        grad = cache.grad(x)
        val_grad = cache.value_and_grad(x)
        hess = cache.hess(x)

        assert val == 3.0
        assert np.allclose(grad, [2.0, 4.0])
        assert val_grad[0] == 3.0 and np.allclose(val_grad[1], [2.0, 4.0])
        assert np.allclose(hess, np.eye(2))

        # Check call counters
        assert cache.value_calls == 1
        assert cache.grad_calls == 1
        assert cache.value_and_grad_calls == 1
        assert cache.hess_calls == 1

    def test_clear_cache(self):
        def f(x):
            return (np.sum(x), x)

        cache = LRUCache1(f)
        x = np.array([1.0, 2.0])
        cache(x)
        cache.value(x)
        cache.grad(x)
        cache.value_and_grad(x)
        cache.hess(x)
        cache.clear_cache()
        assert cache.last_x is None
        assert cache.last_result is None
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
        assert cache.value_calls == 0
        assert cache.grad_calls == 0
        assert cache.value_and_grad_calls == 0
        assert cache.hess_calls == 0

    def test_dtype_and_copyx(self):
        def f(x, *args):
            return (np.sum(x), x)

        x = np.array([1.0, 2.0], dtype=np.float64)
        cache = LRUCache1(f, copy_x=True, dtype="float64")
        result = cache(x)
        assert isinstance(result, tuple)
        assert np.allclose(result[0], 3.0)
        # Changing x after call should not affect cache
        x[0] = 100.0
        result2 = cache(np.array([1.0, 2.0], dtype=np.float64))
        assert np.allclose(result2[0], 3.0)
