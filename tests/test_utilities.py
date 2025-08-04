import re

from contextlib import contextmanager
from itertools import product
from typing import get_args

import numpy as np
import pytest

from scipy.optimize import show_options

from better_optimize.constants import MINIMIZE_MODE_KWARGS, TOLERANCES, minimize_method, root_method
from better_optimize.utilities import (
    LRUCache1,
    check_f_is_fused_minimize,
    check_f_is_fused_root,
    determine_maxiter,
    determine_tolerance,
    get_option_kwargs,
    kwargs_to_jac_options,
    kwargs_to_options,
    validate_provided_functions_minimize,
)

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
                method,
                f_grad,
                f_hess,
                f_hessp,
                has_fused_f_and_grad=False,
                has_fused_f_grad_hess=False,
                verbose=True,
            )


@pytest.mark.parametrize("method", methods, ids=methods)
def test_validate_provided_functions_warnings(caplog, settings, method: minimize_method):
    uses_grad, uses_hess, uses_hessp, *_ = MINIMIZE_MODE_KWARGS[method].values()

    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        if use_hess and use_hessp:
            # Skip this error case, it's caught in another test
            continue

        validate_provided_functions_minimize(
            method,
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


@pytest.mark.parametrize("method", methods, ids=methods)
def test_determine_maxiter(method: minimize_method):
    all_maxiter_kwargs = ["maxiter", "maxfun", "maxfev"]
    method_info = get_option_kwargs(method)
    maxiter_kwargs = [x for x in method_info["valid_options"] if x in all_maxiter_kwargs]

    optimizer_kwargs = {"options": {}}
    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method, n_vars=100)

    expected_maxiter = method_info["f_maxiter_default"](100)
    assert maxiter == expected_maxiter

    for kwarg in maxiter_kwargs:
        assert optimizer_kwargs["options"][kwarg] == expected_maxiter

    for kwarg in all_maxiter_kwargs:
        if kwarg not in maxiter_kwargs:
            assert kwarg not in optimizer_kwargs["options"]


@pytest.mark.parametrize("method", methods, ids=methods)
def test_determine_tolerance(method: minimize_method):
    optimizer_kwargs = {"options": {}, "tol": 1e-8}
    optimizer_kwargs = determine_tolerance(optimizer_kwargs, method)
    options = optimizer_kwargs["options"]

    docstring = show_options(solver="minimize", method=method, disp=False)

    # Parse docstring to dictionary of headings and values
    headings = ["Parameters", "Options", "Returns", "References", "Notes"]
    formatted_headings = [heading + "\n" + "-" * len(heading) for heading in headings]
    pattern = r"(" + r"|".join(formatted_headings) + r")"
    intro, *blocks = re.split(pattern, re.sub(" {4}", "", docstring))

    # Structure of the list is heading - body - heading - body - ...
    block_dict = dict(zip(blocks[::2], blocks[1::2]))

    # Split body text into lines, keep only the parameter of the form "name : dtype"
    block_dict = {
        k.replace("-", "").strip().lower(): [
            x.split(":")[0].strip() for x in v.split("\n") if " : " in x
        ]
        for k, v in block_dict.items()
    }

    # In rare cases two parameters are on the same line, as in "name_1, name_2 : dtype"
    block_dict = {
        k: [item.strip() for x in v for item in x.split(",") if "*" not in item]
        for k, v in block_dict.items()
    }

    FILTER_NAMES = ["fun", "x0", "args", "method", "options", "callback"]
    all_options = sorted(
        [
            x
            for x in block_dict.get("options", []) + block_dict.get("parameters", [])
            if x not in FILTER_NAMES
        ]
    )

    expected_options = sorted(MINIMIZE_MODE_KWARGS[method]["valid_options"])

    # This sucks, but the scipy docstrings are not very consistent and some options are not documented. I hardcode the
    # missing options here
    undocumented_options = {
        "trust-ncg": {"workers"},
        "trust-krylov": {"workers", "eta", "max_trust_radius", "initial_trust_radius", "gtol"},
        "trust-exact": {"subproblem_maxiter"},
        "trust-constr": {
            "initial_barrier_parameter",
            "initial_tr_radius",
            "initial_barrier_tolerance",
        },
    }

    missing_options = (
        set(expected_options) - set(all_options) - set(undocumented_options.get(method, []))
    )
    assert not missing_options, "missing options: " + ", ".join(missing_options)

    expected_tols = [x for x in expected_options if x in TOLERANCES]
    assert all(options[tol] == 1e-8 for tol in expected_tols)


def test_kwargs_to_options():
    kwargs = {
        "return_all": True,
        "initial_simplex": "hello",
        "disp": 1,
        "fun": lambda x: x,
        "x0": [1, 2, 3],
    }
    option_kwargs = ["return_all", "initial_simplex", "disp"]
    not_option_kwargs = ["fun", "x0"]

    method: minimize_method = "nelder-mead"
    new_kwargs = kwargs_to_options(kwargs, method)

    # Test that the kwargs were moved to options
    assert all(x in new_kwargs["options"] for x in option_kwargs)

    # Test that non-option kwargs were not moved to options
    assert not any(x in new_kwargs["options"] for x in not_option_kwargs)
    assert all(x in new_kwargs for x in not_option_kwargs)

    # Test that we didn't silently modify the provided dict in place
    assert all(x in kwargs for x in option_kwargs)


def test_jac_kwargs_to_options():
    kwargs = {
        "tol_norm": None,
        "line_search": None,
        "max_rank": None,
        "alpha": None,
        "reduction_method": None,
        "to_retain": None,
        "disp": None,
    }
    jac_options = ["alpha", "reduction_method", "to_retain"]
    not_jac_options = ["tol_norm", "line_search", "max_rank"]

    method: root_method = "broyden1"
    new_kwargs = kwargs_to_jac_options(kwargs, method)

    assert "options" in new_kwargs
    assert "jac_options" in new_kwargs["options"]

    assert all(x in new_kwargs["options"]["jac_options"] for x in jac_options)
    assert not any(x in new_kwargs["options"]["jac_options"] for x in not_jac_options)

    assert all(x in new_kwargs for x in not_jac_options)


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
