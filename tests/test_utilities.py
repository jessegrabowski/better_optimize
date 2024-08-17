import re

from contextlib import contextmanager
from itertools import product
from typing import get_args

import pytest

from scipy.optimize import show_options

from better_optimize.constants import MINIMIZE_MODE_KWARGS, TOLERANCES, minimize_method, root_method
from better_optimize.utilities import (
    determine_maxiter,
    determine_tolerance,
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
                method, f_grad, f_hess, f_hessp, has_fused_f_and_grad=False, verbose=True
            )


@pytest.mark.parametrize("method", methods, ids=methods)
def test_validate_provided_functions_warnings(caplog, settings, method: minimize_method):
    uses_grad, uses_hess, uses_hessp, _ = MINIMIZE_MODE_KWARGS[method].values()

    for f_grad, f_hess, f_hessp in settings:
        use_grad, use_hess, use_hessp = map(func_not_none, (f_grad, f_hess, f_hessp))

        if use_hess and use_hessp:
            # Skip this error case, it's caught in another test
            continue

        validate_provided_functions_minimize(
            method, f_grad, f_hess, f_hessp, has_fused_f_and_grad=False, verbose=True
        )

        if use_grad and not uses_grad:
            message = f"Gradient provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if (use_hess and not uses_hess) or (use_hessp and not uses_hessp):
            message = f"Hessian provided but not used by method {method}."
            assert any(message in log_message for log_message in caplog.messages)

        if uses_hessp and use_hess and not use_hessp:
            message = (
                f"You provided a function to compute the full hessian, but method {method} "
                f"allows the use of a hessian-vector product instead."
            )
            assert any(message in log_message for log_message in caplog.messages)

        caplog.clear()


@pytest.mark.parametrize("method", methods, ids=methods)
def test_determine_maxiter(method: minimize_method):
    optimizer_kwargs = {"options": {}}
    maxiter, optimizer_kwargs = determine_maxiter(optimizer_kwargs, method)

    assert maxiter == 5000
    assert optimizer_kwargs["options"]["maxiter"] == 5000

    if method in ["L-BFGS-B", "TNC"]:
        assert optimizer_kwargs["options"]["maxfun"] == 5000
    else:
        assert "maxfun" not in optimizer_kwargs["options"]


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

    # trust-constr docstring is really messy, just skip
    if method != "trust-constr":
        assert all(x in all_options for x in expected_options)

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
