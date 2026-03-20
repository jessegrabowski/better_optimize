import json
import os
import subprocess
import sys

from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import root as scipy_root

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper


@pytest.mark.parametrize(
    "root, method", [(True, "lm"), (False, "nelder-mead")], ids=["root-lm", "minimize-nm"]
)
def test_early_return_from_keyboard_interrupt(root, method):
    # Run error_script.py in a separate process to test KeyboardInterrupt handling.

    script_path = os.path.join(os.path.dirname(__file__), "util/error_script.py")
    args = [sys.executable, script_path, "--method", method]
    if root:
        args += ["--root"]

    process = subprocess.run(args, capture_output=True, text=True, check=False)

    res = json.loads(process.stdout.strip())
    assert not res["success"]

    if root:
        # lm doesn't allow callbacks, so we get the back return on interrupt
        assert (
            res["message"]
            == "`StopIteration` or `KeyboardInterrupt` raised -- optimization stopped prematurely."
        )
    else:
        # otherwise it should have stopped gracefully and given back the result object
        assert res["message"] == "`callback` raised `StopIteration`."


@pytest.mark.parametrize("root", [True, False], ids=["root", "minimize"])
def test_exception_returns_failed_result(root, monkeypatch):
    """Exceptions raised inside the optimizer should be caught and returned as a failed result."""
    N_EXEC = 0

    def f(x: np.ndarray):
        nonlocal N_EXEC
        N_EXEC += 1

        if N_EXEC == 1:
            return x**2
        raise Exception("Simulated error")

    objective = ObjectiveWrapper(f=f)

    f_optim = partial(
        scipy_root if root else scipy_minimize,
        fun=objective,
        x0=np.array([1.0]),
        method="lm" if root else "powell",
        callback=objective.callback,
    )

    result = optimizer_early_stopping_wrapper(f_optim)
    assert not result.success
    assert "Simulated error" in result.message


def func2(x, a, b):
    f = [a * x[0] * np.cos(x[1]) - 4, x[1] * x[0] - b * x[1] - 5]

    return np.array(f)


def func2_sparse_jac(x, a, b):
    df = np.array([[a * np.cos(x[1]), -a * x[0] * np.sin(x[1])], [x[1], x[0] - b]])
    return sp.csr_matrix(df)


def test_wrapper_compatible_with_sparse_outputs():
    x0 = np.array([0.8, 0.8])

    objective = ObjectiveWrapper(
        f=partial(func2, a=1, b=1),
        jac=partial(func2_sparse_jac, a=1, b=1),
        maxeval=100,
        progressbar=True,
        root=True,
    )

    with objective.progress:
        result = objective(x0)

    value, grad = result
    assert sp.issparse(grad)
