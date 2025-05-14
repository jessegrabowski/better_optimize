import json
import os
import subprocess
import sys

from functools import partial

import numpy as np
import pytest

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
def test_execption_breaks_optimization(root, monkeypatch):
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

    # Non-KeyboardInterrupt exceptions should break execution.
    with pytest.raises(Exception, match="Simulated error"):
        optimizer_early_stopping_wrapper(f_optim)
