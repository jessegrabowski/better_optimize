import json
import os
import pickle
import signal
import subprocess
import sys
import textwrap

import pytest

from scipy.optimize import OptimizeResult

from better_optimize import OptimizationInterrupted

# The child prints READY from inside the objective once the solver is running, then slows down
# so the SIGINT lands mid-solve. Waiting on READY, not a fixed sleep, keeps the timing stable.
CHILD_TEMPLATE = textwrap.dedent(
    """
    import json
    import sys
    import time

    import numpy as np

    from better_optimize import (
        OptimizationInterrupted,
        basinhopping,
        differential_evolution,
        minimize,
        multi_optimize,
        root,
        sequential_optimize,
    )

    n_calls = 0


    def objective(x):
        global n_calls
        n_calls += 1
        if n_calls == 40:
            print("READY", flush=True)
        if n_calls >= 40:
            time.sleep(0.05)
        return float(np.sum((x - 1.0) ** 2))


    def residual(x):
        objective(x)
        return x**2 + 1.0


    x0 = np.zeros(3)
    nelder_mead = {{"method": "nelder-mead", "options": {{"maxiter": 100_000, "xatol": 0, "fatol": 0}}}}
    try:
        {call}
    except OptimizationInterrupted as interrupt:
        result = interrupt.result
        fun = getattr(result, "fun", None)
        report = {{
            "type": type(result).__name__,
            "fun": None if fun is None else float(np.linalg.norm(fun)),
        }}
        print(json.dumps(report))
        sys.exit(0)
    print(json.dumps({{"type": "returned"}}))
    """
)

CALLS = {
    "minimize": "minimize(objective, x0=x0, progressbar=False, **nelder_mead)",
    "root": "root(residual, x0=np.zeros(100), method='hybr', progressbar=False)",
    "basinhopping": (
        "basinhopping(objective, x0=x0, niter=100_000, progressbar=False, "
        "minimizer_kwargs={'method': 'powell'})"
    ),
    "differential_evolution": (
        "differential_evolution(objective, bounds=[(-5.0, 5.0)] * 3, maxiter=100_000, "
        "tol=0.0, progressbar=False)"
    ),
    "sequential_optimize": (
        "sequential_optimize(objective, x0=x0, progressbar=False, "
        "stages=[{'solver': minimize, **nelder_mead}])"
    ),
    "multi_optimize": (
        "multi_optimize(minimize, x0=x0, n_runs=4, n_jobs=2, backend='loky', progressbar=False, "
        "solver_kwargs={'f': objective, **nelder_mead})"
    ),
}

EXPECTED_TYPES = {
    "minimize": "OptimizeResult",
    "root": "OptimizeResult",
    "basinhopping": "OptimizeResult",
    "differential_evolution": "OptimizeResult",
    "sequential_optimize": "SequentialResult",
    "multi_optimize": "MultiStartResult",
}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and SIGINT only")
@pytest.mark.parametrize("solver", CALLS)
def test_sigint_raises_with_a_partial_result(solver):
    # A new session puts the child and any workers it spawns in one process group, which is
    # what a terminal Ctrl-C signals.
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD_TEMPLATE.format(call=CALLS[solver])],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        os.killpg(child.pid, signal.SIGINT)
        stdout, stderr = child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)

    assert child.returncode == 0, stderr
    report = json.loads(stdout.strip().splitlines()[-1])
    assert report["type"] == EXPECTED_TYPES[solver]
    if report["fun"] is not None:
        assert report["fun"] < float("inf")


def test_interrupt_survives_pickling():
    # loky workers send exceptions back to the parent pickled.
    interrupt = OptimizationInterrupted(OptimizeResult(x=[1.0], fun=2.0), "stopped")
    restored = pickle.loads(pickle.dumps(interrupt))

    assert isinstance(restored, KeyboardInterrupt)
    assert restored.result.fun == 2.0
    assert str(restored) == "stopped"
