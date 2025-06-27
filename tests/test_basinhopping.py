import importlib

import numpy as np
import pytest

from better_optimize.basinhopping import AllowFailureStorage, basinhopping
from better_optimize.utilities import LRUCache1, ToggleableProgress


def func1d(x):
    f = np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * np.sin(14.5 * x - 0.3) + 2.0 * x + 0.2)
    return f, df


def func2d_nograd(x):
    f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f


def func2d(x):
    f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    df = np.zeros(2)
    df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2.0 * x[0] + 0.2
    df[1] = 2.0 * x[1] + 0.2

    return f, df


def func2d_easyderiv(x):
    f = 2.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + 2.0 * x[1] ** 2 - 6.0 * x[0]
    df = np.zeros(2)
    df[0] = 4.0 * x[0] + 2.0 * x[1] - 6.0
    df[1] = 2.0 * x[0] + 4.0 * x[1]

    return f, df


def func2d_hard_fused(x):
    f = np.sin(5 * np.pi * x[0]) * (1 - x[0]) + (x[1] - 0.5) ** 2
    df = np.zeros_like(x)
    df[0] = 5 * np.pi * np.cos(5 * np.pi * x[0]) * (1 - x[0]) - np.sin(5 * np.pi * x[0])
    df[1] = 2 * (x[1] - 0.5)
    return f, df


def func2d_hess(x):
    # Hessian for func2d_easyderiv
    H = np.array([[4.0, 2.0], [2.0, 4.0]])
    return H


def func2d_hessp(x, p):
    # Hessian-vector product for func2d_easyderiv
    H = np.array([[4.0, 2.0], [2.0, 4.0]])
    return H @ p


def func2d_triple_fused(x):
    f, df = func2d_easyderiv(x)
    H = func2d_hess(x)
    return f, df, H


def test_basinhopping_1d():
    res = basinhopping(
        func1d,
        x0=[1.0],
        minimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )
    assert res.success is True
    np.testing.assert_allclose(res.x, -0.195, atol=1e-3, rtol=1e-3)


def test_basinhopping_2d():
    res = basinhopping(
        func2d,
        x0=[1.0, 1.0],
        minimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )

    assert res.success is True
    np.testing.assert_allclose(res.x, np.array([-0.195, -0.1]), atol=1e-3, rtol=1e-3)


def test_basinhopping_nograd():
    res = basinhopping(
        func2d,
        x0=[1.0, 1.0],
        minimizer_kwargs={"method": "nelder-mead", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )

    assert res.success is True
    np.testing.assert_allclose(res.x, np.array([-0.195, -0.1]), atol=1e-3, rtol=1e-3)


def test_progress_global_gradient_updates_on_new_minimum(monkeypatch):
    gradient_updates = []
    original_update = ToggleableProgress.update

    def mock_update(self, task_id, **kwargs):
        if "grad_norm" in kwargs and task_id == 0:
            gradient_updates.append(kwargs["grad_norm"])
        return original_update(self, task_id, **kwargs)

    monkeypatch.setattr("better_optimize.utilities.ToggleableProgress.update", mock_update)

    basinhopping(
        func2d_hard_fused,
        x0=[0.8, 0.8],  # Start away from the global minimum
        minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
        niter=10,
        progressbar=True,
        accept_on_minimizer_fail=True,
    )
    assert (
        len(gradient_updates) > 0
    ), "No gradient norm updates were recorded for the global minimum."
    assert all(
        isinstance(grad, float) for grad in gradient_updates
    ), "Invalid gradient norm values for the global minimum."
    assert all(grad > 0 for grad in gradient_updates), "Gradient norm values should be positive."


def test_move_to_new_minimum_on_optimizer_failure(monkeypatch):
    # Required, because otherwise monkeypatch can't find the *module* basinhopping; only the function
    bh_module = importlib.import_module("better_optimize.basinhopping")
    original_minimize = bh_module.minimize
    original_storage_update = AllowFailureStorage.update

    global_minima = []

    # Mock the inner optimizer to always fail
    def mock_minimize(*args, **kwargs):
        res = original_minimize(*args, **kwargs)
        res["success"] = False
        return res

    # Track when the storage accepts a new minimum
    def mock_storage_update(self, minres):
        result = original_storage_update(self, minres)
        if result:  # If the update was accepted
            global_minima.append(minres.fun)
        return result

    monkeypatch.setattr(bh_module, "minimize", mock_minimize)
    monkeypatch.setattr(AllowFailureStorage, "update", mock_storage_update)

    basinhopping(
        func2d_hard_fused,
        x0=[0.8, 0.8],  # Start away from the global minimum
        minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
        niter=25,
        progressbar=False,
        accept_on_minimizer_fail=True,
    )

    global_minima = np.stack(global_minima)

    # Ensure the global minimum values are monotonically decreasing
    assert len(global_minima) > 0, "No global minima were recorded."

    minima_diff = np.diff(global_minima)[1:]
    assert (minima_diff <= 0.0).all(), "Global minima are not monotonically decreasing."


@pytest.mark.parametrize(
    "minimizer_kwargs",
    [
        {"method": "nelder-mead", "tol": 1e-8},
        {"method": "L-BFGS-B", "tol": 1e-8, "jac": lambda x: func2d_easyderiv(x)[1]},
        {
            "method": "trust-exact",
            "tol": 1e-8,
            "jac": lambda x: func2d_easyderiv(x)[1],
            "hess": func2d_hess,
        },
        {
            "method": "trust-ncg",
            "tol": 1e-8,
            "jac": lambda x: func2d_easyderiv(x)[1],
            "hessp": func2d_hessp,
        },
    ],
    ids=["value_only", "jac_only", "hess_only", "hessp_only"],
)
def test_basinhopping_func2d_easyderiv_variants(minimizer_kwargs):
    res = basinhopping(
        func2d_easyderiv,
        x0=[1.0, 1.0],
        minimizer_kwargs=minimizer_kwargs,
        niter=10,
        progressbar=False,
    )
    assert res.success is True
    np.testing.assert_allclose(res.x, np.array([2.0, -1.0]), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "fused_func, minimizer_kwargs, expected",
    [
        (func2d, {"method": "L-BFGS-B", "tol": 1e-8}, np.array([-0.195, -0.1])),
        (func2d_triple_fused, {"method": "trust-exact", "tol": 1e-8}, np.array([2.0, -1.0])),
    ],
    ids=["func2d_fused", "func2d_triple_fused"],
)
def test_basinhopping_fused_variants(fused_func, minimizer_kwargs, expected):
    res = basinhopping(
        fused_func,
        x0=[1.0, 1.0],
        minimizer_kwargs=minimizer_kwargs,
        niter=25,
        progressbar=False,
    )
    assert res.success is True
    np.testing.assert_allclose(res.x, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "fused_func, minimizer_kwargs, expected",
    [
        (func2d, {"method": "L-BFGS-B", "tol": 1e-8}, np.array([-0.195, -0.1])),
        (func2d_triple_fused, {"method": "trust-exact", "tol": 1e-8}, np.array([2.0, -1.0])),
    ],
    ids=["func2d_fused", "func2d_triple_fused"],
)
def test_basinhopping_fused_variants_cache(monkeypatch, fused_func, minimizer_kwargs, expected):
    cache_holder = {}

    def accessible_LRUCache1(*args, **kwargs):
        cache = LRUCache1(*args, **kwargs)
        cache_holder["cache"] = cache
        return cache

    basinhopping_mod = importlib.import_module("better_optimize.basinhopping")

    with monkeypatch.context() as c:
        c.setattr(basinhopping_mod, "LRUCache1", accessible_LRUCache1)
        res = basinhopping(
            fused_func,
            x0=[1.0, 1.0],
            minimizer_kwargs=minimizer_kwargs,
            niter=25,
            progressbar=False,
        )

    assert res.success is True
    np.testing.assert_allclose(res.x, expected, atol=1e-3, rtol=1e-3)

    cache = cache_holder.get("cache")
    assert cache is not None
    assert (cache.cache_hits + cache.cache_misses) > 0
    assert cache.value_and_grad_calls > 0
    if fused_func is func2d_triple_fused:
        assert cache.cache_hits > 0
        assert cache.hess_calls > 0


def test_initial_minimization_appears_on_progress_bar(monkeypatch):
    bh_module = importlib.import_module("better_optimize.basinhopping")
    AllowFailureStorage = bh_module.AllowFailureStorage

    initial_states = {}
    progress_updates = []

    # Patch AllowFailureStorage._add to record the initial minres and inject nonsense values
    original_add = AllowFailureStorage._add

    def tracking_add(self, minres):
        # Only record the very first call (initialization)
        if not initial_states:
            # Inject nonsense values
            minres.x = np.array([42.0, -99.0])
            minres.fun = 12345.678
            minres.success = False

            initial_states["x"] = np.copy(minres.x)
            initial_states["fun"] = minres.fun
            initial_states["success"] = minres.success
        return original_add(self, minres)

    monkeypatch.setattr(AllowFailureStorage, "_add", tracking_add)
    original_update = ToggleableProgress.update

    def tracking_update(self, task_id, **kwargs):
        # Only track updates for task 0 (the main basinhopping task)
        if task_id == 0:
            progress_updates.append(dict(kwargs))
        return original_update(self, task_id, **kwargs)

    monkeypatch.setattr(ToggleableProgress, "update", tracking_update)

    x0 = np.array([0.5, 0.5])
    bh_module.basinhopping(
        func2d_hard_fused,
        x0=x0,
        minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
        niter=1,
        progressbar=True,
        accept_on_minimizer_fail=True,
    )

    # The initial state should be present and success should be False
    assert "x" in initial_states and "fun" in initial_states and "success" in initial_states
    np.testing.assert_allclose(initial_states["x"], np.array([42.0, -99.0]), atol=1e-8)
    assert initial_states["fun"] == 12345.678
    assert initial_states["success"] is False

    # The first progressbar update for task 0 should reflect the nonsense values
    found = False
    for upd in progress_updates:
        if "f_value" in upd and abs(upd["f_value"] - 12345.678) < 1e-6:
            found = True
            break
    assert found, "Injected nonsense fun value did not appear in progressbar update"


def test_minimizer_progressbar_total_constant(monkeypatch):
    """
    Test that the minimizer progress bar's total (task 1) is always the requested maxiter value
    at the beginning of each iteration.
    """
    bh_module = importlib.import_module("better_optimize.basinhopping")
    ToggleableProgress = importlib.import_module("better_optimize.utilities").ToggleableProgress

    minimizer_totals = []

    original_update = ToggleableProgress.update

    def tracking_update(self, task_id, **kwargs):
        # Task 1 is the minimizer progress bar
        if task_id == 1:
            # Record the total at each update
            minimizer_totals.append(self.tasks[task_id].total)
        return original_update(self, task_id, **kwargs)

    monkeypatch.setattr(ToggleableProgress, "update", tracking_update)

    # Use a small maxiter for a quick test
    maxiter = 7
    x0 = np.array([0.5, 0.5])
    bh_module.basinhopping(
        func2d_hard_fused,
        x0=x0,
        minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "maxiter": maxiter},
        niter=5,
        progressbar=True,
        accept_on_minimizer_fail=True,
    )

    # All recorded totals should be equal to maxiter
    assert minimizer_totals, "No minimizer progress bar updates were recorded."

    # Scipy doesn't guarantee that the minimizer will always run for maxiter iterations, so just check that the
    # majority of the recorded totals are equal to maxiter
    assert sum(total == maxiter for total in minimizer_totals) / len(minimizer_totals) > 0.51
