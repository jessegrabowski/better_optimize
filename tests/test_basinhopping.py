import numpy as np

from better_optimize.basinhopping import AllowFailureStorage, basinhopping
from better_optimize.utilities import ToggleableProgress


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
    import importlib

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
        niter=10,
        progressbar=False,
        accept_on_minimizer_fail=True,
    )

    global_minima = np.stack(global_minima)

    # Ensure the global minimum values are monotonically decreasing
    assert len(global_minima) > 0, "No global minima were recorded."

    minima_diff = np.diff(global_minima)[1:]
    assert (minima_diff <= 0.0).all(), "Global minima are not monotonically decreasing."


if __name__ == "__main__":
    test_basinhopping_1d()
