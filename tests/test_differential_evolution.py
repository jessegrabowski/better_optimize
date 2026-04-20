import itertools

import numpy as np
import pytest

from scipy.optimize import OptimizeResult, rosen, rosen_der

from better_optimize import differential_evolution, minimize, sequential_optimize


def _rosen_fused(x):
    return rosen(x), rosen_der(x)


def _rastrigin(x):
    x = np.asarray(x)
    return 10.0 * x.size + float(np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def test_de_finds_rosenbrock_2d():
    res = differential_evolution(
        rosen,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=200,
        rng=0,
        progressbar=False,
    )
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-5)
    assert res.fun < 1e-8


def test_de_finds_rastrigin_5d():
    res = differential_evolution(
        _rastrigin,
        bounds=[(-5.12, 5.12)] * 5,
        maxiter=500,
        rng=0,
        progressbar=False,
    )
    np.testing.assert_allclose(res.x, np.zeros(5), atol=1e-4)
    assert res.fun < 1e-6


def test_de_fused_function_accepted():
    res = differential_evolution(
        _rosen_fused,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=100,
        rng=0,
        progressbar=False,
    )
    assert np.isfinite(res.fun)
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)


def test_de_vectorized_passthrough():
    """Vectorized=True should dispatch batched evaluations."""

    def batched_rosen(x):
        # x shape: (d, popsize). Returns shape (popsize,).
        x = np.atleast_2d(x)
        return 100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2 @ np.ones(1)

    def rosen_batched(x):
        # Simpler: compute rosen column-wise
        x = np.atleast_2d(x)
        return np.array([rosen(col) for col in x.T])

    res = differential_evolution(
        rosen_batched,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=100,
        rng=0,
        vectorized=True,
        workers=1,
        progressbar=False,
    )
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-3)


def test_de_progress_bar_records_updates(monkeypatch):
    updates = []

    import better_optimize.utilities as _util

    orig = _util.ToggleableProgress.update

    def tracking(self, task_id, **kwargs):
        if "f_value" in kwargs:
            updates.append(float(kwargs["f_value"]))
        return orig(self, task_id, **kwargs)

    monkeypatch.setattr(_util.ToggleableProgress, "update", tracking)

    differential_evolution(
        rosen,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=10,
        rng=0,
        progressbar=True,
    )
    # At least one generation update was recorded, values were finite
    assert len(updates) > 0
    assert all(np.isfinite(v) or v == float("inf") for v in updates)
    # Best-so-far is monotone non-increasing (modulo initial +inf)
    finite = [v for v in updates if np.isfinite(v)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(finite))


def test_de_keyboard_interrupt_returns_result():
    call_count = [0]

    def user_callback(xk, convergence=0.0):
        call_count[0] += 1
        if call_count[0] >= 3:
            return True  # request early stop
        return False

    res = differential_evolution(
        rosen,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=1000,
        callback=user_callback,
        rng=0,
        progressbar=False,
    )
    assert not res.success
    # scipy handles the stop itself; message mentions "early" or our wrapper says "interrupted"
    assert "early" in res.message.lower() or "interrupted" in res.message.lower()


def test_de_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        differential_evolution(
            rosen,
            bounds=[(-1.0, 1.0)],
            strategy="nonexistent",
            progressbar=False,
        )


def test_de_invalid_init_raises():
    with pytest.raises(ValueError, match="init"):
        differential_evolution(
            rosen,
            bounds=[(-1.0, 1.0)],
            init="not_a_method",
            progressbar=False,
        )


def test_de_invalid_bounds_shape():
    with pytest.raises(ValueError, match="bounds"):
        differential_evolution(
            rosen,
            bounds=[1.0, 2.0, 3.0],
            progressbar=False,
        )


def test_de_inverted_bounds():
    with pytest.raises(ValueError, match="low > high"):
        differential_evolution(
            rosen,
            bounds=[(5.0, -5.0)],
            progressbar=False,
        )


def test_de_maxiter_default_scales_with_d():
    from better_optimize.differential_evolution import _default_maxiter

    assert _default_maxiter(2) == 1000
    assert _default_maxiter(5) == 1000
    assert _default_maxiter(10) == 2000
    assert _default_maxiter(50) == 10000


def test_de_sequential_polish_chain():
    """DE followed by L-BFGS-B polish via sequential_optimize."""
    res = sequential_optimize(
        _rosen_fused,
        x0=np.array([0.0, 0.0]),
        stages=[
            {
                "solver": differential_evolution,
                "bounds": [(-5.0, 5.0), (-5.0, 5.0)],
                "maxiter": 50,
                "rng": 0,
                "progressbar": False,
            },
            {
                "solver": minimize,
                "method": "L-BFGS-B",
                "jac": True,
                "tol": 1e-12,
                "progressbar": False,
            },
        ],
        progressbar=False,
    )
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)
    # Polish should win
    assert res.best_stage == 1
    assert res.fun < 1e-10


def test_de_x0_seeds_population():
    """Passing x0 seeds the initial population."""
    res = differential_evolution(
        rosen,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        x0=np.array([1.0, 1.0]),
        maxiter=5,
        rng=0,
        progressbar=False,
    )
    # x0 is the known minimum; initial population contains it, so DE should be near it fast
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-3)


def test_de_returns_optimize_result():
    res = differential_evolution(
        rosen,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=20,
        rng=0,
        progressbar=False,
    )
    assert isinstance(res, OptimizeResult)
    assert hasattr(res, "x") and hasattr(res, "fun") and hasattr(res, "nit")
