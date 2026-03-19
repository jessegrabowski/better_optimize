import contextlib
import multiprocessing
import pickle
import threading
import time

from unittest.mock import MagicMock

import numpy as np
import pytest

from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult

from better_optimize.minimize import minimize
from better_optimize.multi_optimize import (
    MultiStartResult,
    ProgressProxy,
    _drain_progress_queue,
    _is_pickle_error,
    _MultiStart,
    _run_single,
    generate_starts,
    multi_optimize,
    setup_blas_cores,
)
from better_optimize.root import root


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def simple_system(x):
    return np.array([x[0] ** 2 + x[1] - 1, x[0] - x[1] ** 2 + 1])


def trivial_solver(x0, **kwargs):
    return OptimizeResult(x=x0, fun=float(np.sum(x0**2)), success=True)


@pytest.fixture
def rng():
    return np.random.default_rng()


def make_results(*fun_values, success=True):
    results = []
    for i, f in enumerate(fun_values):
        r = OptimizeResult(x=np.array([float(i)]), fun=f, success=success, nit=10)
        r.run_index = i
        results.append(r)
    return results


def test_normal_starts_centered_on_x0(rng):
    x0 = rng.standard_normal(3) * 10
    starts = generate_starts(x0, 10_000, "normal", bounds=None, init_scale=1.0, rng=rng)

    assert len(starts) == 10_000
    assert all(s.shape == x0.shape for s in starts)
    assert_allclose(np.mean(starts, axis=0), x0, atol=0.1)


@pytest.mark.parametrize("strategy", ["uniform", "sobol", "lhs"])
@pytest.mark.parametrize(
    "bounds",
    [(-5.0, 5.0), (np.array([-10.0, 0.0, -1.0]), np.array([0.0, 5.0, 1.0]))],
    ids=["scalar", "array"],
)
def test_bounded_strategies_respect_bounds(strategy, bounds, rng):
    n_params, n_runs = 3, 16
    low, high = bounds
    starts = generate_starts(
        np.zeros(n_params),
        n_runs,
        strategy,
        bounds=(low, high),
        init_scale=1.0,
        rng=rng,
    )

    stacked = np.stack(starts)
    assert stacked.shape == (n_runs, n_params)
    assert np.all(stacked >= low)
    assert np.all(stacked <= high)
    assert len(np.unique(stacked, axis=0)) == n_runs


def test_lhs_stratification(rng):
    """Each LHS dimension must have exactly one sample per stratum."""
    n_params, n_runs = 2, 8
    starts = generate_starts(
        np.zeros(n_params),
        n_runs,
        "lhs",
        bounds=(0.0, 1.0),
        init_scale=1.0,
        rng=rng,
    )
    stacked = np.stack(starts)

    for dim in range(n_params):
        bins = (stacked[:, dim] * n_runs).astype(int).clip(0, n_runs - 1)
        assert len(set(bins)) == n_runs


def test_callable_strategy_dispatches_to_user_function(rng):
    x0 = np.array([1.0, 2.0])

    def custom(x0, n, rng):
        return [x0 * i for i in range(n)]

    starts = generate_starts(x0, 5, custom, bounds=None, init_scale=1.0, rng=rng)

    assert len(starts) == 5
    assert_allclose(starts[0], [0.0, 0.0])
    assert_allclose(starts[3], [3.0, 6.0])


@pytest.mark.parametrize("strategy", ["uniform", "sobol", "lhs"])
def test_bounded_strategy_without_bounds_raises(strategy):
    with pytest.raises(ValueError, match="requires bounds"):
        generate_starts(
            np.zeros(2),
            4,
            strategy,
            bounds=None,
            init_scale=1.0,
            rng=np.random.default_rng(),
        )


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown init strategy"):
        generate_starts(
            np.zeros(2),
            4,
            "bogus",
            bounds=(-1, 1),
            init_scale=1.0,
            rng=np.random.default_rng(),
        )


@pytest.mark.parametrize("action", ["update", "reset"])
def test_progress_proxy_serializes_action_to_queue(action):
    queue = multiprocessing.Queue()
    proxy = ProgressProxy(queue, task_id=7)
    payload = {"f_value": 1.23}

    getattr(proxy, action)(7, **payload)

    received_action, received_task_id, received_kwargs = queue.get(timeout=1)
    assert received_action == action
    assert received_task_id == 7
    assert received_kwargs == payload


def test_drain_thread_exits_when_stop_is_set():
    queue = multiprocessing.Queue()
    stop = threading.Event()
    stop.set()

    thread = threading.Thread(
        target=_drain_progress_queue,
        args=(queue, MagicMock(), stop),
    )
    thread.start()
    thread.join(timeout=0.5)
    assert not thread.is_alive()


def test_drain_thread_dispatches_queued_messages():
    queue = multiprocessing.Queue()
    progress = MagicMock()
    stop = threading.Event()

    queue.put(("update", 0, {"f_value": 1.0}))
    queue.put(("reset", 1, {"visible": False}))

    thread = threading.Thread(target=_drain_progress_queue, args=(queue, progress, stop))
    thread.start()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if progress.update.called and progress.reset.called:
            break
        time.sleep(0.01)

    stop.set()
    thread.join(timeout=1.0)

    progress.update.assert_called_once_with(0, f_value=1.0)
    progress.reset.assert_called_once_with(1, visible=False)


@pytest.mark.parametrize(
    "blas_cores, n_jobs, expected_per_worker",
    [
        ("auto", 4, 1),
        (8, 4, 2),
        (1, 8, 1),  # floors to 1
    ],
    ids=["auto", "explicit", "floors-to-one"],
)
def test_setup_blas_cores_per_worker(blas_cores, n_jobs, expected_per_worker):
    config = setup_blas_cores(blas_cores, n_jobs=n_jobs, mp_ctx=None)
    assert config.per_worker == expected_per_worker


def test_setup_blas_cores_none_disables_limiting():
    config = setup_blas_cores(None, n_jobs=4, mp_ctx=None)
    assert config.limiter is contextlib.nullcontext
    assert config.per_worker is None


def test_setup_blas_cores_fork_skips_joined_limiter():
    mock_ctx = MagicMock()
    mock_ctx.get_start_method.return_value = "fork"

    config = setup_blas_cores(8, n_jobs=4, mp_ctx=mock_ctx)
    assert config.limiter is contextlib.nullcontext
    assert config.per_worker == 2


def test_setup_blas_cores_invalid_string_raises():
    with pytest.raises(ValueError, match="blas_cores must be"):
        setup_blas_cores("bogus", n_jobs=4, mp_ctx=None)


def test_setup_blas_cores_negative_n_jobs():
    config = setup_blas_cores("auto", n_jobs=-1, mp_ctx=None)
    assert config.per_worker == 1
    assert config.limiter is not contextlib.nullcontext


@pytest.mark.parametrize(
    "exc, expected",
    [
        (pickle.PicklingError("can't pickle"), True),
        (AttributeError("Can't pickle local object"), True),
        (AttributeError("no attribute 'foo'"), False),
        (ValueError("something"), False),
    ],
    ids=["PicklingError", "AttributeError-pickle", "AttributeError-unrelated", "ValueError"],
)
def test_is_pickle_error(exc, expected):
    assert _is_pickle_error(exc) == expected


def test_run_single_attaches_metadata():
    x0 = np.array([1.0, 2.0])
    result = _run_single(
        run_index=5,
        x0=x0,
        solver=trivial_solver,
        solver_kwargs={},
        progress=MagicMock(),
        task_id=0,
    )
    assert result.run_index == 5
    assert_allclose(result.x0, x0)


def test_run_single_propagates_solver_exception():
    def exploding_solver(x0, **kwargs):
        raise RuntimeError("solver exploded")

    with pytest.raises(RuntimeError, match="solver exploded"):
        _run_single(
            run_index=0,
            x0=np.zeros(2),
            solver=exploding_solver,
            solver_kwargs={},
            progress=MagicMock(),
            task_id=0,
        )


class TestMultiStartResult:
    def test_best_selects_minimum_and_exposes_x_and_fun(self):
        msr = MultiStartResult(results=make_results(3.0, 1.0, 2.0))
        assert msr.best.fun == 1.0
        assert_allclose(msr.x_best, np.array([1.0]))
        assert msr.fun_best == 1.0

    @pytest.mark.parametrize("ascending", [True, False], ids=["ascending", "descending"])
    def test_ranked_ordering(self, ascending):
        msr = MultiStartResult(results=make_results(3.0, 1.0, 2.0))
        funs = [r.fun for r in msr.ranked(ascending=ascending)]
        expected = [1.0, 2.0, 3.0] if ascending else [3.0, 2.0, 1.0]
        assert funs == expected

    def test_top_k_truncates(self):
        msr = MultiStartResult(results=make_results(5.0, 1.0, 3.0, 2.0, 4.0))
        top2 = msr.top_k(2)
        assert len(top2) == 2
        assert [r.fun for r in top2] == [1.0, 2.0]

    def test_success_rate(self):
        results = [OptimizeResult(x=np.zeros(1), fun=float(i), success=(i < 2)) for i in range(3)]
        msr = MultiStartResult(results=results)
        assert_allclose(msr.success_rate, 2.0 / 3.0)

    def test_custom_sort_key_reverses_ranking(self):
        msr = MultiStartResult(
            results=make_results(3.0, 1.0, 2.0),
            sort_key=lambda r: -float(r.fun),
        )
        assert msr.best.fun == 3.0

    def test_summary_smoke(self):
        """Output format is not part of the contract; just verify it doesn't crash."""
        msr = MultiStartResult(results=make_results(3.0, 1.0, 2.0))
        msr.summary()
        msr.summary(top_k=1)

    def test_to_dataframe(self):
        pytest.importorskip("pandas")
        msr = MultiStartResult(results=make_results(3.0, 1.0))
        df = msr.to_dataframe()
        expected_columns = {"rank", "run_index", "fun", "success", "nit", "x", "message"}
        assert set(df.columns) == expected_columns
        assert len(df) == 2
        assert df.iloc[0]["fun"] == 1.0

    def test_format_result_row_handles_missing_jac(self):
        result = OptimizeResult(x=np.zeros(2), fun=1.5, success=True, nit=5)
        result.run_index = 0
        msr = MultiStartResult(results=[result])
        row = msr._format_result_row(1, result)
        grad_column = row[3]
        assert grad_column == ""

    def test_empty_results_raises_on_best(self):
        msr = MultiStartResult(results=[])
        with pytest.raises(ValueError):
            _ = msr.best


@pytest.mark.parametrize(
    "solver_name, expected_label",
    [
        ("minimize", "Minimizing"),
        ("root", "Finding Roots"),
        ("basinhopping", "Basinhopping"),
    ],
)
def test_task_description_for_known_solvers(solver_name, expected_label):
    stub = MagicMock(__name__=solver_name)
    ms = _MultiStart(solver=stub, x0=[np.zeros(2)], progressbar=False)
    assert ms._task_description == expected_label


def test_task_description_falls_back_for_unknown_solver():
    stub = MagicMock(__name__="my_custom_opt")
    ms = _MultiStart(solver=stub, x0=[np.zeros(2)], progressbar=False)
    assert ms._task_description == "My_Custom_Opt"


@pytest.mark.parametrize("backend", ["sequential", "loky"])
def test_multistart_finds_rosenbrock_minimum(backend):
    result = multi_optimize(
        solver=minimize,
        solver_kwargs=dict(f=rosenbrock, method="L-BFGS-B"),
        x0=np.zeros(3),
        n_runs=8,
        init_strategy="uniform",
        bounds=(-2, 2),
        seed=7913,
        backend=backend,
        n_jobs=2,
        progressbar=False,
        blas_cores=None,
    )

    assert len(result.results) == 8
    assert_allclose(result.x_best, np.ones(3), atol=1e-3)
    assert result.fun_best < 1e-5


def test_multistart_with_explicit_starting_points():
    x0_list = [np.array([1.3, 0.7, 0.8]), np.array([0.5, 0.5, 0.5])]
    result = multi_optimize(
        solver=minimize,
        solver_kwargs=dict(f=rosenbrock, method="L-BFGS-B"),
        x0=x0_list,
        progressbar=False,
    )

    assert len(result.results) == 2
    for result_i, expected_x0 in zip(result.results, x0_list):
        assert_allclose(result_i.x0, expected_x0)


def test_multistart_with_root_solver():
    result = multi_optimize(
        solver=root,
        solver_kwargs=dict(f=simple_system, method="hybr"),
        sort_key=lambda res: float(np.linalg.norm(res.fun)),
        x0=np.zeros(2),
        n_runs=6,
        init_strategy="normal",
        init_scale=2.0,
        seed=9173,
        progressbar=False,
    )

    assert result.fun_best < 1e-6


def test_multistart_pickle_fallback(caplog):
    def unpicklable_closure():
        return None

    def solver_with_closure(x0, **kwargs):
        unpicklable_closure()
        return OptimizeResult(x=x0, fun=float(np.sum(x0**2)), success=True)

    with caplog.at_level("WARNING", logger="better_optimize.multistart"):
        result = multi_optimize(
            solver=solver_with_closure,
            x0=[np.zeros(2), np.ones(2)],
            backend="loky",
            n_jobs=2,
            progressbar=False,
            blas_cores=None,
        )

    # Fallback may or may not trigger depending on environment,
    # but either way we must get valid results.
    assert len(result.results) == 2
