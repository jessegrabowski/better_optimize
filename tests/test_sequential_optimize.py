import logging

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from scipy.optimize import OptimizeResult, rosen, rosen_der

from better_optimize import minimize, sequential_optimize
from better_optimize.configuration import (
    BasinHoppingConfig,
    DifferentialEvolutionConfig,
    LBFGSBConfig,
    NelderMeadConfig,
    OptimizerConfig,
)
from better_optimize.sequential_optimize import SequentialResult, _classify
from better_optimize.utilities import LRUCache1


def _rosen_fused(x):
    return rosen(x), rosen_der(x)


def _quadratic_fused(x):
    return float(np.sum(x * x)), 2.0 * x


def test_two_stage_chain():
    res = sequential_optimize(
        _rosen_fused,
        x0=np.array([-1.2, 1.0]),
        stages=[
            {"solver": minimize, "method": "nelder-mead", "tol": 1e-4, "progressbar": False},
            {
                "solver": minimize,
                "method": "L-BFGS-B",
                "jac": True,
                "tol": 1e-10,
                "progressbar": False,
            },
        ],
        progressbar=False,
    )
    assert res.success
    assert res.best_stage == 1
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)
    assert res.fun < 1e-8


def test_x0_forwarding_uses_best_so_far():
    captured = []

    def recording_solver(f, x0, **kwargs):
        captured.append(np.asarray(x0, dtype=np.float64).copy())
        # Pretend we converged to x0 shifted by a fixed amount (finite, usable)
        new_x = x0 - 0.5
        return OptimizeResult(x=new_x, fun=float(np.sum(new_x**2)), success=True, nit=1)

    x0 = np.array([2.0, 2.0])
    sequential_optimize(
        _quadratic_fused,
        x0=x0,
        stages=[
            {"solver": recording_solver, "name": "s0"},
            {"solver": recording_solver, "name": "s1"},
            {"solver": recording_solver, "name": "s2"},
        ],
        progressbar=False,
    )

    np.testing.assert_allclose(captured[0], [2.0, 2.0])
    np.testing.assert_allclose(captured[1], [1.5, 1.5])
    np.testing.assert_allclose(captured[2], [1.0, 1.0])


def test_stage_override_x0_wins():
    captured = []

    def recorder(f, x0, **kwargs):
        captured.append(np.asarray(x0, dtype=np.float64).copy())
        return OptimizeResult(x=x0, fun=float(np.sum(x0**2)), success=True, nit=1)

    sequential_optimize(
        _quadratic_fused,
        x0=np.array([5.0, 5.0]),
        stages=[
            {"solver": recorder},
            {"solver": recorder, "x0": np.array([0.3, 0.3])},
        ],
        progressbar=False,
    )

    np.testing.assert_allclose(captured[0], [5.0, 5.0])
    np.testing.assert_allclose(captured[1], [0.3, 0.3])


def test_hard_failure_on_failure_stop():
    def crashing(f, x0, **kwargs):
        raise RuntimeError("boom")

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([0.5, 0.5]),
        stages=[
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
            {"solver": crashing, "name": "boom"},
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
        ],
        progressbar=False,
        on_failure="stop",
    )

    assert not res.success
    assert len(res.stage_results) == 2
    assert res.stage_results[1].classification == "hard"
    assert "raised RuntimeError" in res.stage_results[1].message
    # best x comes from stage 0, the last usable
    assert res.best_stage == 0


def test_hard_failure_on_failure_continue():
    def crashing(f, x0, **kwargs):
        raise RuntimeError("still boom")

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([1.0, 1.0]),
        stages=[
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
            {"solver": crashing, "name": "boom"},
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
        ],
        progressbar=False,
        on_failure="continue",
    )

    assert not res.success  # hard failure encountered
    assert len(res.stage_results) == 3
    assert res.stage_results[1].classification == "hard"
    # Stage 2 ran and reached the minimum
    assert res.fun < 1e-8


def test_stage_returns_nan_fun_but_finite_x_is_recovered():
    def returns_nan_fun(f, x0, **kwargs):
        new_x = x0 - 0.1
        return OptimizeResult(x=new_x, fun=float("nan"), success=True, nit=1)

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([2.0]),
        stages=[{"solver": returns_nan_fun}],
        progressbar=False,
    )

    # fun was NaN but x was finite; driver re-evaluates via f_cached and recovers
    assert res.stage_results[0].classification == "usable"
    assert np.isfinite(res.fun)
    assert res.fun == pytest.approx(1.9**2)


def test_stage_regression_does_not_advance():
    captured_x0s = []

    def make_fake_solver(new_x, new_fun):
        def solver(f, x0, **kwargs):
            captured_x0s.append(np.asarray(x0, dtype=np.float64).copy())
            return OptimizeResult(x=new_x, fun=new_fun, success=True, nit=1)

        return solver

    good_x = np.array([0.1, 0.1])
    bad_x = np.array([3.0, 3.0])

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([1.0, 1.0]),
        stages=[
            {"solver": make_fake_solver(good_x, 0.02), "name": "good"},
            {"solver": make_fake_solver(bad_x, 18.0), "name": "regress"},
            {"solver": make_fake_solver(good_x * 0.5, 0.005), "name": "polish"},
        ],
        progressbar=False,
    )

    # Stage 2 was soft-failure (finite but worse than stage 0's 0.02)
    assert res.stage_results[1].classification == "soft"
    # Stage 3 received stage 0's x (best-so-far), NOT stage 1's
    np.testing.assert_allclose(captured_x0s[2], good_x)
    # Final best from stage 2 which further improved
    assert res.best_stage == 2


def test_solver_success_false_with_improvement_is_usable():
    def unconverged(f, x0, **kwargs):
        new_x = np.full_like(x0, 0.001)
        return OptimizeResult(x=new_x, fun=float(np.sum(new_x**2)), success=False, nit=999)

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([5.0, 5.0]),
        stages=[{"solver": unconverged}],
        progressbar=False,
    )

    assert res.stage_results[0].classification == "usable"
    assert res.best_stage == 0
    np.testing.assert_allclose(res.x, [0.001, 0.001])


def test_cache_shared_across_stages():
    call_count = [0]

    def expensive_fused(x):
        call_count[0] += 1
        return float(np.sum(x * x)), 2.0 * x

    def touch_then_return(f, x0, **kwargs):
        # Evaluate f at x0 to check for a cache hit at stage handoff
        f(x0)
        return OptimizeResult(x=x0, fun=float(np.sum(x0**2)), success=True, nit=1)

    sequential_optimize(
        expensive_fused,
        x0=np.array([1.0, 2.0]),
        stages=[
            {"solver": touch_then_return, "name": "s0"},
            {"solver": touch_then_return, "name": "s1"},
            {"solver": touch_then_return, "name": "s2"},
        ],
        progressbar=False,
    )

    # Driver sniff calls expensive_fused once; stage 0 evaluates at x0 (cache hit
    # from sniff); stages 1 and 2 evaluate at same x (all hits). Without the cache
    # each stage would add a miss, giving >= 4 calls.
    assert call_count[0] <= 2


def test_nested_progress_bar(monkeypatch):
    updates = []

    def tracking_add_task(self, *args, **kwargs):
        name = kwargs.get("description", "")
        task_id = _real_add_task(self, *args, **kwargs)
        updates.append(("add_task", str(name), int(task_id)))
        return task_id

    import better_optimize.utilities as _util

    _real_add_task = _util.ToggleableProgress.add_task
    monkeypatch.setattr(_util.ToggleableProgress, "add_task", tracking_add_task)

    sequential_optimize(
        _quadratic_fused,
        x0=np.array([1.0, 1.0]),
        stages=[
            {"solver": minimize, "method": "L-BFGS-B", "jac": True},
            {"solver": minimize, "method": "L-BFGS-B", "jac": True},
        ],
        progressbar=True,
    )

    # Expect one pre-registered task per stage, labeled by method with .1/.2
    # uniquification since both stages use L-BFGS-B.
    descriptions = [d for _, d, _ in updates]
    assert "L-BFGS-B.1" in descriptions
    assert "L-BFGS-B.2" in descriptions


def test_empty_stages_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sequential_optimize(_quadratic_fused, x0=np.array([0.0]), stages=[])


def test_missing_solver_key_raises():
    with pytest.raises(ValueError, match="missing required 'solver'"):
        sequential_optimize(
            _quadratic_fused,
            x0=np.array([0.0]),
            stages=[{"name": "nope"}],
        )


def test_top_level_f_cannot_be_per_stage():
    with pytest.raises(ValueError, match="top-level"):
        sequential_optimize(
            _quadratic_fused,
            x0=np.array([0.0]),
            stages=[{"solver": minimize, "f": lambda x: x**2}],
        )


def test_x0_absent_solver_raises_at_setup():
    def no_x0_solver(**kwargs):
        return OptimizeResult(x=np.zeros(1), fun=0.0, success=True)

    # Rebind the signature so our inspector sees no x0 and no **kwargs
    def no_x0_strict(f):
        return OptimizeResult(x=np.zeros(1), fun=0.0, success=True)

    with pytest.raises(ValueError, match="does not accept 'x0'"):
        sequential_optimize(
            _quadratic_fused,
            x0=np.array([1.0]),
            stages=[{"solver": no_x0_strict}],
        )


def test_x0_suppressed_with_explicit_none():
    captured = []

    def no_x0_strict(f):
        captured.append("called")
        return OptimizeResult(x=np.zeros(1), fun=0.0, success=True)

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([1.0]),
        stages=[{"solver": no_x0_strict, "x0": None}],
        progressbar=False,
    )
    assert captured == ["called"]
    assert res.best_stage == 0


def test_final_best_is_earlier_stage_when_later_regresses():
    def better(f, x0, **kwargs):
        return OptimizeResult(x=np.array([0.0]), fun=0.0, success=True, nit=1)

    def worse(f, x0, **kwargs):
        return OptimizeResult(x=np.array([5.0]), fun=25.0, success=True, nit=1)

    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([2.0]),
        stages=[
            {"solver": better, "name": "better"},
            {"solver": worse, "name": "worse"},
        ],
        progressbar=False,
    )

    assert res.best_stage == 0
    assert res.fun == 0.0
    assert res.stage_results[1].classification == "soft"


def test_args_forwarded_to_all_stages():
    captured_args = []

    def objective_with_args(x, *extra):
        return float(np.sum(x * x)), 2.0 * x

    def recorder(f, x0, args=None, **kwargs):
        captured_args.append(args)
        return OptimizeResult(x=x0, fun=float(np.sum(x0**2)), success=True, nit=1)

    sentinel = (3.14, "hello")
    sequential_optimize(
        objective_with_args,
        x0=np.array([1.0]),
        stages=[
            {"solver": recorder, "name": "a"},
            {"solver": recorder, "name": "b"},
        ],
        args=sentinel,
        progressbar=False,
    )
    assert captured_args == [sentinel, sentinel]


def test_all_stages_hard_fail_returns_x0():
    def crash(f, x0, **kwargs):
        raise RuntimeError("always fails")

    x0 = np.array([1.5, 2.5])
    res = sequential_optimize(
        _quadratic_fused,
        x0=x0,
        stages=[
            {"solver": crash, "name": "a"},
            {"solver": crash, "name": "b"},
        ],
        progressbar=False,
        on_failure="continue",
    )
    assert not res.success
    assert res.best_stage == -1
    np.testing.assert_array_equal(res.x, x0)
    assert all(r.classification == "hard" for r in res.stage_results)


def test_classify_helper_direct():
    f_cached = LRUCache1(_quadratic_fused, f_returns_list=True, dtype=np.float64)

    # Hard: NaN in x
    res_bad_x = OptimizeResult(x=np.array([np.nan, 1.0]), fun=0.0)
    assert _classify(res_bad_x, best_fun=None, f_cached=f_cached) == "hard"

    # Hard: NaN fun, unrecoverable (we force by passing huge x that still computes fine)
    # Actually for this quadratic f always returns finite, so we test recovery:
    res_nan_fun = OptimizeResult(x=np.array([2.0, 0.0]), fun=float("nan"))
    assert _classify(res_nan_fun, best_fun=None, f_cached=f_cached) == "usable"
    assert res_nan_fun.fun == pytest.approx(4.0)

    # Usable: finite, improves
    res_ok = OptimizeResult(x=np.array([1.0]), fun=1.0)
    assert _classify(res_ok, best_fun=5.0, f_cached=f_cached) == "usable"

    # Soft: finite, regresses
    res_regress = OptimizeResult(x=np.array([3.0]), fun=9.0)
    assert _classify(res_regress, best_fun=1.0, f_cached=f_cached) == "soft"


def test_a_stage_that_is_neither_a_config_nor_a_dict_is_rejected():
    with pytest.raises(TypeError, match="stage 0: expected a configuration or a dict"):
        sequential_optimize(
            rosen, np.array([1.0, 1.0]), stages=[("not", "a", "dict")], progressbar=False
        )


def test_result_is_sequential_result_with_properties():
    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([2.0, 2.0]),
        stages=[{"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False}],
        progressbar=False,
    )
    assert isinstance(res, SequentialResult)
    assert isinstance(res.best, OptimizeResult)
    np.testing.assert_allclose(res.x, res.best.x)
    assert res.fun == pytest.approx(float(res.best.fun))


def test_to_dataframe_shape():
    res = sequential_optimize(
        _quadratic_fused,
        x0=np.array([1.0]),
        stages=[
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
            {"solver": minimize, "method": "L-BFGS-B", "jac": True, "progressbar": False},
        ],
        progressbar=False,
    )
    df = res.to_dataframe()
    assert len(df) == 2
    assert {"stage", "solver", "classification", "fun", "success", "best_stage"}.issubset(
        df.columns
    )
    assert df["best_stage"].sum() == 1


def test_a_config_stage_runs_without_naming_a_solver():
    """A configuration already says which solver runs it, so the stage needs no callable."""
    result = sequential_optimize(
        rosen,
        np.array([-1.2, 1.0]),
        stages=[NelderMeadConfig(maxiter=300), LBFGSBConfig()],
        progressbar=False,
    )

    assert [stage.solver_name for stage in result.stage_results] == ["nelder-mead", "L-BFGS-B"]
    assert result.best.fun < 1e-8


def test_a_global_optimizer_config_is_a_stage_like_any_other():
    result = sequential_optimize(
        rosen,
        np.array([-1.2, 1.0]),
        stages=[BasinHoppingConfig(niter=2, rng=0), LBFGSBConfig()],
        progressbar=False,
    )

    assert [stage.solver_name for stage in result.stage_results] == ["basinhopping", "L-BFGS-B"]
    assert result.best.fun < 1e-8


def test_config_and_dict_stages_mix_in_one_chain():
    result = sequential_optimize(
        rosen,
        np.array([-1.2, 1.0]),
        stages=[{"solver": minimize, "method": "nelder-mead", "name": "coarse"}, LBFGSBConfig()],
        progressbar=False,
    )

    assert [stage.solver_name for stage in result.stage_results] == ["coarse", "L-BFGS-B"]
    assert result.best.fun < 1e-8


def test_a_dict_stage_naming_a_method_needs_no_solver():
    result = sequential_optimize(
        rosen,
        np.array([-1.2, 1.0]),
        stages=[{"method": "nelder-mead", "maxiter": 300}, {"method": "L-BFGS-B"}],
        progressbar=False,
    )

    assert [stage.solver_name for stage in result.stage_results] == ["nelder-mead", "L-BFGS-B"]
    assert result.best.fun < 1e-8


def test_a_stage_searching_a_region_is_refused_rather_than_given_a_guessed_one():
    """A box the caller never wrote constrains the answer, so choosing one silently would
    report a boundary-pinned result as a success."""
    with pytest.raises(ValueError, match="searches a bounded region"):
        sequential_optimize(
            rosen,
            np.array([-1.2, 1.0]),
            stages=[DifferentialEvolutionConfig(maxiter=5)],
            progressbar=False,
        )


def test_an_opted_in_width_reaches_the_derived_bounds(caplog):
    """The width has to reach the solver, and the box it produced has to be recoverable
    from the run that used it."""
    with caplog.at_level(logging.INFO, logger="better_optimize.sequential_optimize"):
        result = sequential_optimize(
            rosen,
            np.array([-1.2, 1.0]),
            stages=[DifferentialEvolutionConfig(maxiter=40, rng=0)],
            progressbar=False,
            derived_bounds_width=3.0,
        )

    assert result.best.fun < 1e-8
    assert "x +/- 3 around" in caplog.text


def test_a_stage_with_no_x_to_build_bounds_around_is_refused():
    with pytest.raises(ValueError, match="can only build them around an x0"):
        sequential_optimize(
            rosen,
            np.array([-1.2, 1.0]),
            stages=[{"method": DifferentialEvolutionConfig(maxiter=5), "x0": None}],
            progressbar=False,
            derived_bounds_width=1.0,
        )


def test_any_stage_declaring_requires_bounds_is_given_them():
    """The driver reads the flag rather than naming differential evolution, so a solver
    added later gets the same treatment."""
    received = {}

    def record_bounds(x0, bounds):
        received["bounds"] = bounds
        return OptimizeResult(x=np.asarray(x0), fun=0.0, success=True)

    @dataclass(frozen=True, eq=False)
    class RegionSearch(OptimizerConfig):
        _iteration_options: ClassVar[tuple[str, ...]] = ()
        requires_bounds: ClassVar[bool] = True

        @property
        def method_name(self) -> str:
            return "region-search"

        def solver_function(self):
            return record_bounds

        def build_solver_kwargs(self, problem):
            return {"x0": problem.x0, "bounds": problem.solver_kwargs["bounds"]}

    sequential_optimize(
        rosen,
        np.array([2.0, -3.0]),
        stages=[RegionSearch()],
        progressbar=False,
        derived_bounds_width=0.5,
    )

    assert received["bounds"] == [(1.5, 2.5), (-3.5, -2.5)]
