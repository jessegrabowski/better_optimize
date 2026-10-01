import inspect

from dataclasses import fields

import numpy as np
import pytest

from scipy.optimize import rosen, rosen_der, rosen_hess

from better_optimize import differential_evolution, minimize
from better_optimize.basinhopping import basinhopping
from better_optimize.configuration import (
    BasinHoppingConfig,
    DifferentialEvolutionConfig,
    LBFGSBConfig,
    NelderMeadConfig,
    NewtonCGConfig,
    TrustNCGConfig,
)

BOUNDS = [(-2.0, 2.0), (-2.0, 2.0)]

# What each solver takes beside its options: the problem itself, and how to report on it.
NOT_OPTIONS = {
    basinhopping: {"func", "x0", "minimizer_kwargs", "callback", "progressbar", "verbose"},
    differential_evolution: {
        "f",
        "x0",
        "bounds",
        "args",
        "constraints",
        "callback",
        "progressbar",
        "progress_task",
        "verbose",
    },
}
CONFIGS = [
    (BasinHoppingConfig, basinhopping),
    (DifferentialEvolutionConfig, differential_evolution),
]
IDS = ["basinhopping", "differential_evolution"]


def declared_options(config_class):
    return {field.name for field in fields(config_class)} - config_class._excluded


@pytest.mark.parametrize("config_class, solver", CONFIGS, ids=IDS)
def test_the_fields_are_exactly_the_options_the_solver_takes(config_class, solver):
    parameters = set(inspect.signature(solver).parameters)

    assert declared_options(config_class) == parameters - NOT_OPTIONS[solver]


@pytest.mark.parametrize("config_class, solver", CONFIGS, ids=IDS)
def test_the_defaults_are_the_solvers_own(config_class, solver):
    """Except the budgets, which the config owns so that `default_budget` can scale them."""
    parameters = inspect.signature(solver).parameters
    config = config_class()

    for name in declared_options(config_class) - set(config_class._budget_options()):
        assert getattr(config, name) == parameters[name].default, name


X0 = np.array([-1.2, 1.0])


@pytest.mark.parametrize(
    "minimizer_config, uses_grad, uses_hess, uses_hessp",
    [
        (NelderMeadConfig(), False, False, False),
        (LBFGSBConfig(), True, False, False),
        (NewtonCGConfig(), True, True, True),
    ],
    ids=["nelder-mead", "l-bfgs-b", "newton-cg"],
)
def test_the_inner_minimizer_decides_which_derivatives_are_used(
    minimizer_config, uses_grad, uses_hess, uses_hessp
):
    config = BasinHoppingConfig(minimizer_config=minimizer_config)

    assert (config.uses_grad, config.uses_hess, config.uses_hessp) == (
        uses_grad,
        uses_hess,
        uses_hessp,
    )


def test_the_inner_config_is_not_emitted_as_an_option():
    """It reaches scipy as ``minimizer_kwargs["method"]``, not in the options."""
    assert "minimizer_config" not in BasinHoppingConfig().optimizer_kwargs(n=2)


def test_a_tolerance_belongs_on_the_inner_minimizer():
    """basinhopping has none of its own, so accepting one would discard it."""
    with pytest.raises(TypeError, match="no tolerance of its own"):
        BasinHoppingConfig(tol=1e-9)


def test_an_option_the_caller_left_unset_is_omitted():
    assert set(BasinHoppingConfig().optimizer_kwargs()) < set(
        BasinHoppingConfig(niter_success=3, rng=0).optimizer_kwargs()
    )


def test_a_dimension_fills_the_iteration_budget():
    config = BasinHoppingConfig()

    assert "niter" not in config.optimizer_kwargs()
    assert config.optimizer_kwargs(n=2)["niter"] == config.default_budget(2)


def test_minimize_dispatches_a_config_to_basinhopping():
    through_minimize = minimize(
        rosen, X0, method=BasinHoppingConfig(niter=3, rng=0), jac=rosen_der, progressbar=False
    )
    directly = basinhopping(
        rosen,
        X0,
        niter=3,
        rng=0,
        minimizer_kwargs={"method": LBFGSBConfig(), "jac": rosen_der},
        progressbar=False,
    )

    np.testing.assert_allclose(through_minimize.x, directly.x)


def test_the_inner_minimizer_budget_reaches_a_trust_region_method():
    """Trust-region methods have no evaluation-capping option, so a dropped ``maxiter``
    would leave them running to scipy's default rather than raising."""

    def solve(maxiter):
        return minimize(
            rosen,
            X0,
            method=BasinHoppingConfig(
                niter=2, rng=0, minimizer_config=TrustNCGConfig(maxiter=maxiter)
            ),
            jac=rosen_der,
            hess=rosen_hess,
            progressbar=False,
        )

    assert solve(maxiter=3).fun > solve(maxiter=500).fun


@pytest.mark.parametrize("dimension, expected", [(2, 1000), (5, 1000), (10, 2000), (50, 10000)])
def test_the_generation_budget_scales_with_the_problem_dimension(dimension, expected):
    assert DifferentialEvolutionConfig().default_budget(dimension) == expected
    assert DifferentialEvolutionConfig().optimizer_kwargs(n=dimension)["maxiter"] == expected


def test_a_requested_generation_budget_wins_over_the_default():
    assert DifferentialEvolutionConfig(maxiter=7).optimizer_kwargs(n=50)["maxiter"] == 7


def test_an_unknown_strategy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="strategy must be one of"):
        DifferentialEvolutionConfig(strategy="nonexistent")


def test_an_unknown_initializer_is_rejected_at_construction():
    with pytest.raises(ValueError, match="init must be one of"):
        DifferentialEvolutionConfig(init="not_a_method")


def test_a_callable_strategy_is_left_to_scipy():
    """scipy accepts a callable building the trial vector, and checks nothing about it."""

    def build_trial_vector(candidate, population):
        return population[candidate]

    assert DifferentialEvolutionConfig(strategy=build_trial_vector).strategy is build_trial_vector


def test_minimize_dispatches_a_config_to_differential_evolution():
    through_minimize = minimize(
        rosen,
        X0,
        method=DifferentialEvolutionConfig(maxiter=20, rng=0),
        bounds=BOUNDS,
        progressbar=False,
    )
    directly = differential_evolution(
        rosen, bounds=BOUNDS, x0=X0, maxiter=20, rng=0, progressbar=False
    )

    np.testing.assert_allclose(through_minimize.x, directly.x)


def test_differential_evolution_without_bounds_says_so():
    with pytest.raises(TypeError, match="bounds is required"):
        minimize(rosen, X0, method=DifferentialEvolutionConfig(), progressbar=False)


def test_a_derivative_passed_to_differential_evolution_says_so():
    """It cannot be forwarded, so dropping it silently would lose the caller's gradient."""
    with pytest.raises(TypeError, match="no derivative information"):
        minimize(
            rosen,
            X0,
            method=DifferentialEvolutionConfig(),
            bounds=BOUNDS,
            jac=rosen_der,
            progressbar=False,
        )
