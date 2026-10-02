import numpy as np
import pytest

from better_optimize.configuration.root_direct import DFSaneConfig, HybrConfig, LMConfig

ALL = [HybrConfig, LMConfig, DFSaneConfig]


@pytest.mark.parametrize(
    "config, uses_jac", [(HybrConfig, True), (LMConfig, True), (DFSaneConfig, False)]
)
def test_only_the_minpack_methods_consume_a_jacobian(config, uses_jac):
    assert config.uses_jac is uses_jac


@pytest.mark.parametrize("config, budget", [(HybrConfig, "maxfev"), (LMConfig, "maxiter")])
def test_the_minpack_methods_leave_their_budget_to_scipy(config, budget):
    """scipy halves it to ``100 * (n + 1)`` when a jacobian function is supplied, and
    reaches that branch only when the option is absent."""
    assert budget not in config().optimizer_kwargs(n=10)
    assert config().default_budget(10) == 200 * 11
    assert config().evaluation_budget(10) == 200 * 11


@pytest.mark.parametrize("config, budget", [(HybrConfig, "maxfev"), (LMConfig, "maxiter")])
def test_a_requested_minpack_budget_still_reaches_scipy(config, budget):
    assert config(**{budget: 7}).optimizer_kwargs(n=10)[budget] == 7


def test_df_sane_budgets_a_flat_count():
    """It is the one root method whose default does not scale with the dimension."""
    assert DFSaneConfig().default_budget(2) == DFSaneConfig().default_budget(500) == 1000


@pytest.mark.parametrize(
    "config, filled", [(HybrConfig, "xtol"), (LMConfig, "xtol"), (DFSaneConfig, "ftol")]
)
def test_tol_fills_the_one_tolerance_scipy_routes_it_to(config, filled):
    options = config(tol=1e-11).optimizer_kwargs()

    assert options[filled] == 1e-11


def test_lm_keeps_its_other_tolerances_where_tol_does_not_reach():
    options = LMConfig(tol=1e-11).optimizer_kwargs()

    assert options["ftol"] == 1.49012e-08
    assert options["gtol"] == 0.0


def test_lms_iteration_budget_caps_evaluations():
    """scipy forwards `maxiter` to MINPACK as `maxfev`, so the name is the only thing
    about it that says iterations."""
    assert LMConfig(maxiter=7).evaluation_budget(10) == 7


@pytest.mark.parametrize("config", ALL)
def test_a_budget_the_caller_left_unset_is_omitted(config):
    """Every root method spells its budget differently, so the assertion has to read the
    name each one declares rather than guess at both."""
    budgets = set(config._budget_options())

    assert budgets
    assert not budgets & set(config().optimizer_kwargs())


def test_an_array_option_is_not_shared_between_runs():
    scales = np.ones(3)
    config = HybrConfig(diag=scales)

    config.optimizer_kwargs()["diag"][0] = 99.0

    assert config.diag[0] == 1.0
