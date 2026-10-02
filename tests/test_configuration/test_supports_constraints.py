from dataclasses import fields

import numpy as np

from better_optimize.configuration.supports_constraints import (
    COBYLAConfig,
    COBYQAConfig,
    SLSQPConfig,
    TrustConstrConfig,
)


def test_cobyla_reuses_the_base_tol_as_its_own_option():
    assert "tol" in {field.name for field in fields(COBYLAConfig)}
    assert COBYLAConfig().tol == 1e-4
    assert COBYLAConfig(tol=1e-9).optimizer_kwargs()["tol"] == 1e-9


def test_cobyla_counts_evaluations_not_iterations():
    assert COBYLAConfig().default_budget(1000) == 1000


def test_cobyla_uses_no_derivatives():
    assert not COBYLAConfig.uses_grad
    assert not COBYLAConfig.uses_hess


def test_cobyla_leaves_the_constraint_tolerance_for_scipy_to_resolve():
    assert "catol" not in COBYLAConfig().optimizer_kwargs()
    assert COBYLAConfig().optimizer_kwargs()["f_target"] == -np.inf


def test_slsqp_exposes_the_verbosity_option_scipy_does_not_document():
    assert SLSQPConfig().iprint == 1
    assert SLSQPConfig(tol=1e-9).optimizer_kwargs()["ftol"] == 1e-9


def test_trust_constr_spreads_tol_across_three_tolerances():
    options = TrustConstrConfig(tol=1e-12).optimizer_kwargs()

    assert options["xtol"] == 1e-12
    assert options["gtol"] == 1e-12
    assert options["barrier_tol"] == 1e-12


def test_trust_constr_is_the_only_constrained_method_taking_second_order_information():
    assert TrustConstrConfig.uses_hess
    assert TrustConstrConfig.uses_hessp
    assert not SLSQPConfig.uses_hess


def test_cobyqa_uses_no_derivatives():
    assert not COBYQAConfig.uses_grad
    assert not COBYQAConfig.uses_hess


def test_cobyqa_routes_tol_to_the_final_trust_region_radius():
    assert COBYQAConfig(tol=1e-9).optimizer_kwargs()["final_tr_radius"] == 1e-9


def test_cobyqa_leaves_both_budgets_for_scipy_to_resolve():
    """Its two budgets scale differently, at ``500 * n`` evaluations and ``1000 * n``
    iterations, which one `default_budget` cannot express."""
    options = COBYQAConfig().optimizer_kwargs(n=10)

    assert "maxfev" not in options
    assert "maxiter" not in options
    assert COBYQAConfig().evaluation_budget(10) == 500 * 10


def test_a_requested_cobyqa_budget_still_reaches_scipy():
    assert COBYQAConfig(maxfev=7).optimizer_kwargs(n=10)["maxfev"] == 7
