from dataclasses import fields

import numpy as np

from better_optimize.configuration.supports_constraints import (
    COBYLAConfig,
    SLSQPConfig,
    TrustConstrConfig,
)


def test_cobyla_reuses_the_base_tol_as_its_own_option():
    assert "tol" in {field.name for field in fields(COBYLAConfig)}
    assert COBYLAConfig().tol == 1e-4
    assert COBYLAConfig(tol=1e-9).optimizer_kwargs["tol"] == 1e-9


def test_cobyla_counts_evaluations_not_iterations():
    assert COBYLAConfig().default_maxiter(1000) == 1000


def test_cobyla_uses_no_derivatives():
    assert not COBYLAConfig.uses_grad
    assert not COBYLAConfig.uses_hess


def test_cobyla_leaves_the_constraint_tolerance_for_scipy_to_resolve():
    assert COBYLAConfig().optimizer_kwargs["catol"] is None
    assert COBYLAConfig().optimizer_kwargs["f_target"] == -np.inf


def test_slsqp_exposes_the_verbosity_option_scipy_does_not_document():
    assert SLSQPConfig().iprint == 1
    assert SLSQPConfig(tol=1e-9).optimizer_kwargs["ftol"] == 1e-9


def test_trust_constr_spreads_tol_across_three_tolerances():
    options = TrustConstrConfig(tol=1e-12).optimizer_kwargs

    assert options["xtol"] == 1e-12
    assert options["gtol"] == 1e-12
    assert options["barrier_tol"] == 1e-12


def test_trust_constr_is_the_only_constrained_method_taking_second_order_information():
    assert TrustConstrConfig.uses_hess
    assert TrustConstrConfig.uses_hessp
    assert not SLSQPConfig.uses_hess
