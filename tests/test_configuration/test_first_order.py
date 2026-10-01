from dataclasses import fields

import pytest

from better_optimize.configuration.first_order import (
    BFGSConfig,
    CGConfig,
    LBFGSBConfig,
    TNCConfig,
)

ALL = [BFGSConfig, CGConfig, LBFGSBConfig, TNCConfig]


@pytest.mark.parametrize("config", ALL)
def test_they_all_want_a_gradient_and_nothing_more(config):
    assert config.uses_grad
    assert not config.uses_hess
    assert not config.uses_hessp


def test_cg_and_bfgs_disagree_about_the_curvature_parameter():
    assert BFGSConfig().c2 == 0.9
    assert CGConfig().c2 == 0.4


def test_lbfgsb_uses_a_smaller_finite_difference_step_than_bfgs():
    assert LBFGSBConfig().eps == 1e-8
    assert BFGSConfig().eps == 1.4901161193847656e-08


def test_lbfgsb_exposes_the_deprecated_verbosity_options_scipy_still_accepts():
    names = {field.name for field in fields(LBFGSBConfig)}

    assert {"disp", "iprint"} <= names
    assert "iprint" in LBFGSBConfig(iprint=1).optimizer_kwargs()
    assert "iprint" not in LBFGSBConfig().optimizer_kwargs()


def test_tnc_has_no_maxiter_because_scipy_ignores_it():
    names = {field.name for field in fields(TNCConfig)}

    assert "maxiter" not in names
    assert "maxfun" in names

    with pytest.raises(TypeError, match="maxiter"):
        TNCConfig(maxiter=500)


def test_tnc_budget_default_is_not_the_usual_two_hundred_n():
    assert TNCConfig().default_budget(10) == 100
    assert BFGSConfig().default_budget(10) == 2000


def test_tnc_keeps_its_negative_sentinels():
    options = TNCConfig().optimizer_kwargs()

    assert options["eta"] == -1
    assert options["ftol"] == -1
    assert options["maxCGit"] == -1


def test_every_method_reports_workers():
    for config in ALL:
        assert "workers" in {field.name for field in fields(config)}
        assert config(workers=2).optimizer_kwargs()["workers"] == 2
