from dataclasses import fields

import pytest

from better_optimize.configuration.gradient_free import NelderMeadConfig, PowellConfig

ALL = [NelderMeadConfig, PowellConfig]


@pytest.mark.parametrize("config", ALL)
def test_they_use_no_derivative_information(config):
    assert not config.uses_grad
    assert not config.uses_hess
    assert not config.uses_hessp


@pytest.mark.parametrize("config", ALL)
def test_both_budgets_are_capped_independently(config):
    assert config._budget_options == ("maxiter", "maxfev")
    assert config(maxiter=5, maxfev=900).resolved_maxiter(10) == 900


def test_powell_is_ten_times_more_generous_than_nelder_mead():
    assert PowellConfig().default_maxiter(10) == 10000
    assert NelderMeadConfig().default_maxiter(10) == 2000


def test_they_stop_on_different_tolerances():
    assert NelderMeadConfig(tol=1e-9).optimizer_kwargs["xatol"] == 1e-9
    assert NelderMeadConfig(tol=1e-9).optimizer_kwargs["fatol"] == 1e-9
    assert PowellConfig(tol=1e-9).optimizer_kwargs["xtol"] == 1e-9
    assert PowellConfig(tol=1e-9).optimizer_kwargs["ftol"] == 1e-9


def test_neither_exposes_bounds_as_an_option():
    for config in ALL:
        assert "bounds" not in {field.name for field in fields(config)}
