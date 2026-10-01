from dataclasses import fields

import pytest

from better_optimize.configuration.first_order import BFGSConfig
from better_optimize.configuration.second_order import (
    DoglegConfig,
    NewtonCGConfig,
    TrustExactConfig,
    TrustKrylovConfig,
    TrustNCGConfig,
    TrustRegionConfig,
)

TRUST_REGION = [DoglegConfig, TrustNCGConfig, TrustExactConfig, TrustKrylovConfig]


@pytest.mark.parametrize("config", [*TRUST_REGION, NewtonCGConfig])
def test_they_all_want_a_hessian(config):
    assert config.uses_grad
    assert config.uses_hess


def test_only_the_krylov_and_ncg_subproblems_take_a_hessian_vector_product():
    assert TrustNCGConfig.uses_hessp
    assert TrustKrylovConfig.uses_hessp
    assert not DoglegConfig.uses_hessp
    assert not TrustExactConfig.uses_hessp


@pytest.mark.parametrize("config", TRUST_REGION)
def test_the_shared_driver_gives_them_one_option_set(config):
    shared = {field.name for field in fields(TrustRegionConfig)}

    assert {field.name for field in fields(config)} == shared


def test_the_shared_base_is_not_a_usable_method():
    with pytest.raises(TypeError, match="abstract"):
        TrustRegionConfig()


def test_newton_cg_stops_on_the_step_not_the_gradient():
    names = {field.name for field in fields(NewtonCGConfig)}

    assert "xtol" in names
    assert "gtol" not in names
    assert NewtonCGConfig(tol=1e-9).optimizer_kwargs()["xtol"] == 1e-9


def test_the_trust_region_gradient_tolerance_is_looser_than_the_quasi_newton_one():
    assert TrustNCGConfig().gtol == 1e-4
    assert TrustNCGConfig().gtol > BFGSConfig().gtol
