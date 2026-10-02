from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from scipy.optimize import root

from better_optimize.configuration.root_jac_options import (
    AndersonJacOptions,
    BroydenJacOptions,
    DiagonalJacOptions,
    ExcitingMixingJacOptions,
    KrylovJacOptions,
)
from better_optimize.configuration.root_quasi_newton import (
    AndersonConfig,
    Broyden1Config,
    Broyden2Config,
    DiagBroydenConfig,
    ExcitingMixingConfig,
    KrylovConfig,
    LinearMixingConfig,
    QuasiNewtonRootConfig,
)

ALL = [
    Broyden1Config,
    Broyden2Config,
    AndersonConfig,
    LinearMixingConfig,
    DiagBroydenConfig,
    ExcitingMixingConfig,
    KrylovConfig,
]


def residual(x):
    return x - np.array([1.0, 2.0])


@pytest.mark.parametrize("config", ALL)
def test_they_consume_no_jacobian(config):
    assert not config.uses_jac


@pytest.mark.parametrize("config", ALL)
def test_only_the_absolute_residual_tolerance_is_enabled_by_default(config):
    """scipy resolves three of the four to infinity, so an unset run stops on `fatol`. The
    value it resolves `fatol` to is pinned against scipy in the agreement tests."""
    options = config().optimizer_kwargs()

    assert np.isfinite(options["fatol"])
    assert options["ftol"] == options["xtol"] == options["xatol"] == np.inf


@pytest.mark.parametrize("config", ALL)
def test_tol_switches_the_test_to_the_step_rather_than_filling_every_tolerance(config):
    """A minimize method fills all of its tolerances from `tol`. These disable the other
    three, which turns `fatol` off."""
    options = config(tol=1e-9).optimizer_kwargs()

    assert options["xtol"] == 1e-9
    assert options["fatol"] == options["ftol"] == options["xatol"] == np.inf


def test_a_tolerance_the_caller_set_survives_tol():
    assert Broyden1Config(tol=1e-9, fatol=1e-12).optimizer_kwargs()["fatol"] == 1e-12


@pytest.mark.parametrize("declared", [None, dict], ids=["nothing", "a type that is not one"])
def test_a_subclass_must_name_the_nested_type_its_jacobian_builds(declared):
    """The narrowed annotation on `jac_options` does not enforce itself, so a method naming
    no usable type would accept another method's options and emit them."""
    with pytest.raises(TypeError, match="must name the JacOptions type"):

        @dataclass(frozen=True, eq=False)
        class Unnamed(QuasiNewtonRootConfig):
            _jac_options_type: ClassVar[type] = declared

            @property
            def method_name(self) -> str:
                return "unnamed"


def test_the_line_search_defaults_to_the_one_scipy_applies():
    assert Broyden1Config().optimizer_kwargs()["line_search"] == "armijo"


def test_disabling_the_line_search_reaches_scipy_as_None():
    """None is a value scipy acts on here, so the rule that an unset option is omitted
    would otherwise hand the caller the armijo search they asked to turn off."""
    options = Broyden1Config(line_search=None).optimizer_kwargs()

    assert options["line_search"] is None

    result = root(residual, np.array([0.9, 1.9]), method="broyden1", options=options)

    assert np.allclose(result.x, [1.0, 2.0])


@pytest.mark.parametrize("config", ALL)
def test_the_iteration_budget_scales_with_the_dimension(config):
    assert config().optimizer_kwargs(n=4)["maxiter"] == 100 * 5


def test_the_nested_options_reach_scipy_as_a_mapping():
    """scipy splats `jac_options` into a jacobian class, so it has to arrive as a dict
    rather than as the configuration that describes it."""
    options = Broyden1Config(jac_options=BroydenJacOptions(max_rank=3)).optimizer_kwargs()

    assert options["jac_options"] == {"reduction_method": "restart", "max_rank": 3}


def test_an_unset_nested_option_is_omitted():
    assert "jac_options" not in Broyden1Config().optimizer_kwargs()


@pytest.mark.parametrize(
    "config, options",
    [
        (Broyden1Config, BroydenJacOptions(max_rank=2)),
        (AndersonConfig, AndersonJacOptions(M=3)),
        (DiagBroydenConfig, DiagonalJacOptions(alpha=-0.5)),
        (ExcitingMixingConfig, ExcitingMixingJacOptions(alpha=-1.0)),
        (LinearMixingConfig, DiagonalJacOptions(alpha=-0.5)),
        (KrylovConfig, KrylovJacOptions(inner_maxiter=5)),
    ],
    ids=["broyden1", "anderson", "diagbroyden", "excitingmixing", "linearmixing", "krylov"],
)
def test_each_method_takes_the_nested_options_its_jacobian_accepts(config, options):
    """A key the jacobian class does not take raises from its constructor, so pairing the
    wrong type with a method cannot pass quietly."""
    result = root(
        residual,
        np.array([0.9, 1.9]),
        method=config().method_name,
        options=config(jac_options=options).optimizer_kwargs(n=2),
    )

    assert np.all(np.isfinite(result.x))
