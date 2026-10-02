import inspect

from dataclasses import fields

import numpy as np
import pytest
import scipy.optimize._nonlin as nonlin
import scipy.optimize._root as scipy_root

from scipy.optimize import root
from scipy.optimize._spectral import _root_df_sane

from better_optimize.configuration import ROOT_CONFIGS
from better_optimize.configuration.base import UNSET
from better_optimize.configuration.root_jac_options import (
    AndersonJacOptions,
    BroydenJacOptions,
    DiagonalJacOptions,
    ExcitingMixingJacOptions,
    KrylovJacOptions,
)
from better_optimize.configuration.root_quasi_newton import QuasiNewtonRootConfig

REGISTERED = sorted(ROOT_CONFIGS)

# scipy dispatches the ten methods to four functions, and the seven quasi-Newton methods
# share one of them exactly.
DISPATCH = {
    "hybr": scipy_root._root_hybr,
    "lm": scipy_root._root_leastsq,
    "df-sane": _root_df_sane,
    **dict.fromkeys(
        (
            "broyden1",
            "broyden2",
            "anderson",
            "linearmixing",
            "diagbroyden",
            "excitingmixing",
            "krylov",
        ),
        scipy_root._root_nonlin_solve,
    ),
}

# The problem and the reporting, which `root` supplies and no config describes. `callback`
# is here because `root` passes it positionally, so naming it in options is a TypeError
# rather than an unknown option.
NOT_OPTIONS = {"fun", "func", "x0", "args", "jac", "_callback", "_method", "callback"}

JACOBIANS = {
    "broyden1": nonlin.BroydenFirst,
    "broyden2": nonlin.BroydenSecond,
    "anderson": nonlin.Anderson,
    "linearmixing": nonlin.LinearMixing,
    "diagbroyden": nonlin.DiagBroyden,
    "excitingmixing": nonlin.ExcitingMixing,
    "krylov": nonlin.KrylovJacobian,
}

NESTED = {
    "broyden1": BroydenJacOptions,
    "broyden2": BroydenJacOptions,
    "anderson": AndersonJacOptions,
    "linearmixing": DiagonalJacOptions,
    "diagbroyden": DiagonalJacOptions,
    "excitingmixing": ExcitingMixingJacOptions,
    "krylov": KrylovJacOptions,
}

# These two overflow on scipy's own defaults, so they are only runnable with an alpha.
RUNNABLE_ALPHA = {"linearmixing": -0.5, "excitingmixing": -1.0}

# What `_root_nonlin_solve` renames each tolerance to on its way into TerminationCondition,
# which is where the value a signature check cannot see is chosen.
TERMINATION_NAMES = {"fatol": "f_tol", "ftol": "f_rtol", "xatol": "x_tol", "xtol": "x_rtol"}


def residual(x):
    return x - np.array([1.0, 2.0])


def scipy_options(method):
    """What the function scipy dispatches `method` to accepts, and each one's default."""
    parameters = inspect.signature(DISPATCH[method]).parameters

    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name not in NOT_OPTIONS and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }


def declared_options(method):
    return {field.name for field in fields(ROOT_CONFIGS[method])} - {"tol"}


@pytest.mark.parametrize("method", REGISTERED)
def test_the_fields_are_exactly_the_options_scipy_accepts(method):
    assert declared_options(method) == set(scipy_options(method))


@pytest.mark.parametrize("method", REGISTERED)
def test_the_defaults_match_the_scipy_source(method):
    """Except the budgets, which `better_optimize` owns so `default_budget` can scale them,
    and the options scipy's signature defaults to None and resolves in its body. Keep the
    exclusions to those two: a name excluded here is one this package hardcodes, and a
    hardcoded value nothing reads from scipy is one that can drift."""
    config = ROOT_CONFIGS[method]()
    scipy_defaults = scipy_options(method)
    excluded = set(config._budget_options()) | {
        name for name, default in scipy_defaults.items() if default is None
    }

    for name, default in scipy_defaults.items():
        if name not in excluded:
            assert getattr(config, name) == default, name


@pytest.mark.parametrize("method", sorted(JACOBIANS))
def test_the_quasi_newton_tolerances_match_what_scipy_resolves_them_to(method):
    """`_root_nonlin_solve` defaults all four to None and lets TerminationCondition choose,
    so the signature check sees nothing and this package has to state the values itself."""
    resolved = nonlin.TerminationCondition()
    config = ROOT_CONFIGS[method]()

    for name, termination_name in TERMINATION_NAMES.items():
        assert getattr(config, name) == getattr(resolved, termination_name), name


@pytest.mark.parametrize("method", REGISTERED)
def test_a_budget_the_caller_left_unset_is_filled_from_the_dimension(method):
    """Except where scipy derives the budget from something the configuration cannot see,
    and so has to be left to do it."""
    config = ROOT_CONFIGS[method]()
    options = config.optimizer_kwargs(n=10)

    for name in set(config._budget_options()) - set(config._scipy_resolved_budgets):
        assert options[name] == config.default_budget(10)

    for name in config._scipy_resolved_budgets:
        assert name not in options


@pytest.mark.parametrize("method", REGISTERED)
def test_no_sentinel_survives_construction(method):
    assert UNSET not in ROOT_CONFIGS[method]().optimizer_kwargs(n=10).values()


@pytest.mark.parametrize("method", sorted(JACOBIANS))
def test_the_nested_fields_are_exactly_what_the_jacobian_accepts(method):
    """The seven quasi-Newton methods differ only here, so this is what tells one from
    another."""
    parameters = inspect.signature(JACOBIANS[method].__init__).parameters
    accepted = {
        name
        for name, parameter in list(parameters.items())[1:]
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    declared = {field.name for field in fields(NESTED[method])} - {"inner_options"}

    assert declared == accepted


@pytest.mark.parametrize("method", sorted(JACOBIANS))
def test_the_wrong_nested_type_is_refused_at_construction(method):
    """Scipy cannot be relied on to object: a nested type whose every field is unset emits
    an empty mapping, which it accepts whichever jacobian it is building."""
    for wrong in {other for other in NESTED.values() if other is not NESTED[method]}:
        with pytest.raises(TypeError, match=f"builds a {NESTED[method].__name__}"):
            ROOT_CONFIGS[method](jac_options=wrong())


@pytest.mark.parametrize("method", sorted(JACOBIANS))
def test_the_nested_options_may_be_given_as_a_plain_mapping(method):
    """It is the shape scipy's own documentation shows, so it is what a caller tries
    first. The declared type checks the keys."""
    config = ROOT_CONFIGS[method](jac_options={})

    assert isinstance(config.jac_options, NESTED[method])


@pytest.mark.parametrize("method", REGISTERED)
def test_scipy_accepts_every_option_the_config_emits(method):
    """The signature check cannot see a rename scipy applies before dispatch; this can."""
    config = ROOT_CONFIGS[method]()
    alpha = RUNNABLE_ALPHA.get(method)
    if alpha is not None:
        config = ROOT_CONFIGS[method](jac_options=NESTED[method](alpha=alpha))

    result = root(
        residual, np.array([0.9, 1.9]), method=method, options=config.optimizer_kwargs(n=2)
    )

    assert np.all(np.isfinite(result.x))


def test_every_quasi_newton_method_shares_the_one_surface():
    quasi_newton = [m for m in REGISTERED if issubclass(ROOT_CONFIGS[m], QuasiNewtonRootConfig)]

    assert len(quasi_newton) == 7
    assert len({frozenset(declared_options(m)) for m in quasi_newton}) == 1


def test_no_two_configs_claim_the_same_method():
    assert len({ROOT_CONFIGS[m]().method_name for m in REGISTERED}) == len(REGISTERED)


@pytest.mark.parametrize("method", REGISTERED)
def test_the_config_names_the_method_it_is_registered_under(method):
    assert ROOT_CONFIGS[method]().method_name == method
