import warnings

from dataclasses import fields

import numpy as np
import pytest

from scipy.optimize import OptimizeWarning
from scipy.optimize import minimize as sp_minimize

from better_optimize.configuration import MINIMIZE_CONFIGS
from tests.test_configuration.scipy_reference import (
    OMITTED_OPTIONS,
    SCIPY_OPTIONS,
    tol_targets,
)

REGISTERED = sorted(MINIMIZE_CONFIGS)


def config_fields(method):
    return {field.name: field.default for field in fields(MINIMIZE_CONFIGS[method])}


def declared_options(method):
    """The config's fields, less the base's `tol` unless the method really has that option."""
    return set(config_fields(method)) - ({"tol"} - set(SCIPY_OPTIONS[method]))


@pytest.mark.parametrize("method", REGISTERED)
def test_fields_are_exactly_the_options_scipy_accepts(method):
    declared = declared_options(method)
    accepted = set(SCIPY_OPTIONS[method]) - set(OMITTED_OPTIONS.get(method, {}))

    assert declared == accepted


@pytest.mark.parametrize("method", REGISTERED)
def test_omitted_options_are_real_options_with_a_stated_reason(method):
    for name, reason in OMITTED_OPTIONS.get(method, {}).items():
        assert name in SCIPY_OPTIONS[method]
        assert reason


@pytest.mark.parametrize("method", REGISTERED)
def test_defaults_match_the_scipy_source(method):
    config = MINIMIZE_CONFIGS[method]
    ours = config_fields(method)

    for name, scipy_default in SCIPY_OPTIONS[method].items():
        if name in config._budget_options or name in OMITTED_OPTIONS.get(method, {}):
            continue
        assert ours[name] == scipy_default, name


@pytest.mark.parametrize("method", REGISTERED)
def test_budget_options_are_unset_so_better_optimize_can_choose(method):
    config = MINIMIZE_CONFIGS[method]

    for name in config._budget_options:
        assert name in SCIPY_OPTIONS[method]
        assert config_fields(method)[name] is None

    assert config().default_maxiter(10) > 0


@pytest.mark.parametrize("method", REGISTERED)
def test_tol_options_match_scipys_own_dispatch(method):
    assert MINIMIZE_CONFIGS[method]._tol_options == tol_targets(method)


@pytest.mark.parametrize("method", REGISTERED)
def test_scipy_accepts_every_emitted_option(method):
    """The signature check cannot see renames scipy applies before dispatch; this can."""
    config = MINIMIZE_CONFIGS[method]()
    derivatives = {}
    if config.uses_grad:
        derivatives["jac"] = lambda x: 2 * x
    if config.uses_hessp:
        derivatives["hessp"] = lambda x, p: 2 * p
    elif config.uses_hess:
        derivatives["hess"] = lambda x: 2 * np.eye(x.size)

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        sp_minimize(
            lambda x: (x**2).sum(),
            np.array([1.0, 1.0]),
            method=method,
            options=config.optimizer_kwargs,
            **derivatives,
        )
