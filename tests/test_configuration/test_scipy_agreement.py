from dataclasses import fields

import pytest

from better_optimize.configuration import MINIMIZE_CONFIGS
from better_optimize.configuration.base import UNSET
from tests.test_configuration.scipy_reference import (
    OMITTED_OPTIONS,
    SCIPY_OPTIONS,
    tol_targets,
)

REGISTERED = sorted(MINIMIZE_CONFIGS)


def config_fields(method):
    """Declared defaults, taking a tolerance's from ``_tol_options`` where the field holds
    the unset sentinel."""
    config = MINIMIZE_CONFIGS[method]

    return {
        field.name: config._tol_options.get(field.name, field.default) for field in fields(config)
    }


def declared_options(method):
    """The config's fields, less the base's `tol` unless the method really has that option."""
    return set(config_fields(method)) - ({"tol"} - set(SCIPY_OPTIONS[method]))


@pytest.mark.parametrize("method", REGISTERED)
def test_fields_are_exactly_the_options_scipy_accepts(method):
    declared = declared_options(method)
    accepted = set(SCIPY_OPTIONS[method]) - set(OMITTED_OPTIONS.get(method, {}))

    assert declared == accepted


@pytest.mark.parametrize("method", REGISTERED)
def test_omitted_options_are_real_scipy_options(method):
    for name in OMITTED_OPTIONS.get(method, {}):
        assert name in SCIPY_OPTIONS[method]


@pytest.mark.parametrize("method", REGISTERED)
def test_defaults_match_the_scipy_source(method):
    config = MINIMIZE_CONFIGS[method]
    ours = config_fields(method)

    for name, scipy_default in SCIPY_OPTIONS[method].items():
        if name in config._budget_options() or name in OMITTED_OPTIONS.get(method, {}):
            continue
        assert ours[name] == scipy_default, name


@pytest.mark.parametrize("method", REGISTERED)
def test_budget_options_are_unset_so_better_optimize_can_choose(method):
    config = MINIMIZE_CONFIGS[method]

    for name in config._budget_options():
        assert name in SCIPY_OPTIONS[method]
        assert config_fields(method)[name] is None

    assert config().default_budget(10) > 0


@pytest.mark.parametrize("method", REGISTERED)
def test_every_declared_option_reaches_scipy(method):
    config_class = MINIMIZE_CONFIGS[method]
    config = config_class(**{name: 123 for name in config_class._budget_options()})

    assert set(config.optimizer_kwargs()) == declared_options(method)


@pytest.mark.parametrize("method", REGISTERED)
def test_tol_options_match_scipys_own_dispatch(method):
    assert tuple(MINIMIZE_CONFIGS[method]._tol_options) == tol_targets(method)


@pytest.mark.parametrize("method", REGISTERED)
def test_tol_fills_every_tolerance_the_caller_did_not_pass(method):
    options = MINIMIZE_CONFIGS[method](tol=1e-9).optimizer_kwargs()

    assert {options[name] for name in MINIMIZE_CONFIGS[method]._tol_options} == {1e-9}


@pytest.mark.parametrize("method", REGISTERED)
def test_an_unset_tolerance_resolves_to_the_scipy_default(method):
    config = MINIMIZE_CONFIGS[method]()

    for name, scipy_default in config._tol_options.items():
        assert getattr(config, name) == scipy_default


@pytest.mark.parametrize("method", REGISTERED)
def test_a_dimension_fills_every_budget_the_caller_left_unset(method):
    config = MINIMIZE_CONFIGS[method]()
    options = config.optimizer_kwargs(n=10)

    for name in config._budget_options():
        assert options[name] == config.default_budget(10)


@pytest.mark.parametrize("method", REGISTERED)
def test_no_sentinel_survives_construction(method):
    """A field defaulting to UNSET but missing from ``_tol_options`` would reach scipy."""
    assert UNSET not in MINIMIZE_CONFIGS[method]().optimizer_kwargs(n=10).values()
