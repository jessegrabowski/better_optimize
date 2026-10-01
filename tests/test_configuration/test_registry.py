from typing import get_args

import pytest

from better_optimize.configuration import (
    MINIMIZE_CONFIGS,
    SOLVER_ARGUMENTS,
    BFGSConfig,
    config_for_method,
    config_from_kwargs,
)
from better_optimize.constants import minimize_method

REGISTERED = sorted(MINIMIZE_CONFIGS)


def test_the_registry_covers_exactly_the_advertised_methods():
    assert set(MINIMIZE_CONFIGS) == set(get_args(minimize_method))


@pytest.mark.parametrize("method", REGISTERED)
def test_the_config_names_the_method_it_is_registered_under(method):
    assert config_for_method(method).method_name == method


def test_no_two_configs_claim_the_same_method():
    names = [config().method_name for config in MINIMIZE_CONFIGS.values()]

    assert len(names) == len(set(names))


def test_options_reach_the_config():
    assert config_for_method("BFGS", gtol=1e-9).gtol == 1e-9


@pytest.mark.parametrize("method", REGISTERED)
def test_a_method_name_may_be_written_in_any_case(method):
    assert config_for_method(method.lower()).method_name == method
    assert config_for_method(method.upper()).method_name == method


def test_no_two_methods_differ_only_in_case():
    """A collision would drop one config out of the case-insensitive lookup silently."""
    assert len({method.lower() for method in MINIMIZE_CONFIGS}) == len(MINIMIZE_CONFIGS)


def test_an_unknown_method_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="Unknown method 'nonsense'"):
        config_for_method("nonsense")


def test_an_unknown_option_is_a_type_error():
    with pytest.raises(TypeError, match="gtoll"):
        config_for_method("BFGS", gtoll=1e-9)


def test_a_configuration_is_used_as_given():
    config = BFGSConfig(gtol=1e-9)

    assert config_from_kwargs(config, {})[0] is config


def test_a_configuration_cannot_be_combined_with_options():
    with pytest.raises(TypeError, match=r"BFGSConfig and the option\(s\) \['gtol'\]"):
        config_from_kwargs(BFGSConfig(), {"gtol": 1e-9})


@pytest.mark.parametrize("name", sorted(SOLVER_ARGUMENTS))
def test_an_argument_describing_the_problem_goes_to_scipy_not_the_config(name):
    config, solver_kwargs = config_from_kwargs("L-BFGS-B", {name: "value", "gtol": 1e-9})

    assert solver_kwargs == {name: "value"}
    assert name not in config.optimizer_kwargs()


def test_a_configuration_still_yields_the_problem_arguments():
    _, solver_kwargs = config_from_kwargs(BFGSConfig(), {"bounds": "value"})

    assert solver_kwargs == {"bounds": "value"}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("BFGS", {"maxiter": 500}),
        ("TNC", {"maxfun": 500}),
        ("L-BFGS-B", {"maxiter": 500, "maxfun": 500}),
        ("nelder-mead", {"maxiter": 500, "maxfev": 500}),
        ("COBYLA", {"maxiter": 500}),
    ],
)
def test_a_top_level_maxiter_fills_whatever_the_method_calls_its_budget(method, expected):
    """TNC has no maxiter of its own, and three of these cap two things separately."""
    config, _ = config_from_kwargs(method, {"maxiter": 500})
    options = config.optimizer_kwargs()

    assert {name: options[name] for name in expected} == expected


def test_an_explicit_budget_survives_a_top_level_maxiter():
    config, _ = config_from_kwargs("nelder-mead", {"maxiter": 900, "maxfev": 5})
    options = config.optimizer_kwargs()

    assert options["maxiter"] == 900
    assert options["maxfev"] == 5


def test_an_options_dictionary_is_merged():
    config, _ = config_from_kwargs("BFGS", {"options": {"gtol": 1e-3}})

    assert config.optimizer_kwargs()["gtol"] == 1e-3


def test_a_name_given_both_ways_takes_its_top_level_value():
    config, _ = config_from_kwargs("BFGS", {"options": {"gtol": 1e-3}, "gtol": 1e-7})

    assert config.optimizer_kwargs()["gtol"] == 1e-7


def test_an_unknown_method_names_the_ones_that_exist_from_kwargs():
    with pytest.raises(ValueError, match="Unknown method 'nonsense'"):
        config_from_kwargs("nonsense", {"maxiter": 5})
