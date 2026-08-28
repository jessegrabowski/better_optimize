from typing import get_args

import pytest

from better_optimize.configuration import MINIMIZE_CONFIGS, config_for_method
from better_optimize.constants import minimize_method

REGISTERED = sorted(MINIMIZE_CONFIGS)


@pytest.mark.parametrize("method", REGISTERED)
def test_every_key_is_a_method_better_optimize_advertises(method):
    assert method in get_args(minimize_method)


@pytest.mark.parametrize("method", REGISTERED)
def test_the_config_names_the_method_it_is_registered_under(method):
    assert config_for_method(method).method_name == method


def test_no_two_configs_claim_the_same_method():
    names = [config().method_name for config in MINIMIZE_CONFIGS.values()]

    assert len(names) == len(set(names))


def test_options_reach_the_config():
    assert config_for_method("BFGS", gtol=1e-9).gtol == 1e-9


def test_an_unknown_method_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="Unknown method 'bfgs'"):
        config_for_method("bfgs")


def test_an_unknown_option_is_a_type_error():
    with pytest.raises(TypeError, match="gtoll"):
        config_for_method("BFGS", gtoll=1e-9)
