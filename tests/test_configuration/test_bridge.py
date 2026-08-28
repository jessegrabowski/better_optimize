from dataclasses import fields

import pytest

from better_optimize.configuration import MINIMIZE_CONFIGS
from better_optimize.constants import MINIMIZE_MODE_KWARGS
from tests.test_configuration.scipy_reference import SCIPY_OPTIONS

REGISTERED = sorted(MINIMIZE_CONFIGS)


def declared_options(method: str) -> set[str]:
    """The config's fields, less the base's `tol` unless the method really has that option."""
    fields_declared = {field.name for field in fields(MINIMIZE_CONFIGS[method])}

    return fields_declared - ({"tol"} - set(SCIPY_OPTIONS[method]))


# Options scipy accepts that the table never listed. Verified against the scipy source by
# tests/test_configuration/test_scipy_agreement.py.
MISSING_FROM_TABLE = {
    "CG": {"workers"},
    "BFGS": {"workers"},
    "L-BFGS-B": {"workers"},
    "TNC": {"mesg_num", "workers"},
}

# Options the table lists that we deliberately do not expose.
NOT_CARRIED_OVER = {
    "L-BFGS-B": {"disp", "iprint"},
}


@pytest.mark.parametrize("method", REGISTERED)
def test_the_config_carries_the_table_forward(method):
    declared = declared_options(method)
    tabled = set(MINIMIZE_MODE_KWARGS[method]["valid_options"])

    assert tabled - declared == NOT_CARRIED_OVER.get(method, set())
    assert declared - tabled == MISSING_FROM_TABLE.get(method, set())


@pytest.mark.parametrize("method", REGISTERED)
def test_the_budget_default_is_unchanged(method):
    config = MINIMIZE_CONFIGS[method]()
    tabled = MINIMIZE_MODE_KWARGS[method]["f_maxiter_default"]

    for n in (1, 10, 100):
        assert config.default_maxiter(n) == tabled(n)


@pytest.mark.parametrize("method", REGISTERED)
def test_the_derivative_appetite_is_unchanged(method):
    config = MINIMIZE_CONFIGS[method]
    tabled = MINIMIZE_MODE_KWARGS[method]

    assert config.uses_grad == tabled["uses_grad"]
    assert config.uses_hess == tabled["uses_hess"]
    assert config.uses_hessp == tabled["uses_hessp"]
