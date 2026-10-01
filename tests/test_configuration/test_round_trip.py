import warnings

import numpy as np
import pytest

from scipy.optimize import OptimizeWarning, rosen, rosen_der, rosen_hess, rosen_hess_prod
from scipy.optimize import minimize as sp_minimize

from better_optimize.configuration import MINIMIZE_CONFIGS, PowellConfig

REGISTERED = sorted(MINIMIZE_CONFIGS)

# Far enough from the minimum that every method needs many iterations to get close, so a
# budget that was silently dropped shows up as a better result than it should be.
X0 = np.array([-1.2, 1.0, -1.2, 1.0])

# COBYLA rejects a function-evaluation budget below ``n + 2``.
TIGHT_BUDGET = X0.size + 2


def solve(config, **options):
    derivatives = {}
    if config.uses_grad:
        derivatives["jac"] = rosen_der
    if config.uses_hessp:
        derivatives["hessp"] = rosen_hess_prod
    elif config.uses_hess:
        derivatives["hess"] = rosen_hess

    return sp_minimize(rosen, X0, method=config.method_name, **derivatives, **options)


@pytest.mark.parametrize("method", REGISTERED)
def test_a_tight_budget_reaches_the_solver(method):
    """Nothing else checks that an option a config emits changes what scipy does."""
    config_class = MINIMIZE_CONFIGS[method]
    tight = config_class(**{name: TIGHT_BUDGET for name in config_class._budget_options()})

    stopped_early = solve(tight, options=tight.optimizer_kwargs())
    ran_to_default = solve(config_class(), options=config_class().optimizer_kwargs(n=X0.size))

    assert stopped_early.fun > ran_to_default.fun


@pytest.mark.parametrize("method", REGISTERED)
def test_a_loose_tolerance_reaches_the_solver(method):
    """COBYLA rejects a tolerance above its initial trust radius, so stay below 1.0."""
    config_class = MINIMIZE_CONFIGS[method]

    loose = solve(config_class(tol=0.1), options=config_class(tol=0.1).optimizer_kwargs())
    tight = solve(config_class(tol=1e-10), options=config_class(tol=1e-10).optimizer_kwargs())

    assert loose.fun >= tight.fun


@pytest.mark.parametrize("method", REGISTERED)
def test_scipy_accepts_every_emitted_option(method):
    """The signature check cannot see renames scipy applies before dispatch; this can."""
    config = MINIMIZE_CONFIGS[method]()

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        result = solve(config, options=config.optimizer_kwargs(n=X0.size))

    assert np.isfinite(result.fun)
    assert result.fun < rosen(X0)


def test_an_array_option_is_not_written_through_by_the_solver():
    """scipy reorders Powell's ``direc`` in place, which would leak one run into the next."""
    directions = np.eye(X0.size)
    config = PowellConfig(direc=directions)

    solve(config, options=config.optimizer_kwargs(n=X0.size))

    assert np.array_equal(config.direc, np.eye(X0.size))
    assert np.array_equal(directions, np.eye(X0.size))


@pytest.mark.parametrize(
    "method", [m for m in REGISTERED if MINIMIZE_CONFIGS[m]._evaluation_options]
)
def test_the_evaluation_budget_bounds_what_the_solver_spends(method):
    """The wrapper counts objective calls against this, so it has to track the option
    scipy actually caps evaluations with."""
    config_class = MINIMIZE_CONFIGS[method]
    budget = 30
    config = config_class(**dict.fromkeys(config_class._evaluation_options, budget))

    result = solve(config, options=config.optimizer_kwargs())

    assert config.evaluation_budget(X0.size) == budget
    assert result.nfev <= budget + X0.size
