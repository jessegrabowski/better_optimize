import numpy as np
import pytest

from scipy.optimize import rosen, rosen_der, rosen_hess

from better_optimize import minimize
from better_optimize.basinhopping import basinhopping
from better_optimize.configuration import (
    BasinHoppingConfig,
    LBFGSBConfig,
    NelderMeadConfig,
    NewtonCGConfig,
    TrustNCGConfig,
)

X0 = np.array([-1.2, 1.0])


@pytest.mark.parametrize(
    "minimizer_config, uses_grad, uses_hess, uses_hessp",
    [
        (NelderMeadConfig(), False, False, False),
        (LBFGSBConfig(), True, False, False),
        (NewtonCGConfig(), True, True, True),
    ],
    ids=["nelder-mead", "l-bfgs-b", "newton-cg"],
)
def test_the_inner_minimizer_decides_which_derivatives_are_used(
    minimizer_config, uses_grad, uses_hess, uses_hessp
):
    config = BasinHoppingConfig(minimizer_config=minimizer_config)

    assert (config.uses_grad, config.uses_hess, config.uses_hessp) == (
        uses_grad,
        uses_hess,
        uses_hessp,
    )


def test_the_inner_config_is_not_emitted_as_an_option():
    """It reaches scipy as ``minimizer_kwargs["method"]``, not in the options."""
    options = BasinHoppingConfig(tol=1e-9).optimizer_kwargs(n=2)

    assert "minimizer_config" not in options
    assert "tol" not in options


def test_an_option_the_caller_left_unset_is_omitted():
    assert set(BasinHoppingConfig().optimizer_kwargs()) < set(
        BasinHoppingConfig(niter_success=3, rng=0).optimizer_kwargs()
    )


def test_a_dimension_fills_the_iteration_budget():
    config = BasinHoppingConfig()

    assert "niter" not in config.optimizer_kwargs()
    assert config.optimizer_kwargs(n=2)["niter"] == config.default_budget(2)


def test_minimize_dispatches_a_config_to_basinhopping():
    through_minimize = minimize(
        rosen, X0, method=BasinHoppingConfig(niter=3, rng=0), jac=rosen_der, progressbar=False
    )
    directly = basinhopping(
        rosen,
        X0,
        niter=3,
        rng=0,
        minimizer_kwargs={"method": LBFGSBConfig(), "jac": rosen_der},
        progressbar=False,
    )

    np.testing.assert_allclose(through_minimize.x, directly.x)


def test_the_inner_minimizer_budget_reaches_a_trust_region_method():
    """Trust-region methods have no evaluation-capping option, so a dropped ``maxiter``
    would leave them running to scipy's default rather than raising."""

    def solve(maxiter):
        return minimize(
            rosen,
            X0,
            method=BasinHoppingConfig(
                niter=2, rng=0, minimizer_config=TrustNCGConfig(maxiter=maxiter)
            ),
            jac=rosen_der,
            hess=rosen_hess,
            progressbar=False,
        )

    assert solve(maxiter=3).fun > solve(maxiter=500).fun
