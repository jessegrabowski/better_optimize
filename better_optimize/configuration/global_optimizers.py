from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from better_optimize.configuration.base import MinimizeConfig
from better_optimize.configuration.first_order import LBFGSBConfig

__all__ = ["BasinHoppingConfig"]


@dataclass(frozen=True, eq=False)
class BasinHoppingConfig(MinimizeConfig):
    """Basin hopping, which restarts a local minimizer from perturbed points.

    The inner minimizer is a configuration of its own, and this one reports whatever
    derivatives that minimizer needs.

    Parameters
    ----------
    minimizer_config : MinimizeConfig, optional
        The local minimizer run at each basin. Defaults to :class:`LBFGSBConfig`, which is
        what scipy uses.
    niter : int, optional
        Number of basin hopping iterations. Defaults to 100.
    T : float, optional
        Temperature of the Metropolis acceptance criterion. Higher values accept larger
        increases in the objective. Defaults to 1.0.
    stepsize : float, optional
        Initial size of the random displacement between basins, adapted during the run to
        hit `target_accept_rate`. Defaults to 0.5.
    interval : int, optional
        How often, in iterations, to adapt `stepsize`. Defaults to 50.
    niter_success : int, optional
        Stop once the best objective has not improved for this many iterations. Defaults to
        None, which runs all `niter` iterations.
    target_accept_rate : float, optional
        Acceptance rate the `stepsize` adaptation aims for. Defaults to 0.5.
    stepwise_factor : float, optional
        Multiplier applied to `stepsize` at each adaptation. Defaults to 0.9.
    accept_on_minimizer_fail : bool, optional
        Treat a failed local minimization as a candidate rather than discarding it.
        Defaults to False.
    rng : int or numpy.random.Generator, optional
        Seed or generator for the displacements. Defaults to None.
    take_step : callable, optional
        Replaces the built-in displacement. Defaults to None.
    accept_test : callable, optional
        Extra acceptance criterion applied alongside the Metropolis test. Defaults to None.
    """

    minimizer_config: MinimizeConfig = field(default_factory=LBFGSBConfig)
    niter: int | None = None
    T: float = 1.0
    stepsize: float = 0.5
    interval: int = 50
    niter_success: int | None = None
    target_accept_rate: float = 0.5
    stepwise_factor: float = 0.9
    accept_on_minimizer_fail: bool = False
    rng: int | float | np.random.Generator | None = None
    take_step: Callable[..., Any] | None = None
    accept_test: Callable[..., Any] | None = None

    _excluded: ClassVar[frozenset[str]] = frozenset({"tol", "minimizer_config"})
    _iteration_options: ClassVar[tuple[str, ...]] = ("niter",)

    @property
    def method_name(self) -> str:
        return "basinhopping"

    @property
    def uses_grad(self) -> bool:
        return self.minimizer_config.uses_grad

    @property
    def uses_hess(self) -> bool:
        return self.minimizer_config.uses_hess

    @property
    def uses_hessp(self) -> bool:
        return self.minimizer_config.uses_hessp

    def default_budget(self, n: int) -> int:
        return 100

    def solver_function(self) -> Callable[..., Any]:
        from better_optimize.basinhopping import basinhopping

        return basinhopping
