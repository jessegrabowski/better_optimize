from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from better_optimize.configuration.base import (
    UNSET,
    MinimizeConfig,
    OptimizerConfig,
    SolverProblem,
)
from better_optimize.configuration.first_order import LBFGSBConfig

__all__ = ["BasinHoppingConfig", "DifferentialEvolutionConfig"]

DE_STRATEGY_OPTIONS = (
    "best1bin",
    "best1exp",
    "rand1bin",
    "rand1exp",
    "randtobest1bin",
    "randtobest1exp",
    "currenttobest1bin",
    "currenttobest1exp",
    "best2bin",
    "best2exp",
    "rand2bin",
    "rand2exp",
)
"""The mutation strategies scipy dispatches on, read off its ``_binomial`` and
``_exponential`` tables. A strategy may also be a callable, which scipy does not check."""

DE_INIT_OPTIONS = ("sobol", "halton", "latinhypercube", "random")
"""The population initializers scipy accepts, beside an array of starting points."""


@dataclass(frozen=True, eq=False)
class BasinHoppingConfig(OptimizerConfig):
    """Basin hopping, which restarts a local minimizer from perturbed points.

    The inner minimizer is a configuration of its own, and it is the one that says which
    derivatives the run needs.

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

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.tol is not None:
            raise TypeError(
                "basinhopping has no tolerance of its own. Set one on minimizer_config, "
                f"as {type(self.minimizer_config).__name__}(tol={self.tol!r})."
            )

    @property
    def method_name(self) -> str:
        return "basinhopping"

    def default_budget(self, n: int) -> int:
        return 100

    def solver_function(self) -> Callable[..., Any]:
        # basinhopping calls back into minimize, so the import cannot be at module scope.
        from better_optimize.basinhopping import basinhopping

        return basinhopping

    def build_solver_kwargs(self, problem: SolverProblem) -> dict[str, Any]:
        if problem.progressbar_update_interval != 1:
            raise TypeError(
                "basinhopping reports once per basin, so it cannot take "
                "progressbar_update_interval."
            )

        return {
            "func": problem.f,
            "x0": problem.x0,
            "minimizer_kwargs": {
                "method": self.minimizer_config,
                "jac": problem.jac,
                "hess": problem.hess,
                "hessp": problem.hessp,
                "args": problem.args,
                **problem.solver_kwargs,
            },
            "callback": problem.callback,
            "progressbar": problem.progressbar,
            "verbose": problem.verbose,
            **self.optimizer_kwargs(n=len(problem.x0)),
        }


@dataclass(frozen=True, eq=False)
class DifferentialEvolutionConfig(OptimizerConfig):
    """Differential evolution, a population-based global search over a bounded region.

    It uses no derivative information, and `bounds` is required rather than optional.

    Two defaults are `better_optimize`'s rather than scipy's: `init` is `"sobol"` where
    scipy uses `"latinhypercube"`, and an unset `maxiter` scales with the problem dimension
    where scipy applies a flat 1000.

    Parameters
    ----------
    strategy : str or callable, optional
        Mutation strategy, one of `DE_STRATEGY_OPTIONS` or a callable building a trial
        vector. Defaults to "best1bin".
    maxiter : int, optional
        Maximum number of generations. Defaults to None, meaning
        ``max(1000, 200 * d)`` for a problem of dimension `d`.
    popsize : int, optional
        Population size multiplier. The population holds `popsize` times the number of
        free parameters, except under ``init="sobol"``, which rounds up to a power of
        two. Defaults to 15.
    tol : float, optional
        Relative tolerance on the population's spread, which is the convergence
        criterion alongside `atol`. Defaults to 0.01; None leaves scipy to apply that
        same default itself.
    mutation : float or tuple of float, optional
        Differential weight, or the low and high ends of the range it is drawn from each
        generation. Defaults to (0.5, 1.0).
    recombination : float, optional
        Crossover probability, between 0 and 1. Defaults to 0.7.
    rng : int or numpy.random.Generator, optional
        Seed or generator for the population and the mutations. Defaults to None.
    init : str or numpy.ndarray, optional
        Population initializer, one of `DE_INIT_OPTIONS` or an array of starting points
        shaped ``(population_size, d)``. Defaults to "sobol".
    atol : float, optional
        Absolute tolerance on the population's spread. Defaults to 0.0.
    updating : str, optional
        "immediate" feeds a better trial vector back into the same generation, which
        converges faster but cannot be parallelized. "deferred" updates once per
        generation and is what `workers` requires. Defaults to "immediate".
    workers : int or callable, optional
        Processes to evaluate the population across, -1 meaning every core, or a
        map-like callable. Anything but 1 forces ``updating="deferred"``. Defaults to 1.
    integrality : sequence of bool or numpy.ndarray, optional
        Which parameters are constrained to integers. Defaults to None, meaning none
        are.
    vectorized : bool, optional
        Call the objective once per generation with an ``(d, population_size)`` array
        rather than once per candidate. Defaults to False.
    """

    strategy: str | Callable[..., Any] = "best1bin"
    maxiter: int | None = None
    popsize: int = 15
    tol: float | None = UNSET
    mutation: float | tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    rng: int | np.random.Generator | None = None
    init: str | np.ndarray = "sobol"
    atol: float = 0.0
    updating: str = "immediate"
    workers: int | Callable[..., Any] = 1
    integrality: Sequence[bool] | np.ndarray | None = None
    vectorized: bool = False

    _excluded: ClassVar[frozenset[str]] = frozenset()
    _tol_options: ClassVar[Mapping[str, float]] = {"tol": 0.01}
    requires_bounds: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if isinstance(self.strategy, str) and self.strategy not in DE_STRATEGY_OPTIONS:
            raise ValueError(
                f"strategy must be one of {DE_STRATEGY_OPTIONS} or a callable; "
                f"got {self.strategy!r}"
            )

        if isinstance(self.init, str) and self.init not in DE_INIT_OPTIONS:
            raise ValueError(
                f"init must be one of {DE_INIT_OPTIONS} or an array; got {self.init!r}"
            )

    @property
    def method_name(self) -> str:
        return "differential_evolution"

    def default_budget(self, n: int) -> int:
        return max(1000, 200 * n)

    def solver_function(self) -> Callable[..., Any]:
        # differential_evolution imports this package, so the import cannot be at module
        # scope.
        from better_optimize.differential_evolution import differential_evolution

        return differential_evolution

    def build_solver_kwargs(self, problem: SolverProblem) -> dict[str, Any]:
        if "bounds" not in problem.solver_kwargs:
            raise TypeError(
                "differential_evolution searches a bounded region, so bounds is required."
            )
        if problem.jac is not None or problem.hess is not None or problem.hessp is not None:
            raise TypeError(
                "differential_evolution uses no derivative information, so it cannot take "
                "jac, hess, or hessp."
            )
        if problem.progressbar_update_interval != 1:
            raise TypeError(
                "differential_evolution reports once per generation, so it cannot take "
                "progressbar_update_interval."
            )

        return {
            "f": problem.f,
            "x0": problem.x0,
            "args": problem.args,
            "callback": problem.callback,
            "progressbar": problem.progressbar,
            "progress_task": problem.progress_task,
            "verbose": problem.verbose,
            **problem.solver_kwargs,
            **self.optimizer_kwargs(n=len(problem.x0)),
        }
