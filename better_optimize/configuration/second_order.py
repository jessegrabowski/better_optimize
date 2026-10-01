from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from better_optimize.configuration.base import (
    SQRT_EPS,
    UNSET,
    MinimizeConfig,
    Workers,
)

__all__ = [
    "DoglegConfig",
    "NewtonCGConfig",
    "TrustExactConfig",
    "TrustKrylovConfig",
    "TrustNCGConfig",
    "TrustRegionConfig",
]


@dataclass
class NewtonCGConfig(MinimizeConfig):
    r"""Newton conjugate gradient, solving the Newton step iteratively.

    This method stops on the step size rather than the gradient norm, so it has no ``gtol``.
    The inner conjugate gradient loop is capped at ``20 * n`` and is not configurable.

    Parameters
    ----------
    xtol : float, optional
        Terminate successfully when the average relative change in `x` falls below this.
        Scaled internally by `n`. Defaults to 1e-5.
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient, used only when no gradient
        function is supplied. Defaults to :math:`\sqrt{\epsilon}` for float64.
    maxiter : int, optional
        Maximum number of outer iterations. Defaults to ``200 * n``.
    disp : bool, optional
        Print scipy's own convergence message. Defaults to False.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    c1 : float, optional
        Armijo parameter for the line search, satisfying ``0 < c1 < c2 < 1``. Defaults to
        1e-4.
    c2 : float, optional
        Curvature parameter for the line search. Defaults to 0.9.
    workers : int or map-like callable, optional
        Parallelize the finite-difference derivatives. Defaults to None.
    """

    xtol: float = UNSET
    eps: float | np.ndarray = SQRT_EPS
    maxiter: int | None = None
    disp: bool = False
    return_all: bool = False
    c1: float = 1e-4
    c2: float = 0.9
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = True
    uses_hessp: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {"xtol": 1e-5}

    @property
    def method_name(self) -> str:
        return "Newton-CG"


@dataclass
class TrustRegionConfig(MinimizeConfig, ABC):
    """Shared options for the four trust-region methods.

    All four route through scipy's ``_minimize_trust_region`` driver and therefore accept
    exactly the same options; they differ only in the subproblem they solve and in which
    of these options that subproblem actually reads. Each subclass documents its own
    inert options. Instances of the concrete subclasses are what a caller builds.

    Parameters
    ----------
    initial_trust_radius : float, optional
        Initial trust region radius, which must be positive and below `max_trust_radius`.
        Defaults to 1.0.
    max_trust_radius : float, optional
        Largest radius the region may grow to, bounding the longest proposed step. Defaults
        to 1000.0.
    eta : float, optional
        Acceptance threshold on the ratio of actual to predicted reduction, required to lie
        in ``[0, 0.25)``. Defaults to 0.15.
    gtol : float, optional
        Terminate successfully when the gradient norm falls below this. Defaults to 1e-4.
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``200 * n``.
    disp : bool, optional
        Print scipy's own convergence message. Defaults to False.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    inexact : bool, optional
        Solve the subproblem to lower accuracy, which is faster per iteration. Read only by
        :class:`TrustKrylovConfig`. Defaults to True.
    workers : int or map-like callable, optional
        Parallelize the finite-difference Hessian. Has no effect for the methods requiring
        a callable Hessian. Defaults to None.
    subproblem_maxiter : int, optional
        Iteration cap for the subproblem solver. Read only by :class:`TrustExactConfig`.
        Defaults to None, meaning 25 there.
    """

    initial_trust_radius: float = 1.0
    max_trust_radius: float = 1000.0
    eta: float = 0.15
    gtol: float = UNSET
    maxiter: int | None = None
    disp: bool = False
    return_all: bool = False
    inexact: bool = True
    workers: Workers = None
    subproblem_maxiter: int | None = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {"gtol": 1e-4}


@dataclass
class DoglegConfig(TrustRegionConfig):
    """Trust region with the dogleg subproblem, needing a positive definite Hessian.

    scipy raises if the Hessian is not positive definite at any iterate. The Hessian must
    be a callable, so ``workers`` has no effect, and the dogleg subproblem takes no
    iteration cap, so ``subproblem_maxiter`` and ``inexact`` are both inert. See
    :class:`TrustRegionConfig` for the shared options.
    """

    uses_hessp: ClassVar[bool] = False

    @property
    def method_name(self) -> str:
        return "dogleg"


@dataclass
class TrustNCGConfig(TrustRegionConfig):
    """Trust region with the Steihaug conjugate gradient subproblem.

    The subproblem's tolerance is derived from the gradient magnitude and takes no
    iteration cap, so ``subproblem_maxiter`` and ``inexact`` are both inert. See
    :class:`TrustRegionConfig` for the shared options.
    """

    uses_hessp: ClassVar[bool] = True

    @property
    def method_name(self) -> str:
        return "trust-ncg"


@dataclass
class TrustExactConfig(TrustRegionConfig):
    """Trust region solving the subproblem almost exactly, needing a callable Hessian.

    This is the only method that reads ``subproblem_maxiter``, whose effective default is
    25. The Hessian must be a callable, so ``workers`` has no effect, and ``inexact`` is
    inert. See :class:`TrustRegionConfig` for the shared options.
    """

    uses_hessp: ClassVar[bool] = False

    @property
    def method_name(self) -> str:
        return "trust-exact"


@dataclass
class TrustKrylovConfig(TrustRegionConfig):
    """Trust region with the trlib Krylov subproblem, suited to large problems.

    This is the only method that reads ``inexact``, which selects between two internal
    tolerance pairs rather than being passed through. ``subproblem_maxiter`` is inert.
    See :class:`TrustRegionConfig` for the shared options.
    """

    uses_hessp: ClassVar[bool] = True

    @property
    def method_name(self) -> str:
        return "trust-krylov"
