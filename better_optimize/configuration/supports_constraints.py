from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from better_optimize.configuration.base import (
    SQRT_EPS,
    UNSET,
    FiniteDiffStep,
    MinimizeConfig,
    Workers,
)

__all__ = ["COBYLAConfig", "SLSQPConfig", "TrustConstrConfig"]


@dataclass
class COBYLAConfig(MinimizeConfig):
    r"""Constrained optimization by linear approximation, using no derivative information.

    `maxiter` caps function evaluations here, not iterations, despite the name.

    Parameters
    ----------
    tol : float, optional
        Final value of the trust region radius, which is the convergence criterion. Defaults
        to 1e-4; None leaves scipy to apply that same default itself.
    rhobeg : float, optional
        Initial value of the trust region radius. Defaults to 1.0.
    maxiter : int, optional
        Maximum number of function evaluations. Defaults to 1000.
    disp : int, optional
        Verbosity from 0 to 3, which scipy rejects outside that range. Defaults to 0.
    catol : float, optional
        Absolute tolerance on constraint violation, which also decides whether the result
        is reported as successful. Defaults to None, meaning
        :math:`\sqrt{\epsilon}` for float64.
    f_target : float, optional
        Stop as soon as the objective falls to this value. Defaults to negative infinity,
        which never triggers.
    """

    tol: float | None = UNSET
    rhobeg: float = 1.0
    maxiter: int | None = None
    disp: int = 0
    catol: float | None = None
    f_target: float = -np.inf

    uses_grad: ClassVar[bool] = False
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _excluded: ClassVar[frozenset[str]] = frozenset()
    _tol_options: ClassVar[Mapping[str, float]] = {"tol": 1e-4}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxiter",)

    @property
    def method_name(self) -> str:
        return "COBYLA"

    def default_budget(self, n: int) -> int:
        return 1000


@dataclass
class SLSQPConfig(MinimizeConfig):
    r"""Sequential least squares programming, for problems with equality and bound constraints.

    Parameters
    ----------
    maxiter : int, optional
        Maximum number of major iterations. Defaults to 100.
    ftol : float, optional
        Convergence tolerance on the objective. Defaults to 1e-6.
    iprint : int, optional
        Verbosity, where 2 and above prints per-iterate rows and 1 prints only a summary.
        Absent from scipy's own option list, and inert unless `disp` is True. Defaults to 1.
    disp : bool, optional
        Enable printing at all. When False, `iprint` is forced to 0. Defaults to False.
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient, used for the objective and
        for any constraint that supplies no Jacobian. Defaults to
        :math:`\sqrt{\epsilon}` for float64.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for a named finite-difference gradient scheme. Defaults to None.
    workers : int or map-like callable, optional
        Parallelize the objective's finite-difference gradient. Constraint Jacobians are
        not parallelized. Defaults to None.
    """

    maxiter: int | None = None
    ftol: float = UNSET
    iprint: int = 1
    disp: bool = False
    eps: float | np.ndarray = SQRT_EPS
    finite_diff_rel_step: FiniteDiffStep = None
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"ftol": 1e-6}

    @property
    def method_name(self) -> str:
        return "SLSQP"

    def default_budget(self, n: int) -> int:
        return 100


@dataclass
class TrustConstrConfig(MinimizeConfig):
    """Trust region interior point method, the most capable of the constrained solvers.

    Parameters
    ----------
    xtol : float, optional
        Terminate when the trust region radius falls below this. Defaults to 1e-8.
    gtol : float, optional
        Terminate when the infinity norm of the Lagrangian gradient falls below this.
        Defaults to 1e-8.
    barrier_tol : float, optional
        Terminate when the barrier parameter falls below this, for problems with inequality
        constraints. Defaults to 1e-8.
    sparse_jacobian : bool, optional
        Force the constraint Jacobians to be treated as sparse or dense. Defaults to None,
        letting scipy match whatever the constraints return.
    maxiter : int, optional
        Maximum number of iterations. Defaults to 1000.
    verbose : int, optional
        Verbosity from 0 to 3, where 2 prints a progress table. Defaults to 0.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for a named finite-difference gradient scheme. Defaults to None.
    initial_constr_penalty : float, optional
        Starting penalty on constraint violation in the merit function, trading progress
        against feasibility. Defaults to 1.0.
    initial_tr_radius : float, optional
        Initial trust region radius. Defaults to 1.0.
    initial_barrier_parameter : float, optional
        Starting barrier parameter, used only with inequality constraints. Defaults to 0.1.
    initial_barrier_tolerance : float, optional
        Starting barrier subproblem tolerance, used only with inequality constraints.
        Defaults to 0.1.
    factorization_method : str, optional
        How to factorize the constraint Jacobian, one of ``"NormalEquation"``,
        ``"AugmentedSystem"``, ``"QRFactorization"`` or ``"SVDFactorization"``. Defaults to
        None, letting scipy choose from the problem's sparsity.
    disp : bool, optional
        Raise `verbose` to 1 when it is 0. Defaults to False.
    workers : int or map-like callable, optional
        Parallelize the objective's finite-difference derivatives. Defaults to None.
    """

    xtol: float = UNSET
    gtol: float = UNSET
    barrier_tol: float = UNSET
    sparse_jacobian: bool | None = None
    maxiter: int | None = None
    verbose: int = 0
    finite_diff_rel_step: FiniteDiffStep = None
    initial_constr_penalty: float = 1.0
    initial_tr_radius: float = 1.0
    initial_barrier_parameter: float = 0.1
    initial_barrier_tolerance: float = 0.1
    factorization_method: str | None = None
    disp: bool = False
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = True
    uses_hessp: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {
        "xtol": 1e-8,
        "gtol": 1e-8,
        "barrier_tol": 1e-8,
    }

    @property
    def method_name(self) -> str:
        return "trust-constr"

    def default_budget(self, n: int) -> int:
        return 1000
