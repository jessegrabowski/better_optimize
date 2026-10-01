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

__all__ = ["BFGSConfig", "CGConfig", "LBFGSBConfig", "TNCConfig"]


@dataclass(frozen=True, eq=False)
class BFGSConfig(MinimizeConfig):
    r"""Broyden-Fletcher-Goldfarb-Shanno, a quasi-Newton method with a full inverse Hessian.

    Parameters
    ----------
    gtol : float, optional
        Terminate successfully when the gradient norm falls below this. Defaults to 1e-5.
    norm : float, optional
        Order of the norm used for `gtol`, where Inf is the maximum component and -Inf the
        minimum. Defaults to Inf.
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient, used only when no gradient
        function is supplied and ignored when the gradient is estimated by a named finite
        difference scheme, which uses `finite_diff_rel_step` instead. Defaults to
        :math:`\sqrt{\epsilon}` for float64.
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``200 * n``.
    disp : bool, optional
        Print scipy's own convergence message, independently of the progress bar. Defaults
        to False.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for the gradient when it is estimated by a named finite
        difference scheme. Defaults to None, letting scipy choose per component.
    xrtol : float, optional
        Terminate successfully when the step falls below ``xk * xrtol``. Defaults to 0,
        which disables the test.
    c1 : float, optional
        Armijo parameter for the line search, satisfying ``0 < c1 < c2 < 1``. Defaults to
        1e-4.
    c2 : float, optional
        Curvature parameter for the line search, satisfying ``0 < c1 < c2 < 1``. Defaults
        to 0.9.
    hess_inv0 : ndarray, optional
        Initial inverse Hessian estimate of shape ``(n, n)``, which scipy rejects unless it
        is positive definite. Defaults to None, meaning the identity.
    workers : int or map-like callable, optional
        Parallelize the finite-difference gradient. Has no effect when a gradient function
        is supplied, and never parallelizes the objective itself. Defaults to None.
    """

    gtol: float = UNSET
    norm: float = np.inf
    eps: float | np.ndarray = SQRT_EPS
    maxiter: int | None = None
    disp: bool = False
    return_all: bool = False
    finite_diff_rel_step: FiniteDiffStep = None
    xrtol: float = 0
    c1: float = 1e-4
    c2: float = 0.9
    hess_inv0: np.ndarray | None = None
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"gtol": 1e-5}

    @property
    def method_name(self) -> str:
        return "BFGS"


@dataclass(frozen=True, eq=False)
class CGConfig(MinimizeConfig):
    r"""Nonlinear conjugate gradient, Polak-Ribiere variant.

    Parameters
    ----------
    gtol : float, optional
        Terminate successfully when the gradient norm falls below this. Defaults to 1e-5.
    norm : float, optional
        Order of the norm used for `gtol`. Defaults to Inf.
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient, used only when no gradient
        function is supplied. Defaults to :math:`\sqrt{\epsilon}` for float64.
    maxiter : int, optional
        Maximum number of iterations. This method has no cap on function evaluations.
        Defaults to ``200 * n``.
    disp : bool, optional
        Print scipy's own convergence message. Defaults to False.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for a named finite-difference gradient scheme. Defaults to None.
    c1 : float, optional
        Armijo parameter for the line search. Defaults to 1e-4.
    c2 : float, optional
        Curvature parameter for the line search. Lower than the BFGS default because
        conjugate gradient needs a stricter curvature condition to stay descent-directed.
        Defaults to 0.4.
    workers : int or map-like callable, optional
        Parallelize the finite-difference gradient. Defaults to None.
    """

    gtol: float = UNSET
    norm: float = np.inf
    eps: float | np.ndarray = SQRT_EPS
    maxiter: int | None = None
    disp: bool = False
    return_all: bool = False
    finite_diff_rel_step: FiniteDiffStep = None
    c1: float = 1e-4
    c2: float = 0.4
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"gtol": 1e-5}

    @property
    def method_name(self) -> str:
        return "CG"


@dataclass(frozen=True, eq=False)
class LBFGSBConfig(MinimizeConfig):
    r"""Limited-memory BFGS with box constraints.

    Parameters
    ----------
    maxcor : int, optional
        Number of stored correction pairs defining the limited-memory Hessian
        approximation, passed to the Fortran core as ``m``. Defaults to 10.
    ftol : float, optional
        Terminate when ``(f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol``, passed to the
        core as ``factr = ftol / eps``. Defaults to 2.220446049250313e-09, the classic
        ``factr=1e7``.
    gtol : float, optional
        Terminate when the largest component of the projected gradient falls below this,
        passed to the core as ``pgtol``. Defaults to 1e-5.
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient. Defaults to 1e-8, which is
        not the :math:`\sqrt{\epsilon}` used by :class:`BFGSConfig` and :class:`CGConfig`.
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``200 * n`` once the solver supplies the
        problem dimension, overriding scipy's own default of 15000.
    maxfun : int, optional
        Maximum number of function evaluations, checked after `maxiter` within an iteration
        and so able to overshoot slightly. Defaults to ``200 * n`` once the solver supplies
        the problem dimension, overriding scipy's own default of 15000.
    maxls : int, optional
        Maximum line search steps per iteration, which scipy requires to be positive.
        Defaults to 20.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for a named finite-difference gradient scheme. Defaults to None.
    workers : int or map-like callable, optional
        Parallelize the finite-difference gradient. Defaults to None.
    disp : bool, optional
        Deprecated no-op, slated for removal in scipy 1.18. Setting it emits a
        ``DeprecationWarning`` and changes nothing. Defaults to None.
    iprint : int, optional
        Deprecated no-op, slated for removal in scipy 1.18. Setting it emits a
        ``DeprecationWarning`` and changes nothing. Defaults to None.
    """

    maxcor: int = 10
    ftol: float = UNSET
    gtol: float = UNSET
    eps: float | np.ndarray = 1e-8
    maxiter: int | None = None
    maxfun: int | None = None
    maxls: int = 20
    finite_diff_rel_step: FiniteDiffStep = None
    workers: Workers = None
    disp: bool | None = None
    iprint: int | None = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"ftol": 2.220446049250313e-09, "gtol": 1e-5}
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfun",)

    @property
    def method_name(self) -> str:
        return "L-BFGS-B"


@dataclass(frozen=True, eq=False)
class TNCConfig(MinimizeConfig):
    r"""Truncated Newton with box constraints, wrapping Nash's Fortran code.

    This method has no ``maxiter`` option; `maxfun` is its only budget. Several options
    take a negative sentinel meaning "let the solver choose", and the value each resolves
    to is given below -- do not normalize these to None, because the solver reads the sign.

    Parameters
    ----------
    eps : float or ndarray, optional
        Absolute step size for the forward-difference gradient. Defaults to 1e-8.
    scale : ndarray, optional
        Per-variable scaling factors. Defaults to None, meaning unit scaling for bounded
        variables and ``1 + |x0|`` otherwise.
    offset : ndarray, optional
        Per-variable offset subtracted before scaling. Defaults to None.
    mesg_num : int, optional
        Verbosity from 0 (silent) through 5 (everything), overriding `disp` when set. Absent
        from scipy's own option list but functional. Defaults to None.
    maxCGit : int, optional
        Maximum Hessian-vector evaluations per iteration, where 0 gives steepest descent.
        Defaults to -1, resolving to ``max(1, min(50, n / 2))``.
    eta : float, optional
        Severity of the line search. Defaults to -1; any value outside ``[0, 1]`` resolves
        to 0.25.
    stepmx : float, optional
        Maximum step for the line search. Defaults to 0; too small a value resolves to 10.0.
    accuracy : float, optional
        Relative precision of the finite-difference calculations. Defaults to 0; at or below
        machine epsilon it resolves to :math:`\sqrt{\epsilon}`.
    minfev : float, optional
        Estimate of the minimum function value, passed to the core as ``fmin``. Defaults
        to 0.
    ftol : float, optional
        Precision goal on the function value. Defaults to -1, resolving to 0.0.
    xtol : float, optional
        Precision goal on `x`. Defaults to -1, resolving to :math:`\sqrt{\epsilon}`.
    gtol : float, optional
        Precision goal on the projected gradient, passed to the core as ``pgtol``. Defaults
        to -1, resolving to ``1e-2 * sqrt(accuracy)``.
    rescale : float, optional
        Log10 scaling factor triggering rescaling of the function value. Defaults to -1,
        resolving to 1.3.
    disp : bool, optional
        Print convergence messages, ignored when `mesg_num` is set. Defaults to False.
    finite_diff_rel_step : float or ndarray, optional
        Relative step size for a named finite-difference gradient scheme. Defaults to None.
    maxfun : int, optional
        Maximum number of function evaluations. Defaults to ``max(100, 10 * n)``.
    workers : int or map-like callable, optional
        Parallelize the finite-difference gradient. Defaults to None.
    """

    eps: float | np.ndarray = 1e-8
    scale: np.ndarray | None = None
    offset: np.ndarray | None = None
    mesg_num: int | None = None
    maxCGit: int = -1
    eta: float = -1
    stepmx: float = 0
    accuracy: float = 0
    minfev: float = 0
    ftol: float = UNSET
    xtol: float = UNSET
    gtol: float = UNSET
    rescale: float = -1
    disp: bool = False
    finite_diff_rel_step: FiniteDiffStep = None
    maxfun: int | None = None
    workers: Workers = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"xtol": -1, "ftol": -1, "gtol": -1}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfun",)

    @property
    def method_name(self) -> str:
        return "TNC"

    def default_budget(self, n: int) -> int:
        return max(100, 10 * n)
