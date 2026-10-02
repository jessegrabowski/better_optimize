from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from better_optimize.configuration.base import UNSET, RootConfig

__all__ = ["DFSaneConfig", "HybrConfig", "LMConfig"]

MINPACK_TOL = 1.49012e-08
"""The square root of machine epsilon, as MINPACK spells it."""


@dataclass(frozen=True, eq=False)
class HybrConfig(RootConfig):
    """MINPACK's modified Powell method, the default root finder.

    Parameters
    ----------
    col_deriv : bool, optional
        Whether the jacobian function returns derivatives down the columns, which saves a
        transpose. Defaults to False.
    xtol : float, optional
        Terminate when the relative error between two consecutive iterates falls below
        this. Defaults to 1.49012e-08.
    maxfev : int, optional
        Maximum number of calls to the objective. Defaults to None, meaning
        ``200 * (n + 1)`` when no jacobian is supplied and ``100 * (n + 1)`` when one is.
    band : tuple of int, optional
        The counts of sub- and super-diagonals within the band of the jacobian, which
        declares it banded. Read only when no jacobian function is supplied. Defaults to
        None.
    eps : float, optional
        Step length for the forward-difference jacobian, used only when no jacobian
        function is supplied. A value below machine precision is taken to mean the
        objective itself is accurate to about machine precision. Defaults to None.
    factor : float, optional
        Sets the initial step bound, as ``factor * || diag * x ||``, and belongs in
        ``(0.1, 100)``. Defaults to 100.
    diag : sequence of float, optional
        Positive scale factors for the variables, one per dimension. Defaults to None.
    """

    col_deriv: bool = False
    xtol: float = UNSET
    maxfev: int | None = None
    band: tuple[int, int] | None = None
    eps: float | None = None
    factor: float = 100
    diag: Sequence[float] | np.ndarray | None = None

    uses_jac: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {"xtol": MINPACK_TOL}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfev",)

    @property
    def method_name(self) -> str:
        return "hybr"

    def default_budget(self, n: int) -> int:
        """scipy halves this to ``100 * (n + 1)`` when a jacobian function is supplied,
        which is the only value its own documentation states."""
        return 200 * (n + 1)


@dataclass(frozen=True, eq=False)
class LMConfig(RootConfig):
    """Levenberg-Marquardt, solving the system as a least-squares problem.

    `maxiter` caps function evaluations here, not iterations, despite the name: scipy
    forwards it to MINPACK as ``maxfev``.

    Parameters
    ----------
    col_deriv : int, optional
        Non-zero to declare that the jacobian function returns derivatives down the
        columns, which saves a transpose. Defaults to 0.
    ftol : float, optional
        Relative error desired in the sum of squares. Defaults to 1.49012e-08.
    xtol : float, optional
        Relative error desired in the approximate solution. Defaults to 1.49012e-08.
    gtol : float, optional
        Orthogonality desired between the residual vector and the columns of the jacobian.
        Defaults to 0.0.
    maxiter : int, optional
        Maximum number of calls to the objective. Defaults to None, meaning
        ``200 * (n + 1)`` when no jacobian is supplied and ``100 * (n + 1)`` when one is.
    eps : float, optional
        Step length for the forward-difference jacobian, used only when no jacobian
        function is supplied. A value below machine precision is taken to mean the
        objective itself is accurate to about machine precision. Defaults to 0.0.
    factor : float, optional
        Sets the initial step bound, as ``factor * || diag * x ||``, and belongs in
        ``(0.1, 100)``. Defaults to 100.
    diag : sequence of float, optional
        Positive scale factors for the variables, one per dimension. Defaults to None.
    """

    col_deriv: int = 0
    ftol: float = MINPACK_TOL
    xtol: float = UNSET
    gtol: float = 0.0
    maxiter: int | None = None
    eps: float = 0.0
    factor: float = 100
    diag: Sequence[float] | np.ndarray | None = None

    uses_jac: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {"xtol": MINPACK_TOL}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxiter",)

    @property
    def method_name(self) -> str:
        return "lm"

    def default_budget(self, n: int) -> int:
        """scipy halves this to ``100 * (n + 1)`` when a jacobian function is supplied,
        which is the only value its own documentation states."""
        return 200 * (n + 1)


@dataclass(frozen=True, eq=False)
class DFSaneConfig(RootConfig):
    r"""Derivative-free spectral residual method, for large systems with no jacobian.

    Parameters
    ----------
    ftol : float, optional
        Relative norm tolerance. Terminates when
        :math:`\|F(x)\| < \mathrm{fatol} + \mathrm{ftol}\,\|F(x_0)\|`. Defaults to 1e-08.
    fatol : float, optional
        Absolute norm tolerance in the same test. Defaults to 1e-300.
    maxfev : int, optional
        Maximum number of calls to the objective. Defaults to None, meaning 1000
        regardless of dimension.
    fnorm : callable, optional
        Norm used in the convergence check. Defaults to None, meaning the 2-norm.
    disp : bool, optional
        Print the convergence history to stdout. Defaults to False.
    M : int, optional
        Number of past iterates the nonmonotonic line search considers. Defaults to 10.
    eta_strategy : callable, optional
        Chooses the slack allowed for growth in :math:`\|F\|^2`, called as
        ``eta_strategy(k, x, F)``. Must be positive and summable over `k`. Defaults to
        None, meaning :math:`\|F\|^2 / (1 + k)^2`.
    sigma_eps : float, optional
        Bounds the spectral coefficient to ``sigma_eps < sigma < 1 / sigma_eps``. Defaults
        to 1e-10.
    sigma_0 : float, optional
        Initial spectral coefficient. Defaults to 1.0.
    line_search : {"cruz", "cheng"}, optional
        Which nonmonotonic line search to run. Defaults to "cruz".
    """

    ftol: float = UNSET
    fatol: float = 1e-300
    maxfev: int | None = None
    fnorm: Callable[..., Any] | None = None
    disp: bool = False
    M: int = 10
    eta_strategy: Callable[..., Any] | None = None
    sigma_eps: float = 1e-10
    sigma_0: float = 1.0
    line_search: str = "cruz"

    uses_jac: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"ftol": 1e-08}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfev",)

    @property
    def method_name(self) -> str:
        return "df-sane"

    def default_budget(self, n: int) -> int:
        return 1000
