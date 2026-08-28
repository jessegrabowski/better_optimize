from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from better_optimize.configuration.base import MinimizeConfig

__all__ = ["NelderMeadConfig", "PowellConfig"]


@dataclass
class NelderMeadConfig(MinimizeConfig):
    """Nelder-Mead simplex, a direct search method using no derivative information.

    `maxiter` and `maxfev` interact: setting only one lifts the other to infinity, so a
    run capped by iterations is not also capped by evaluations unless both are given.

    Parameters
    ----------
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``200 * n``.
    maxfev : int, optional
        Maximum number of function evaluations. Defaults to ``200 * n``.
    disp : bool, optional
        Print scipy's own convergence message. Defaults to False.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    initial_simplex : ndarray, optional
        Starting simplex of shape ``(n + 1, n)``, which replaces `x0` as the starting point.
        Defaults to None, letting scipy build one around `x0`.
    xatol : float, optional
        Absolute tolerance on `x` for convergence. Defaults to 1e-4.
    fatol : float, optional
        Absolute tolerance on the function value for convergence. Defaults to 1e-4.
    adaptive : bool, optional
        Adapt the simplex reflection, expansion and contraction coefficients to the problem
        dimension, which helps in high dimensions. Defaults to False.
    """

    maxiter: int | None = None
    maxfev: int | None = None
    disp: bool = False
    return_all: bool = False
    initial_simplex: np.ndarray | None = None
    xatol: float = 1e-4
    fatol: float = 1e-4
    adaptive: bool = False

    uses_grad: ClassVar[bool] = False
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[tuple[str, ...]] = ("xatol", "fatol")
    _budget_options: ClassVar[tuple[str, ...]] = ("maxiter", "maxfev")

    @property
    def method_name(self) -> str:
        return "nelder-mead"

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return self._finalize(
            {
                "maxiter": self.maxiter,
                "maxfev": self.maxfev,
                "disp": self.disp,
                "return_all": self.return_all,
                "initial_simplex": self.initial_simplex,
                "xatol": self.xatol,
                "fatol": self.fatol,
                "adaptive": self.adaptive,
            }
        )


@dataclass
class PowellConfig(MinimizeConfig):
    """Powell's conjugate direction method, a direct search using no derivative information.

    `maxiter` and `maxfev` interact the same way as for :class:`NelderMeadConfig`.

    Parameters
    ----------
    xtol : float, optional
        Relative tolerance on `x` for convergence. The line search uses ``xtol * 100``.
        Defaults to 1e-4.
    ftol : float, optional
        Relative tolerance on the function value for convergence. Defaults to 1e-4.
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``1000 * n``.
    maxfev : int, optional
        Maximum number of function evaluations. Defaults to ``1000 * n``.
    disp : bool, optional
        Print scipy's own convergence message. Defaults to False.
    direc : ndarray, optional
        Initial set of search directions, shape ``(n, n)``. scipy warns if it is rank
        deficient. Defaults to None, meaning the unit vectors.
    return_all : bool, optional
        Return the full list of iterates on the result object. Defaults to False.
    """

    xtol: float = 1e-4
    ftol: float = 1e-4
    maxiter: int | None = None
    maxfev: int | None = None
    disp: bool = False
    direc: np.ndarray | None = None
    return_all: bool = False

    uses_grad: ClassVar[bool] = False
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[tuple[str, ...]] = ("xtol", "ftol")
    _budget_options: ClassVar[tuple[str, ...]] = ("maxiter", "maxfev")

    @property
    def method_name(self) -> str:
        return "powell"

    def default_maxiter(self, n: int) -> int:
        return 1000 * n

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return self._finalize(
            {
                "xtol": self.xtol,
                "ftol": self.ftol,
                "maxiter": self.maxiter,
                "maxfev": self.maxfev,
                "disp": self.disp,
                "direc": self.direc,
                "return_all": self.return_all,
            }
        )
