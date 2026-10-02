from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from better_optimize.configuration.base import UNSET, RootConfig
from better_optimize.configuration.root_jac_options import (
    AndersonJacOptions,
    BroydenJacOptions,
    DiagonalJacOptions,
    ExcitingMixingJacOptions,
    JacOptions,
    KrylovJacOptions,
)

__all__ = [
    "AndersonConfig",
    "Broyden1Config",
    "Broyden2Config",
    "DiagBroydenConfig",
    "ExcitingMixingConfig",
    "KrylovConfig",
    "LinearMixingConfig",
    "QuasiNewtonRootConfig",
]

RESIDUAL_TOL = float(np.finfo(np.float64).eps ** (1 / 3))
"""scipy's absolute residual tolerance when none is given, from ``TerminationCondition``."""


@dataclass(frozen=True, eq=False)
class QuasiNewtonRootConfig(RootConfig, ABC):
    """Options shared by the seven root methods scipy solves with a quasi-Newton jacobian.

    All seven reach the same function, so they share one option surface exactly. What
    distinguishes them is `jac_options`, the nested configuration of the jacobian
    approximation each one builds.

    Three of the four tolerances default to infinity, so an unset run stops on `fatol`
    alone. Passing `tol` does not fill them the way it does for a minimize method: scipy
    routes it to `xtol` and leaves the other three disabled.

    Parameters
    ----------
    nit : int, optional
        Run exactly this many iterations and ignore the tolerances. Defaults to None. When
        `maxiter` is unset, this also caps it, at ``nit + 1``.
    disp : bool, optional
        Print the convergence history to stdout. Defaults to False.
    maxiter : int, optional
        Maximum number of iterations. Defaults to None, meaning ``100 * (n + 1)``.
    ftol : float, optional
        Relative tolerance on the residual norm. Defaults to infinity, which disables it.
    fatol : float, optional
        Absolute tolerance on the residual norm, and the only one of the four enabled by
        default. Defaults to the cube root of machine epsilon.
    xtol : float, optional
        Relative tolerance on the step. Defaults to infinity, which disables it.
    xatol : float, optional
        Absolute tolerance on the step. Defaults to infinity, which disables it.
    tol_norm : callable, optional
        Norm used in every tolerance test. Defaults to None, meaning the maximum norm.
    line_search : {"armijo", "wolfe"} or None, optional
        Line search used on the Newton direction. Defaults to "armijo".
    jac_options : JacOptions, optional
        Configuration of the jacobian approximation. Defaults to None, meaning scipy's own
        defaults for whichever one this method builds.
    """

    nit: int | None = None
    disp: bool = False
    maxiter: int | None = None
    ftol: float = UNSET
    fatol: float = UNSET
    xtol: float = UNSET
    xatol: float = UNSET
    tol_norm: Any = None
    line_search: str | None = UNSET
    jac_options: JacOptions | None = None

    uses_jac: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {
        "ftol": np.inf,
        "fatol": RESIDUAL_TOL,
        "xtol": np.inf,
        "xatol": np.inf,
    }
    # None disables the line search, so it cannot double as "the caller said nothing".
    _nullable_options: ClassVar[Mapping[str, Any]] = {"line_search": "armijo"}
    _iteration_options: ClassVar[tuple[str, ...]] = ("maxiter",)
    _evaluation_options: ClassVar[tuple[str, ...]] = ()

    _jac_options_type: ClassVar[type[JacOptions]]
    """The nested type this method's jacobian takes, which `jac_options` is checked against."""

    @classmethod
    def _check_declarations(cls) -> None:
        """Also check that the subclass names the nested type its jacobian builds.

        Raises
        ------
        TypeError
            If it names none, or names something that is not a `JacOptions`.
        """
        super()._check_declarations()

        declared = getattr(cls, "_jac_options_type", None)
        if not (isinstance(declared, type) and issubclass(declared, JacOptions)):
            raise TypeError(
                f"{cls.__name__} must name the JacOptions type its jacobian builds, "
                f"because the narrowed annotation on jac_options does not enforce itself"
            )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.jac_options is None:
            return

        # scipy's own documentation shows a plain mapping here, so accept one and let the
        # declared type check the keys.
        if isinstance(self.jac_options, Mapping):
            object.__setattr__(self, "jac_options", self._jac_options_type(**self.jac_options))
        elif not isinstance(self.jac_options, self._jac_options_type):
            raise TypeError(
                f"{self.method_name} builds a {self._jac_options_type.__name__}, so "
                f"jac_options cannot be a {type(self.jac_options).__name__}"
            )

    def _resolve_tolerances(self) -> None:
        """scipy routes a top-level `tol` to `xtol` and disables the other three, rather
        than filling each one the way ``minimize`` does."""
        filled = (
            {"xtol": self.tol, "xatol": np.inf, "ftol": np.inf, "fatol": np.inf}
            if self.tol is not None
            else dict(self._tol_options)
        )

        for name, value in filled.items():
            if getattr(self, name) is UNSET:
                object.__setattr__(self, name, value)

    def default_budget(self, n: int) -> int:
        return 100 * (n + 1)

    def optimizer_kwargs(self, n: int | None = None) -> dict[str, Any]:
        options = super().optimizer_kwargs(n)
        jac_options = options.get("jac_options")

        if jac_options is not None:
            options["jac_options"] = jac_options.as_dict()

        return options


@dataclass(frozen=True, eq=False)
class Broyden1Config(QuasiNewtonRootConfig):
    """Broyden's first method, updating an approximate inverse jacobian."""

    jac_options: BroydenJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = BroydenJacOptions

    @property
    def method_name(self) -> str:
        return "broyden1"


@dataclass(frozen=True, eq=False)
class Broyden2Config(QuasiNewtonRootConfig):
    """Broyden's second method, updating an approximate jacobian."""

    jac_options: BroydenJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = BroydenJacOptions

    @property
    def method_name(self) -> str:
        return "broyden2"


@dataclass(frozen=True, eq=False)
class AndersonConfig(QuasiNewtonRootConfig):
    """Anderson mixing, extrapolating from the last few residuals."""

    jac_options: AndersonJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = AndersonJacOptions

    @property
    def method_name(self) -> str:
        return "anderson"


@dataclass(frozen=True, eq=False)
class LinearMixingConfig(QuasiNewtonRootConfig):
    """Scalar mixing, using a multiple of the identity as the inverse jacobian."""

    jac_options: DiagonalJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = DiagonalJacOptions

    @property
    def method_name(self) -> str:
        return "linearmixing"


@dataclass(frozen=True, eq=False)
class DiagBroydenConfig(QuasiNewtonRootConfig):
    """Diagonal Broyden updates, which keep only the jacobian's diagonal."""

    jac_options: DiagonalJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = DiagonalJacOptions

    @property
    def method_name(self) -> str:
        return "diagbroyden"


@dataclass(frozen=True, eq=False)
class ExcitingMixingConfig(QuasiNewtonRootConfig):
    """Diagonal mixing whose entries are adapted per coordinate."""

    jac_options: ExcitingMixingJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = ExcitingMixingJacOptions

    @property
    def method_name(self) -> str:
        return "excitingmixing"


@dataclass(frozen=True, eq=False)
class KrylovConfig(QuasiNewtonRootConfig):
    """Newton-Krylov, solving each step with an inner iterative solver.

    This is the method for large systems, where forming a jacobian is the cost.
    """

    jac_options: KrylovJacOptions | None = None

    _jac_options_type: ClassVar[type[JacOptions]] = KrylovJacOptions

    @property
    def method_name(self) -> str:
        return "krylov"
