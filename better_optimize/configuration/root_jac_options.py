from collections.abc import Callable
from dataclasses import dataclass, fields
from inspect import signature
from typing import Any

import numpy as np

from scipy.sparse.linalg import bicgstab, cgs, gmres, lgmres, minres, tfqmr

__all__ = [
    "AndersonJacOptions",
    "BroydenJacOptions",
    "DiagonalJacOptions",
    "ExcitingMixingJacOptions",
    "JacOptions",
    "KrylovJacOptions",
]

KRYLOV_SOLVERS: dict[str, Callable[..., Any]] = {
    "bicgstab": bicgstab,
    "gmres": gmres,
    "lgmres": lgmres,
    "cgs": cgs,
    "minres": minres,
    "tfqmr": tfqmr,
}
"""The inner solvers `KrylovJacOptions.method` selects by name."""


@dataclass(frozen=True, eq=False)
class JacOptions:
    """Options for the jacobian approximation a quasi-Newton root method builds.

    scipy takes these as one nested ``jac_options`` mapping and splats it into a jacobian
    class, so an unrecognized key raises there rather than being warned about and dropped.
    The fields are that class's parameters.
    """

    def as_dict(self) -> dict[str, Any]:
        """The ``jac_options`` mapping to hand scipy, less anything left unset."""
        values = {
            entry.name: getattr(self, entry.name)
            for entry in fields(self)
            if getattr(self, entry.name) is not None
        }

        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in values.items()
        }


@dataclass(frozen=True, eq=False)
class BroydenJacOptions(JacOptions):
    """Rank-one jacobian updates, for `broyden1` and `broyden2`.

    Parameters
    ----------
    alpha : float, optional
        Initial guess for the inverse jacobian, as a multiple of the identity. Defaults to
        None, meaning scipy estimates one from the first residual.
    reduction_method : str or tuple, optional
        How to collapse the update history once it reaches `max_rank`. One of "restart",
        "simple", or ``("svd", to_retain)``. Defaults to "restart".
    max_rank : int, optional
        Largest rank the update history may reach. Defaults to None, meaning no limit.
    """

    alpha: float | None = None
    reduction_method: str | tuple[str, int] = "restart"
    max_rank: int | None = None


@dataclass(frozen=True, eq=False)
class AndersonJacOptions(JacOptions):
    """Anderson mixing, which extrapolates from the last `M` residuals.

    Parameters
    ----------
    alpha : float, optional
        Initial guess for the inverse jacobian, as a multiple of the identity. Defaults to
        None, meaning scipy estimates one from the first residual.
    w0 : float, optional
        Regularization weight on the extrapolation, which stabilizes it at the cost of
        accuracy. Defaults to 0.01.
    M : int, optional
        Number of past vectors the extrapolation uses. Defaults to 5.
    """

    alpha: float | None = None
    w0: float = 0.01
    M: int = 5


@dataclass(frozen=True, eq=False)
class DiagonalJacOptions(JacOptions):
    """A scalar or diagonal jacobian, for `linearmixing` and `diagbroyden`.

    Parameters
    ----------
    alpha : float, optional
        Initial guess for the inverse jacobian, as a multiple of the identity. Defaults to
        None, meaning scipy estimates one from the first residual.
    """

    alpha: float | None = None


@dataclass(frozen=True, eq=False)
class ExcitingMixingJacOptions(JacOptions):
    """A diagonal jacobian whose entries are adapted per coordinate.

    Parameters
    ----------
    alpha : float, optional
        Initial guess for the inverse jacobian, as a multiple of the identity. Defaults to
        None, meaning scipy estimates one from the first residual.
    alphamax : float, optional
        Largest magnitude any diagonal entry may reach. Defaults to 1.0.
    """

    alpha: float | None = None
    alphamax: float = 1.0


@dataclass(frozen=True, eq=False)
class KrylovJacOptions(JacOptions):
    """A jacobian-vector product solved by an inner Krylov method.

    `inner_options` reaches the inner solver itself, whose accepted names depend on which
    one `method` selects. scipy warns and ignores a name that solver does not take, so
    this checks them at construction instead.

    Parameters
    ----------
    rdiff : float, optional
        Relative step size for the finite-difference jacobian-vector product. Defaults to
        None, meaning scipy derives one from machine precision.
    method : str or callable, optional
        Inner solver, named in `KRYLOV_SOLVERS` or supplied as a callable with the same
        signature. Defaults to "lgmres".
    inner_maxiter : int, optional
        Iteration cap handed to the inner solver. Defaults to 20.
    inner_M : sparse matrix or LinearOperator, optional
        Preconditioner for the inner solver. Defaults to None.
    outer_k : int, optional
        Number of vectors carried between nonlinear iterations, read only by "lgmres".
        Defaults to 10.
    inner_options : dict, optional
        Further options for the inner solver, keyed without the ``inner_`` prefix scipy
        spells them with. Defaults to None.

    Raises
    ------
    ValueError
        If `method` names no known solver, or `inner_options` names something that solver
        does not accept.
    """

    rdiff: float | None = None
    method: str | Callable[..., Any] = "lgmres"
    inner_maxiter: int = 20
    inner_M: Any = None
    outer_k: int = 10
    inner_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.method, str) and self.method not in KRYLOV_SOLVERS:
            raise ValueError(
                f"method must be one of {tuple(KRYLOV_SOLVERS)} or a callable; "
                f"got {self.method!r}"
            )

        requested = set(self.inner_options or ())

        unknown = requested - self._inner_parameters()
        if unknown:
            raise ValueError(
                f"inner_options {sorted(unknown)} are not accepted by the inner solver "
                f"{self.method!r}, which scipy would warn about and ignore"
            )

        shadowed = sorted(requested & self._field_inner_keywords())
        if shadowed:
            fields_shadowed = [self._field_for_inner_keyword(name) for name in shadowed]
            raise ValueError(
                f"inner_options {shadowed} override the fields {fields_shadowed}, which "
                f"would discard what those are set to; set them directly instead"
            )

    def as_dict(self) -> dict[str, Any]:
        emitted = super().as_dict()
        inner = emitted.pop("inner_options", {})

        return emitted | {f"inner_{name}": value for name, value in inner.items()}

    @classmethod
    def _field_inner_keywords(cls) -> set[str]:
        """The inner-solver keywords a field of this class already sets.

        scipy writes each ``inner_<name>`` option it is handed onto the inner keyword
        ``<name>``, on top of what the jacobian constructor set there from its own
        arguments. The ``inner_`` prefix is not reliable on this side: ``outer_k`` is
        spelled without one and still lands on the inner keyword of the same name.
        """
        return {
            entry.name.removeprefix("inner_")
            for entry in fields(cls)
            if entry.name != "inner_options"
        }

    @classmethod
    def _field_for_inner_keyword(cls, inner_name: str) -> str:
        declared = {entry.name for entry in fields(cls)}

        return inner_name if inner_name in declared else f"inner_{inner_name}"

    def _inner_parameters(self) -> set[str]:
        # __post_init__ has already refused a name that is not a key here.
        solver = KRYLOV_SOLVERS[self.method] if isinstance(self.method, str) else self.method

        return {
            name for name in signature(solver).parameters if name not in ("self", "args", "kwargs")
        }
