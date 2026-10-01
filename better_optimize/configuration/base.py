from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np

__all__ = ["UNSET", "FiniteDiffStep", "MinimizeConfig", "SQRT_EPS", "Workers"]

SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

Workers = int | Callable[..., Any] | None
FiniteDiffStep = float | np.ndarray | None


class _Unset:
    """Marks a tolerance the caller did not set, so ``tol`` can tell that from a value."""

    def __repr__(self) -> str:
        return "<unset>"


UNSET: Any = _Unset()


@dataclass
class MinimizeConfig(ABC):
    """One scipy ``minimize`` method, its options, and what it needs from the objective.

    Each subclass declares one field per option the method actually accepts, typed and
    defaulted to the value in scipy's own source. The fields are therefore the complete
    and authoritative option list: passing anything else raises ``TypeError`` rather than
    the ``OptimizeWarning`` scipy would emit and drop.

    Methods disagree about what their work budget is called, so there is no ``maxiter``
    field here. Each subclass lists the names scipy accepts in ``_iteration_options`` and
    ``_evaluation_options``.

    Parameters
    ----------
    tol : float, optional
        Convenience tolerance filling every option in ``_tol_options`` the caller did not
        pass, mirroring what ``scipy.optimize.minimize`` does with its own top-level
        ``tol``. Defaults to None, leaving each tolerance at its own default.
    """

    tol: float | None = None

    uses_grad: ClassVar[bool]
    uses_hess: ClassVar[bool]
    uses_hessp: ClassVar[bool]

    _excluded: ClassVar[frozenset[str]] = frozenset({"tol"})
    """Fields that are not options of the method."""

    _tol_options: ClassVar[Mapping[str, float]] = {}
    """Tolerance options that ``tol`` fills, mapped to scipy's default for each."""

    _iteration_options: ClassVar[tuple[str, ...]] = ("maxiter",)
    _evaluation_options: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # COBYLA's own option is named ``tol``, so the convenience knob and one target are
        # the same field; an unset one there means "no tol given", not "fill from tol".
        requested = None if self.tol is UNSET else self.tol

        for name, scipy_default in self._tol_options.items():
            if getattr(self, name) is UNSET:
                setattr(self, name, scipy_default if requested is None else requested)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """The string scipy knows this method by."""

    @classmethod
    def _budget_options(cls) -> tuple[str, ...]:
        return (*cls._iteration_options, *cls._evaluation_options)

    def default_budget(self, n: int) -> int:
        """The budget `better_optimize` applies to an `n`-dimensional problem by default."""
        return 200 * n

    def optimizer_kwargs(self, n: int | None = None) -> dict[str, Any]:
        """The options dictionary to hand to scipy.

        Parameters
        ----------
        n : int, optional
            Problem dimension. When given, every budget option the caller left unset is
            filled with :meth:`default_budget`, which is what makes that default
            authoritative rather than advisory. Defaults to None, which emits only the
            budgets the caller set and leaves scipy to apply its own.
        """
        options = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in self._excluded
        }

        # Several methods compare against their budget directly and would raise on a None,
        # so an unset budget is omitted rather than sent, letting scipy apply its own.
        for name in self._budget_options():
            if options[name] is not None:
                continue
            if n is None:
                del options[name]
            else:
                options[name] = self.default_budget(n)

        return options

    def evaluation_budget(self, n: int) -> int:
        """The cap on objective evaluations, for the wrapper that counts them.

        Prefers an evaluation-capping option the caller set, because that is what the
        counter compares against. Falls back to an iteration-capping one, then to
        :meth:`default_budget`.
        """
        for group in (self._evaluation_options, self._iteration_options):
            budgets = [getattr(self, name) for name in group if getattr(self, name) is not None]
            if budgets:
                return min(budgets)

        return self.default_budget(n)
