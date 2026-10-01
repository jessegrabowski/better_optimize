from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_origin

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


@dataclass(frozen=True, eq=False)
class MinimizeConfig(ABC):
    """One scipy ``minimize`` method, its options, and what it needs from the objective.

    Each subclass declares one field per option the method accepts, named and typed as
    scipy's own signature has them. The fields are therefore the complete and authoritative
    option list: passing anything else raises ``TypeError`` rather than the
    ``OptimizeWarning`` scipy would emit and drop.

    An option the caller leaves unset is omitted from :meth:`optimizer_kwargs` rather than
    sent, so scipy applies its own default and a config never names an option the installed
    scipy does not have.

    Instances are frozen and compared by identity. One config is often shared across many
    runs -- `multi_optimize` splats a single set of solver arguments into every start --
    so a config that could be written to would carry one run's state into the next. Array
    fields make value equality impossible, so there is none.

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check a concrete subclass's declarations against its fields, at import time.

        Raises
        ------
        TypeError
            If a capability flag is neither a bool nor a property, an option group names
            something that is not a field, or a field defaults to `UNSET` without
            appearing in ``_tol_options``.
        """
        super().__init_subclass__(**kwargs)

        if getattr(cls.method_name, "__isabstractmethod__", False):
            return

        # A config composing another delegates its capabilities, so a property declares
        # them just as a ClassVar does.
        for flag in ("uses_grad", "uses_hess", "uses_hessp"):
            if not isinstance(getattr(cls, flag, None), bool | property):
                raise TypeError(f"{cls.__name__} must set {flag}")

        declared = cls._declared_fields()
        for group in ("_excluded", "_tol_options", "_iteration_options", "_evaluation_options"):
            unknown = set(getattr(cls, group)) - declared
            if unknown:
                raise TypeError(f"{cls.__name__}.{group} names non-fields: {sorted(unknown)}")

        unresolved = {name for name in declared if getattr(cls, name, None) is UNSET} - set(
            cls._tol_options
        )
        if unresolved:
            raise TypeError(
                f"{cls.__name__} defaults {sorted(unresolved)} to UNSET without listing "
                f"them in _tol_options, so the sentinel would reach scipy"
            )

    def __post_init__(self) -> None:
        # A method whose own option is named ``tol`` shares this one field with the
        # convenience knob, where unset means "no tol given" rather than "fill from tol".
        requested = None if self.tol is UNSET else self.tol

        for name, scipy_default in self._tol_options.items():
            if getattr(self, name) is UNSET:
                resolved = scipy_default if requested is None else requested
                object.__setattr__(self, name, resolved)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """The string scipy knows this method by."""

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
        # scipy writes through some array options -- Powell reorders ``direc`` in place --
        # so a config reused across runs would carry one run's state into the next.
        options = {
            field.name: self._copy_if_array(getattr(self, field.name))
            for field in fields(self)
            if field.name not in self._excluded
        }

        if n is not None:
            for name in self._budget_options():
                if options[name] is None:
                    options[name] = self.default_budget(n)

        # An option the caller never set is omitted rather than sent as None. scipy's own
        # default for each of these is None too, so the run is unchanged, and omitting means
        # a config never names an option the installed scipy has not heard of.
        return {name: value for name, value in options.items() if value is not None}

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

    @classmethod
    def _budget_options(cls) -> tuple[str, ...]:
        return (*cls._iteration_options, *cls._evaluation_options)

    @classmethod
    def _annotations(cls) -> dict[str, Any]:
        annotations: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            annotations.update(getattr(klass, "__annotations__", {}))

        return annotations

    @classmethod
    def _declared_fields(cls) -> set[str]:
        return {
            name
            for name, annotation in cls._annotations().items()
            if get_origin(annotation) is not ClassVar
        }

    @staticmethod
    def _copy_if_array(value: Any) -> Any:
        return value.copy() if isinstance(value, np.ndarray) else value
