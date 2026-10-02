from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_origin

import numpy as np

from rich.progress import Progress, TaskID

__all__ = [
    "UNSET",
    "FiniteDiffStep",
    "MinimizeConfig",
    "OptimizerConfig",
    "RootConfig",
    "SolverProblem",
    "SQRT_EPS",
    "Workers",
]

SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

Workers = int | Callable[..., Any] | None
FiniteDiffStep = float | np.ndarray | None


class _Unset:
    """Marks a tolerance the caller did not set, so ``tol`` can tell that from a value."""

    def __repr__(self) -> str:
        return "<unset>"


UNSET: Any = _Unset()


@dataclass(frozen=True, eq=False)
class SolverProblem:
    """What :func:`minimize` was handed, for a configuration that shapes it into the call
    another entry point expects.

    It carries the problem and the reporting settings rather than any option of the method,
    so a configuration reads it without storing any of it.
    """

    f: Callable[..., Any]
    x0: np.ndarray
    jac: Callable[..., Any] | None
    hess: Callable[..., Any] | None
    hessp: Callable[..., Any] | None
    args: tuple[Any, ...]
    callback: Callable[..., Any] | None
    progressbar: bool | Progress
    progress_task: TaskID | None
    progressbar_update_interval: int
    verbose: bool
    solver_kwargs: dict[str, Any]
    """The arguments describing the problem rather than the method, such as ``bounds``."""


@dataclass(frozen=True, eq=False)
class OptimizerConfig(ABC):
    """One solver, and the options it accepts.

    Each subclass declares one field per option the solver accepts, named and typed as
    scipy's own signature has them. The fields are therefore the complete and authoritative
    option list: passing anything else raises ``TypeError`` rather than the
    ``OptimizeWarning`` scipy would emit and drop.

    :class:`MinimizeConfig` adds the flags saying which derivatives a scipy ``minimize``
    method uses. The global optimizers subclass this one instead, because neither has an
    answer of its own. Differential evolution uses no derivatives, and a basinhopping run
    uses whatever the minimizer it composes uses.

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

    _excluded: ClassVar[frozenset[str]] = frozenset({"tol"})
    """Fields that are not options of the method."""

    _tol_options: ClassVar[Mapping[str, float]] = {}
    """Tolerance options that ``tol`` fills, mapped to scipy's default for each."""

    _nullable_options: ClassVar[Mapping[str, Any]] = {}
    """Options where None is a value scipy acts on, mapped to scipy's default for each.

    The omit-unset rule in :meth:`optimizer_kwargs` cannot tell such a None apart from an
    option the caller never set, so these fields default to `UNSET` and are resolved here
    instead."""

    _iteration_options: ClassVar[tuple[str, ...]] = ("maxiter",)
    _evaluation_options: ClassVar[tuple[str, ...]] = ()

    _scipy_resolved_budgets: ClassVar[tuple[str, ...]] = ()
    """Budget options to leave out rather than fill with :meth:`default_budget`.

    scipy derives a few of these from something the configuration cannot see, such as
    whether a jacobian function was supplied, and reaches that branch only when the option
    is absent. Sending a number of our own would make the branch unreachable."""

    requires_bounds: ClassVar[bool] = False
    """Whether the solver searches a region rather than starting from a point.

    This answers the question for a caller that has to know before building the call, such
    as `sequential_optimize` deciding what to hand a stage. It is not where the requirement
    is enforced: a configuration that declares it refuses the call itself, in
    :meth:`build_solver_kwargs`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # An abstract intermediate declares only part of the contract, leaving the rest to
        # the concrete subclasses, so there is nothing to check yet.
        if not getattr(cls.method_name, "__isabstractmethod__", False):
            cls._check_declarations()

    @classmethod
    def _check_declarations(cls) -> None:
        """Check a concrete subclass's declarations against its fields, at import time.

        Raises
        ------
        TypeError
            If an option group names something that is not a field, or a field defaults to
            `UNSET` without appearing in ``_tol_options``.
        """
        declared = cls._declared_fields()
        for group in (
            "_excluded",
            "_tol_options",
            "_nullable_options",
            "_iteration_options",
            "_evaluation_options",
            "_scipy_resolved_budgets",
        ):
            unknown = set(getattr(cls, group)) - declared
            if unknown:
                raise TypeError(f"{cls.__name__}.{group} names non-fields: {sorted(unknown)}")

        # Naming a solver without shaping its call, or the reverse, leaves a config that
        # dispatches to nothing or builds a call nobody makes.
        overrides = {
            name
            for name in ("solver_function", "build_solver_kwargs")
            if getattr(cls, name) is not getattr(OptimizerConfig, name)
        }
        if len(overrides) == 1:
            missing = {"solver_function", "build_solver_kwargs"} - overrides
            raise TypeError(
                f"{cls.__name__} overrides {overrides.pop()} without {missing.pop()}, "
                "and they only work as a pair"
            )

        resolvable = set(cls._tol_options) | set(cls._nullable_options)
        unresolved = {name for name in declared if getattr(cls, name, None) is UNSET} - resolvable
        if unresolved:
            raise TypeError(
                f"{cls.__name__} defaults {sorted(unresolved)} to UNSET without listing them "
                f"in _tol_options or _nullable_options, so the sentinel would reach scipy"
            )

    def __post_init__(self) -> None:
        self._resolve_tolerances()

        for name, scipy_default in self._nullable_options.items():
            if getattr(self, name) is UNSET:
                object.__setattr__(self, name, scipy_default)

    def _resolve_tolerances(self) -> None:
        """Fill each tolerance the caller left unset, from ``tol`` or from scipy's default.

        Overridden where a method's `tol` does not mean what ``minimize``'s means.
        """
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

    def solver_function(self) -> Callable[..., Any] | None:
        """The entry point that runs this configuration.

        Defaults to None, meaning :func:`minimize` runs it rather than handing it on. A
        configuration that overrides this owns its own call shape in
        :meth:`build_solver_kwargs`, so `minimize` never branches on which one it has.
        """
        return None

    def build_solver_kwargs(self, problem: SolverProblem) -> dict[str, Any]:
        """Keyword arguments for :meth:`solver_function`, shaped from `problem`.

        Raises
        ------
        TypeError
            If `problem` carries something this solver cannot honor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} names no solver function, so minimize runs it directly "
            "and there is no call to build"
        )

    def default_budget(self, n: int) -> int:
        """The budget `better_optimize` applies to an `n`-dimensional problem by default."""
        return 200 * n

    def optimizer_kwargs(self, n: int | None = None) -> dict[str, Any]:
        """The options dictionary to hand to scipy.

        Parameters
        ----------
        n : int, optional
            Problem dimension. When given, each budget option the caller left unset is
            filled with :meth:`default_budget`, which is what makes that default
            authoritative rather than advisory. Options named in
            ``_scipy_resolved_budgets`` are left out even so. Defaults to None, which emits
            only the budgets the caller set and leaves scipy to apply its own.
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
                if options[name] is None and name not in self._scipy_resolved_budgets:
                    options[name] = self.default_budget(n)

        # An option the caller never set is omitted rather than sent as None. scipy's own
        # default for each of these is None too, so the run is unchanged, and omitting means
        # a config never names an option the installed scipy has not heard of.
        return {
            name: value
            for name, value in options.items()
            if value is not None or name in self._nullable_options
        }

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


@dataclass(frozen=True, eq=False)
class MinimizeConfig(OptimizerConfig, ABC):
    """One scipy ``minimize`` method, and what it needs from the objective.

    The three capability flags are what :func:`validate_provided_functions_minimize`
    reconciles a caller's `jac`, `hess` and `hessp` against.
    """

    uses_grad: ClassVar[bool]
    uses_hess: ClassVar[bool]
    uses_hessp: ClassVar[bool]

    @classmethod
    def _check_declarations(cls) -> None:
        """Also check that the subclass declares its derivative appetite.

        Raises
        ------
        TypeError
            If a capability flag is unset.
        """
        super()._check_declarations()

        for flag in ("uses_grad", "uses_hess", "uses_hessp"):
            if not isinstance(getattr(cls, flag, None), bool):
                raise TypeError(f"{cls.__name__} must set {flag}")


@dataclass(frozen=True, eq=False)
class RootConfig(OptimizerConfig, ABC):
    """One scipy ``root`` method, and whether it consumes a jacobian.

    `uses_jac` is what :func:`validate_provided_functions_root` reconciles a caller's `jac`
    against. There is one flag rather than the three a minimize method declares, because no
    root finder takes a Hessian.

    Unlike the minimize family, the root methods do not share a budget rule. Each subclass
    states its own, because scipy's differs by method and, for `hybr` and `lm`, by whether a
    jacobian was supplied.
    """

    uses_jac: ClassVar[bool]

    @classmethod
    def _check_declarations(cls) -> None:
        """Also check that the subclass declares whether it consumes a jacobian.

        Raises
        ------
        TypeError
            If `uses_jac` is unset.
        """
        super()._check_declarations()

        if not isinstance(getattr(cls, "uses_jac", None), bool):
            raise TypeError(f"{cls.__name__} must set uses_jac")

        if cls.default_budget is OptimizerConfig.default_budget:
            raise TypeError(
                f"{cls.__name__} must state its own default_budget, because the root "
                "methods share none and the inherited one is a minimize number"
            )
