from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import numpy as np
import pytest

from better_optimize.configuration.base import (
    UNSET,
    MinimizeConfig,
    OptimizerConfig,
    RootConfig,
    SolverProblem,
)


@dataclass(frozen=True, eq=False)
class StubConfig(MinimizeConfig):
    """Minimal concrete subclass, so the base's own behavior can be tested directly."""

    gtol: float = UNSET
    ftol: float = UNSET
    maxiter: int | None = None
    maxfun: int | None = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[Mapping[str, float]] = {"gtol": 1e-5, "ftol": 1e-3}
    _iteration_options: ClassVar[tuple[str, ...]] = ("maxiter",)
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfun",)

    @property
    def method_name(self) -> str:
        return "stub"


def test_the_base_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        MinimizeConfig()


def test_an_unset_tolerance_resolves_to_the_scipy_default():
    assert StubConfig().gtol == 1e-5
    assert StubConfig().optimizer_kwargs()["ftol"] == 1e-3


def test_tol_fills_every_tolerance_the_caller_did_not_pass():
    options = StubConfig(tol=1e-9).optimizer_kwargs()

    assert options["gtol"] == 1e-9
    assert options["ftol"] == 1e-9


def test_an_explicit_tolerance_survives_tol_even_at_its_default_value():
    options = StubConfig(tol=1e-9, gtol=1e-5).optimizer_kwargs()

    assert options["gtol"] == 1e-5
    assert options["ftol"] == 1e-9


def test_unset_budgets_are_dropped_rather_than_sent_as_none():
    assert "maxiter" not in StubConfig().optimizer_kwargs()
    assert StubConfig(maxiter=7).optimizer_kwargs()["maxiter"] == 7


def test_a_dimension_fills_every_unset_budget():
    options = StubConfig().optimizer_kwargs(n=10)

    assert options["maxiter"] == 2000
    assert options["maxfun"] == 2000


def test_a_dimension_does_not_overwrite_a_budget_the_caller_set():
    options = StubConfig(maxiter=7).optimizer_kwargs(n=10)

    assert options["maxiter"] == 7
    assert options["maxfun"] == 2000


def test_the_evaluation_budget_prefers_an_evaluation_capping_option():
    assert StubConfig(maxiter=900, maxfun=5).evaluation_budget(10) == 5
    assert StubConfig(maxiter=5).evaluation_budget(10) == 5
    assert StubConfig().evaluation_budget(10) == 2000


def test_a_config_cannot_be_written_to_after_construction():
    """One config is shared across every start in a multi-start run."""
    with pytest.raises(FrozenInstanceError):
        StubConfig().gtol = 1.0


def test_an_unknown_option_is_a_type_error():
    with pytest.raises(TypeError, match="gtoll"):
        StubConfig(gtoll=1e-5)


def test_a_subclass_must_set_every_capability_flag():
    with pytest.raises(TypeError, match="must set uses_hessp"):

        @dataclass(frozen=True, eq=False)
        class MissingFlag(MinimizeConfig):
            uses_grad: ClassVar[bool] = True
            uses_hess: ClassVar[bool] = False
            _iteration_options: ClassVar[tuple[str, ...]] = ()

            @property
            def method_name(self) -> str:
                return "missing-flag"


def test_a_config_that_is_not_a_minimize_method_needs_no_capability_flags():
    """A global optimizer has no derivative appetite of its own, so it subclasses the base
    that does not ask for one."""

    @dataclass(frozen=True, eq=False)
    class Flagless(OptimizerConfig):
        _iteration_options: ClassVar[tuple[str, ...]] = ()

        @property
        def method_name(self) -> str:
            return "flagless"

    assert Flagless().optimizer_kwargs() == {}


def test_an_option_group_may_only_name_declared_fields():
    with pytest.raises(TypeError, match=r"_tol_options names non-fields: \['nope'\]"):

        @dataclass(frozen=True, eq=False)
        class StrayTolerance(MinimizeConfig):
            uses_grad: ClassVar[bool] = True
            uses_hess: ClassVar[bool] = False
            uses_hessp: ClassVar[bool] = False
            _iteration_options: ClassVar[tuple[str, ...]] = ()
            _tol_options: ClassVar[Mapping[str, float]] = {"nope": 1.0}

            @property
            def method_name(self) -> str:
                return "stray-tolerance"


def test_a_sentinel_default_must_be_listed_as_a_tolerance():
    """Otherwise ``__post_init__`` never resolves it and the sentinel reaches scipy."""
    with pytest.raises(TypeError, match=r"defaults \['gtol'\] to UNSET"):

        @dataclass(frozen=True, eq=False)
        class LeakedSentinel(MinimizeConfig):
            gtol: float = UNSET

            uses_grad: ClassVar[bool] = True
            uses_hess: ClassVar[bool] = False
            uses_hessp: ClassVar[bool] = False
            _iteration_options: ClassVar[tuple[str, ...]] = ()

            @property
            def method_name(self) -> str:
                return "leaked-sentinel"


def test_a_config_minimize_runs_itself_has_no_solver_to_name():
    """`solver_function` returning None is what tells `minimize` to run the config rather
    than hand it on, and such a config never builds a call."""
    assert StubConfig().solver_function() is None

    with pytest.raises(NotImplementedError, match="names no solver function"):
        StubConfig().build_solver_kwargs(problem=None)


def test_naming_a_solver_without_shaping_its_call_is_rejected():
    """The pair dispatches to a function with no arguments to give it."""
    with pytest.raises(TypeError, match="overrides solver_function without build_solver_kwargs"):

        @dataclass(frozen=True, eq=False)
        class NamesOnly(OptimizerConfig):
            _iteration_options: ClassVar[tuple[str, ...]] = ()

            def solver_function(self):
                return print

            @property
            def method_name(self) -> str:
                return "names-only"


def test_shaping_a_call_without_naming_a_solver_is_rejected():
    """The pair builds arguments nobody will ever pass."""
    with pytest.raises(TypeError, match="overrides build_solver_kwargs without solver_function"):

        @dataclass(frozen=True, eq=False)
        class ShapesOnly(OptimizerConfig):
            _iteration_options: ClassVar[tuple[str, ...]] = ()

            def build_solver_kwargs(self, problem):
                return {}

            @property
            def method_name(self) -> str:
                return "shapes-only"


def test_a_solver_problem_compares_by_identity():
    """It carries `x0`, so value equality would raise on the ambiguous truth of an array,
    the same reason a configuration has none."""
    arguments = dict(
        f=print,
        x0=np.zeros(3),
        jac=None,
        hess=None,
        hessp=None,
        args=(),
        callback=None,
        progressbar=False,
        progress_task=None,
        progressbar_update_interval=1,
        verbose=False,
        solver_kwargs={},
    )
    problem = SolverProblem(**arguments)

    assert problem == problem
    assert problem != SolverProblem(**arguments)
    assert isinstance(hash(problem), int)


@dataclass(frozen=True, eq=False)
class StubRootConfig(RootConfig):
    """Minimal concrete root subclass, so the base's own behavior can be tested directly."""

    xtol: float = UNSET
    maxfev: int | None = None

    uses_jac: ClassVar[bool] = True

    _tol_options: ClassVar[Mapping[str, float]] = {"xtol": 1.49012e-08}
    _iteration_options: ClassVar[tuple[str, ...]] = ()
    _evaluation_options: ClassVar[tuple[str, ...]] = ("maxfev",)

    @property
    def method_name(self) -> str:
        return "stub-root"

    def default_budget(self, n: int) -> int:
        return 200 * (n + 1)


def test_a_root_subclass_must_say_whether_it_consumes_a_jacobian():
    with pytest.raises(TypeError, match="must set uses_jac"):

        @dataclass(frozen=True, eq=False)
        class NoFlag(RootConfig):
            _iteration_options: ClassVar[tuple[str, ...]] = ()

            @property
            def method_name(self) -> str:
                return "no-flag"


def test_a_root_config_is_not_a_minimize_config():
    """A root finder consumes at most a jacobian, so the two share the option machinery and
    nothing else. Which base a configuration has is what the drivers route on."""
    assert issubclass(StubRootConfig, OptimizerConfig)
    assert not issubclass(StubRootConfig, MinimizeConfig)
    assert StubRootConfig().uses_jac


def test_a_root_config_gets_the_shared_option_machinery():
    config = StubRootConfig()

    assert config.optimizer_kwargs() == {"xtol": 1.49012e-08}
    assert config.optimizer_kwargs(n=4)["maxfev"] == 1000
    assert StubRootConfig(tol=1e-12).optimizer_kwargs()["xtol"] == 1e-12
