from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import pytest

from better_optimize.configuration.base import UNSET, MinimizeConfig, OptimizerConfig


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
