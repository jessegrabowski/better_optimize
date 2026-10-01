from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import pytest

from better_optimize.configuration.base import UNSET, MinimizeConfig


@dataclass
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


def test_an_unknown_option_is_a_type_error():
    with pytest.raises(TypeError, match="gtoll"):
        StubConfig(gtoll=1e-5)
