from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from better_optimize.configuration.base import MinimizeConfig


@dataclass
class StubConfig(MinimizeConfig):
    """Minimal concrete subclass, so the base's own behavior can be tested directly."""

    gtol: float = 1e-5
    ftol: float = 1e-3
    maxiter: int | None = None
    maxfun: int | None = None

    uses_grad: ClassVar[bool] = True
    uses_hess: ClassVar[bool] = False
    uses_hessp: ClassVar[bool] = False

    _tol_options: ClassVar[tuple[str, ...]] = ("gtol", "ftol")
    _budget_options: ClassVar[tuple[str, ...]] = ("maxiter", "maxfun")

    @property
    def method_name(self) -> str:
        return "stub"

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        return self._finalize(
            {
                "gtol": self.gtol,
                "ftol": self.ftol,
                "maxiter": self.maxiter,
                "maxfun": self.maxfun,
            }
        )


def test_the_base_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        MinimizeConfig()


def test_tol_fills_every_tolerance_option():
    options = StubConfig(tol=1e-9).optimizer_kwargs

    assert options["gtol"] == 1e-9
    assert options["ftol"] == 1e-9


def test_an_explicit_tolerance_survives_tol():
    options = StubConfig(tol=1e-9, gtol=1e-2).optimizer_kwargs

    assert options["gtol"] == 1e-2
    assert options["ftol"] == 1e-9


def test_unset_budgets_are_dropped_rather_than_sent_as_none():
    assert "maxiter" not in StubConfig().optimizer_kwargs
    assert StubConfig(maxiter=7).optimizer_kwargs["maxiter"] == 7


def test_resolved_maxiter_falls_back_to_the_scipy_default():
    assert StubConfig().resolved_maxiter(10) == 2000


def test_resolved_maxiter_takes_the_largest_budget_the_caller_set():
    assert StubConfig(maxiter=50, maxfun=900).resolved_maxiter(10) == 900


def test_an_unknown_option_is_a_type_error():
    with pytest.raises(TypeError, match="gtoll"):
        StubConfig(gtoll=1e-5)
