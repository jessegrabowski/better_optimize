from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import numpy as np

from scipy.optimize import minimize as sp_minimize

__all__ = ["FiniteDiffStep", "MinimizeConfig", "SQRT_EPS", "Workers"]

# scipy's ``_epsilon``: the forward-difference step most methods default to.
SQRT_EPS = 1.4901161193847656e-08

Workers = int | Callable[..., Any] | None
FiniteDiffStep = float | np.ndarray | None


@dataclass
class MinimizeConfig(ABC):
    """One scipy ``minimize`` method, its options, and what it needs from the objective.

    Each subclass declares one field per option the method actually accepts, typed and
    defaulted to the value in scipy's own source. The fields are therefore the complete
    and authoritative option list: passing anything else raises ``TypeError`` rather than
    the ``OptimizeWarning`` scipy would emit and drop.

    Methods disagree about what their work budget is called, so there is no ``maxiter``
    field here; each subclass declares the names scipy accepts and lists them in
    ``_budget_options``.

    Parameters
    ----------
    tol : float, optional
        Convenience tolerance filling every option named in ``_tol_options`` that was left
        at its default, mirroring what ``scipy.optimize.minimize`` does with its own
        top-level ``tol``. Defaults to None, leaving each tolerance at its own default.
    """

    tol: float | None = None

    uses_grad: ClassVar[bool]
    uses_hess: ClassVar[bool]
    uses_hessp: ClassVar[bool]

    # Taken from scipy's own tol mapping in ``_minimize.py``, not from its documentation.
    _excluded: ClassVar[frozenset[str]] = frozenset({"tol"})
    """Fields that are not options of the method."""

    _tol_options: ClassVar[tuple[str, ...]] = ()
    _budget_options: ClassVar[tuple[str, ...]] = ("maxiter",)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """The string scipy knows this method by."""

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        """The options dictionary to hand to scipy, with ``tol`` already distributed."""
        options = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in self._excluded
        }

        return self._finalize(options)

    def solver_function(self) -> Callable[..., Any]:
        """The callable that runs this configuration."""
        return sp_minimize

    def default_maxiter(self, n: int) -> int:
        """The budget scipy chooses for an `n`-dimensional problem when none is given."""
        return 200 * n

    def resolved_maxiter(self, n: int) -> int:
        """The effective budget: the largest the caller set, else :meth:`default_maxiter`."""
        options = self.optimizer_kwargs
        explicit_budgets = [
            options[name] for name in self._budget_options if options.get(name) is not None
        ]

        return max(explicit_budgets) if explicit_budgets else self.default_maxiter(n)

    def _finalize(self, options: dict[str, Any]) -> dict[str, Any]:
        """Distribute ``tol`` and drop budget options the caller left unset."""
        if self.tol is not None:
            defaults = {field.name: field.default for field in fields(self)}
            for name in self._tol_options:
                if options[name] == defaults[name]:
                    options[name] = self.tol

        # An unset budget or tolerance is omitted rather than sent as a None, both because
        # several methods compare against it directly and would raise, and because omitting
        # it is what lets scipy apply the default that "unset" means.
        for name in (*self._budget_options, *self._tol_options):
            if name in options and options[name] is None:
                del options[name]

        return options
