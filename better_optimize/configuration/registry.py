from typing import Any, Literal, overload

from better_optimize.configuration.base import MinimizeConfig
from better_optimize.configuration.first_order import (
    BFGSConfig,
    CGConfig,
    LBFGSBConfig,
    TNCConfig,
)
from better_optimize.configuration.gradient_free import NelderMeadConfig, PowellConfig
from better_optimize.configuration.second_order import (
    DoglegConfig,
    NewtonCGConfig,
    TrustExactConfig,
    TrustKrylovConfig,
    TrustNCGConfig,
)
from better_optimize.configuration.supports_constraints import (
    COBYLAConfig,
    SLSQPConfig,
    TrustConstrConfig,
)
from better_optimize.constants import minimize_method

__all__ = ["MINIMIZE_CONFIGS", "SOLVER_ARGUMENTS", "config_for_method", "config_from_kwargs"]

# Keyed by the method names better_optimize already advertises, so a caller may keep
# passing a string.
MINIMIZE_CONFIGS: dict[str, type[MinimizeConfig]] = {
    "nelder-mead": NelderMeadConfig,
    "powell": PowellConfig,
    "CG": CGConfig,
    "BFGS": BFGSConfig,
    "Newton-CG": NewtonCGConfig,
    "L-BFGS-B": LBFGSBConfig,
    "TNC": TNCConfig,
    "COBYLA": COBYLAConfig,
    "SLSQP": SLSQPConfig,
    "trust-constr": TrustConstrConfig,
    "dogleg": DoglegConfig,
    "trust-ncg": TrustNCGConfig,
    "trust-exact": TrustExactConfig,
    "trust-krylov": TrustKrylovConfig,
}

# scipy lowercases the method name before dispatching, so a caller who writes "bfgs" gets
# BFGS there and should get it here.
_CONFIGS_BY_LOWER_NAME = {name.lower(): config for name, config in MINIMIZE_CONFIGS.items()}


@overload
def config_for_method(method: Literal["nelder-mead"], **options: Any) -> NelderMeadConfig: ...


@overload
def config_for_method(method: Literal["powell"], **options: Any) -> PowellConfig: ...


@overload
def config_for_method(method: Literal["CG"], **options: Any) -> CGConfig: ...


@overload
def config_for_method(method: Literal["BFGS"], **options: Any) -> BFGSConfig: ...


@overload
def config_for_method(method: Literal["Newton-CG"], **options: Any) -> NewtonCGConfig: ...


@overload
def config_for_method(method: Literal["L-BFGS-B"], **options: Any) -> LBFGSBConfig: ...


@overload
def config_for_method(method: Literal["TNC"], **options: Any) -> TNCConfig: ...


@overload
def config_for_method(method: Literal["COBYLA"], **options: Any) -> COBYLAConfig: ...


@overload
def config_for_method(method: Literal["SLSQP"], **options: Any) -> SLSQPConfig: ...


@overload
def config_for_method(method: Literal["trust-constr"], **options: Any) -> TrustConstrConfig: ...


@overload
def config_for_method(method: Literal["dogleg"], **options: Any) -> DoglegConfig: ...


@overload
def config_for_method(method: Literal["trust-ncg"], **options: Any) -> TrustNCGConfig: ...


@overload
def config_for_method(method: Literal["trust-exact"], **options: Any) -> TrustExactConfig: ...


@overload
def config_for_method(method: Literal["trust-krylov"], **options: Any) -> TrustKrylovConfig: ...


@overload
def config_for_method(method: str, **options: Any) -> MinimizeConfig: ...


def config_for_method(method: str, **options: Any) -> MinimizeConfig:
    """Build the configuration for a method named by string.

    Raises
    ------
    ValueError
        If `method` is not a method `better_optimize` supports.
    TypeError
        If an option is not one the method accepts, raised by the dataclass itself.
    """
    return _config_class(method)(**options)


def _config_class(method: str) -> type[MinimizeConfig]:
    config_class = _CONFIGS_BY_LOWER_NAME.get(method.lower())
    if config_class is None:
        known = ", ".join(sorted(MINIMIZE_CONFIGS))
        raise ValueError(f"Unknown method {method!r}. Must be one of: {known}")

    return config_class


SOLVER_ARGUMENTS = frozenset({"bounds", "constraints"})
"""Keyword arguments scipy takes beside the options dictionary, describing the problem
rather than the method, so they never belong to a config."""


def config_from_kwargs(
    method: minimize_method | MinimizeConfig, kwargs: dict[str, Any]
) -> tuple[MinimizeConfig, dict[str, Any]]:
    """Resolve what the flat API was given into a config and scipy's remaining arguments.

    Parameters
    ----------
    method : str or MinimizeConfig
        A method name, or a configuration to use as given.
    kwargs : dict
        Everything the caller passed beside the problem and the reporting settings.

    Returns
    -------
    config : MinimizeConfig
        The configuration for the method.
    solver_kwargs : dict
        The arguments scipy takes beside its options dictionary.

    Raises
    ------
    TypeError
        If `method` is a configuration and an option was also passed, since the two
        would answer the same question and neither obviously wins.
    ValueError
        If `method` names a method with no configuration.
    """
    kwargs = dict(kwargs)
    solver_kwargs = {name: kwargs.pop(name) for name in SOLVER_ARGUMENTS & kwargs.keys()}

    if isinstance(method, MinimizeConfig):
        if kwargs:
            raise TypeError(
                f"Got both a {type(method).__name__} and the option(s) "
                f"{sorted(kwargs)}. Set them on the configuration instead."
            )
        return method, solver_kwargs

    config_class = _config_class(method)
    # A name given both ways takes its top-level value, as promotion did.
    kwargs = (kwargs.pop("options", None) or {}) | kwargs

    # A top-level maxiter fills whichever names this method caps its work with, the way
    # tol fills its tolerances. TNC has no maxiter of its own and spells it maxfun.
    budget = kwargs.pop("maxiter", None)
    if budget is not None:
        for name in config_class._budget_options():
            kwargs.setdefault(name, budget)

    return config_for_method(method, **kwargs), solver_kwargs
