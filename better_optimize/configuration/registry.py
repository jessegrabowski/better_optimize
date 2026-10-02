from typing import Any, Literal, overload

from better_optimize.configuration.base import MinimizeConfig, OptimizerConfig, RootConfig
from better_optimize.configuration.first_order import (
    BFGSConfig,
    CGConfig,
    LBFGSBConfig,
    TNCConfig,
)
from better_optimize.configuration.gradient_free import NelderMeadConfig, PowellConfig
from better_optimize.configuration.root_direct import DFSaneConfig, HybrConfig, LMConfig
from better_optimize.configuration.root_quasi_newton import (
    AndersonConfig,
    Broyden1Config,
    Broyden2Config,
    DiagBroydenConfig,
    ExcitingMixingConfig,
    KrylovConfig,
    LinearMixingConfig,
)
from better_optimize.configuration.second_order import (
    DoglegConfig,
    NewtonCGConfig,
    TrustExactConfig,
    TrustKrylovConfig,
    TrustNCGConfig,
)
from better_optimize.configuration.supports_constraints import (
    COBYLAConfig,
    COBYQAConfig,
    SLSQPConfig,
    TrustConstrConfig,
)
from better_optimize.constants import minimize_method, root_method

__all__ = [
    "MINIMIZE_CONFIGS",
    "MINIMIZE_CONFIGS_BY_LOWER_NAME",
    "ROOT_CONFIGS",
    "ROOT_CONFIGS_BY_LOWER_NAME",
    "SOLVER_ARGUMENTS",
    "config_for_method",
    "config_for_root_method",
    "config_from_kwargs",
    "root_config_from_kwargs",
]

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
    "COBYQA": COBYQAConfig,
    "SLSQP": SLSQPConfig,
    "trust-constr": TrustConstrConfig,
    "dogleg": DoglegConfig,
    "trust-ncg": TrustNCGConfig,
    "trust-exact": TrustExactConfig,
    "trust-krylov": TrustKrylovConfig,
}

MINIMIZE_CONFIGS_BY_LOWER_NAME: dict[str, type[MinimizeConfig]] = {
    name.lower(): config for name, config in MINIMIZE_CONFIGS.items()
}
"""The same configurations, keyed for a case-insensitive lookup. scipy lowercases the
method name before dispatching, so anything resolving a name here should match it."""


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
def config_for_method(method: Literal["COBYQA"], **options: Any) -> COBYQAConfig: ...


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
    config_class = MINIMIZE_CONFIGS_BY_LOWER_NAME.get(method.lower())
    if config_class is None:
        known = ", ".join(sorted(MINIMIZE_CONFIGS))
        raise ValueError(f"Unknown method {method!r}. Must be one of: {known}")

    return config_class


SOLVER_ARGUMENTS = frozenset({"bounds", "constraints"})
"""Keyword arguments scipy takes beside the options dictionary, describing the problem
rather than the method, so they never belong to a config."""


def config_from_kwargs(
    method: minimize_method | OptimizerConfig, kwargs: dict[str, Any]
) -> tuple[OptimizerConfig, dict[str, Any]]:
    """Resolve what the flat API was given into a config and scipy's remaining arguments.

    Parameters
    ----------
    method : str or OptimizerConfig
        A method name, or a configuration to use as given.
    kwargs : dict
        Everything the caller passed beside the problem and the reporting settings.

    Returns
    -------
    config : OptimizerConfig
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

    if isinstance(method, OptimizerConfig):
        if kwargs:
            raise TypeError(
                f"Got both a {type(method).__name__} and the option(s) "
                f"{sorted(kwargs)}. Set them on the configuration instead."
            )
        return method, solver_kwargs

    config_class = _config_class(method)
    # A name given both ways takes its top-level value, as promotion did.
    kwargs = (kwargs.pop("options", None) or {}) | kwargs

    return config_for_method(method, **_spread_budget(config_class, kwargs)), solver_kwargs


def _spread_budget(config_class: type[OptimizerConfig], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill whichever names a method caps its work with from a top-level `maxiter`.

    Methods spell that cap `maxiter`, `maxfev` or `maxfun`, and a caller is not told which
    theirs uses, so this does for the budget what ``tol`` does for the tolerances.
    """
    spread = dict(kwargs)
    budget = spread.pop("maxiter", None)
    if budget is not None:
        for name in config_class._budget_options():
            spread.setdefault(name, budget)

    return spread


ROOT_CONFIGS: dict[str, type[RootConfig]] = {
    "hybr": HybrConfig,
    "lm": LMConfig,
    "broyden1": Broyden1Config,
    "broyden2": Broyden2Config,
    "anderson": AndersonConfig,
    "linearmixing": LinearMixingConfig,
    "diagbroyden": DiagBroydenConfig,
    "excitingmixing": ExcitingMixingConfig,
    "krylov": KrylovConfig,
    "df-sane": DFSaneConfig,
}
"""Keyed by the method names `root` advertises, so a caller may keep passing a string."""

ROOT_CONFIGS_BY_LOWER_NAME: dict[str, type[RootConfig]] = {
    name.lower(): config for name, config in ROOT_CONFIGS.items()
}
"""The same configurations, keyed for a case-insensitive lookup, as `root` resolves one."""


@overload
def config_for_root_method(method: Literal["hybr"], **options: Any) -> HybrConfig: ...


@overload
def config_for_root_method(method: Literal["lm"], **options: Any) -> LMConfig: ...


@overload
def config_for_root_method(method: Literal["broyden1"], **options: Any) -> Broyden1Config: ...


@overload
def config_for_root_method(method: Literal["broyden2"], **options: Any) -> Broyden2Config: ...


@overload
def config_for_root_method(method: Literal["anderson"], **options: Any) -> AndersonConfig: ...


@overload
def config_for_root_method(
    method: Literal["linearmixing"], **options: Any
) -> LinearMixingConfig: ...


@overload
def config_for_root_method(method: Literal["diagbroyden"], **options: Any) -> DiagBroydenConfig: ...


@overload
def config_for_root_method(
    method: Literal["excitingmixing"], **options: Any
) -> ExcitingMixingConfig: ...


@overload
def config_for_root_method(method: Literal["krylov"], **options: Any) -> KrylovConfig: ...


@overload
def config_for_root_method(method: Literal["df-sane"], **options: Any) -> DFSaneConfig: ...


@overload
def config_for_root_method(method: str, **options: Any) -> RootConfig: ...


def config_for_root_method(method: str, **options: Any) -> RootConfig:
    """The configuration for one scipy ``root`` method, built from `options`.

    Raises
    ------
    ValueError
        If `method` is not a method `better_optimize` supports.
    TypeError
        If an option is not one the method accepts, raised by the dataclass itself.
    """
    return _root_config_class(method)(**options)


def _root_config_class(method: str) -> type[RootConfig]:
    config_class = ROOT_CONFIGS_BY_LOWER_NAME.get(method.lower())
    if config_class is None:
        known = ", ".join(sorted(ROOT_CONFIGS))
        raise ValueError(f"Unknown method {method!r}. Must be one of: {known}")

    return config_class


def root_config_from_kwargs(method: root_method | RootConfig, kwargs: dict[str, Any]) -> RootConfig:
    """Resolve what `root` was given into a configuration.

    Raises
    ------
    TypeError
        If `method` is a configuration and an option was also passed, since the two would
        answer the same question and neither obviously wins.
    ValueError
        If `method` names a method with no configuration.
    """
    kwargs = dict(kwargs)

    if isinstance(method, RootConfig):
        if kwargs:
            raise TypeError(
                f"Got both a {type(method).__name__} and the option(s) "
                f"{sorted(kwargs)}. Set them on the configuration instead."
            )
        return method

    # A name given both ways takes its top-level value, as promotion did.
    kwargs = (kwargs.pop("options", None) or {}) | kwargs
    kwargs = _spread_budget(_root_config_class(method), kwargs)

    return config_for_root_method(method, **kwargs)
