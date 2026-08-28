from typing import Any

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
    TrustRegionConfig,
)
from better_optimize.configuration.supports_constraints import (
    COBYLAConfig,
    SLSQPConfig,
    TrustConstrConfig,
)

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


def config_for_method(method: str, **options: Any) -> MinimizeConfig:
    """Build the configuration for a method named by string.

    Raises
    ------
    ValueError
        If `method` is not a method `better_optimize` supports.
    TypeError
        If an option is not one the method accepts, raised by the dataclass itself.
    """
    if method not in MINIMIZE_CONFIGS:
        known = ", ".join(sorted(MINIMIZE_CONFIGS))
        raise ValueError(f"Unknown method {method!r}. Must be one of: {known}")

    return MINIMIZE_CONFIGS[method](**options)


__all__ = [
    "MINIMIZE_CONFIGS",
    "BFGSConfig",
    "COBYLAConfig",
    "CGConfig",
    "DoglegConfig",
    "LBFGSBConfig",
    "MinimizeConfig",
    "NelderMeadConfig",
    "NewtonCGConfig",
    "PowellConfig",
    "SLSQPConfig",
    "TNCConfig",
    "TrustConstrConfig",
    "TrustExactConfig",
    "TrustKrylovConfig",
    "TrustNCGConfig",
    "TrustRegionConfig",
    "config_for_method",
]
