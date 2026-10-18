import warnings

from dataclasses import fields
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from better_optimize.configuration import OptimizerConfig

minimize_method = Literal[
    "nelder-mead",
    "powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "COBYQA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]

root_method = Literal[
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
    "krylov",
    "df-sane",
]

CONSOLE_WIDTH = 100


# SciPy's direct solvers ignore the callback argument; the iterative root finders honor it.
ROOT_METHODS_WITHOUT_CALLBACK = frozenset({"hybr", "lm"})


def _option_names(config_class: "type[OptimizerConfig]") -> list[str]:
    return sorted(config_class._declared_fields() - config_class._excluded)


def _minimize_mode_kwargs() -> dict[str, dict[str, Any]]:
    # Imported here because the configuration registry itself imports this module.
    from better_optimize.configuration import MINIMIZE_CONFIGS

    return {
        method: {
            "uses_grad": config_class.uses_grad,
            "uses_hess": config_class.uses_hess,
            "uses_hessp": config_class.uses_hessp,
            "f_maxiter_default": config_class().default_budget,
            "valid_options": _option_names(config_class),
        }
        for method, config_class in MINIMIZE_CONFIGS.items()
    }


def _root_mode_kwargs() -> dict[str, dict[str, Any]]:
    from better_optimize.configuration import ROOT_CONFIGS, QuasiNewtonRootConfig

    mode_kwargs: dict[str, dict[str, Any]] = {}
    for method, config_class in ROOT_CONFIGS.items():
        mode_kwargs[method] = {
            "uses_jac": config_class.uses_jac,
            "f_maxiter_default": config_class().default_budget,
            "valid_options": _option_names(config_class),
        }

        if issubclass(config_class, QuasiNewtonRootConfig):
            jac_options = fields(config_class._jac_options_type)
            mode_kwargs[method]["jac_options"] = [field.name for field in jac_options]

    return mode_kwargs


_DEPRECATED = {
    "MINIMIZE_MODE_KWARGS": (
        _minimize_mode_kwargs,
        "better_optimize.configuration.MINIMIZE_CONFIGS[method].uses_grad",
    ),
    "ROOT_MODE_KWARGS": (
        _root_mode_kwargs,
        "better_optimize.configuration.ROOT_CONFIGS[method].uses_jac",
    ),
    "TOLERANCES": (
        lambda: ["xtol", "ftol", "gtol", "fatol", "xatol"],
        "the tolerance fields of the method's configuration class",
    ),
    "DE_STRATEGY_OPTIONS": (
        lambda: (
            "best1bin",
            "best1exp",
            "rand1bin",
            "rand1exp",
            "randtobest1bin",
            "randtobest1exp",
            "currenttobest1bin",
            "currenttobest1exp",
            "best2bin",
            "best2exp",
            "rand2bin",
            "rand2exp",
        ),
        "the strategy names documented by scipy.optimize.differential_evolution",
    ),
    "DE_INIT_OPTIONS": (
        lambda: ("sobol", "halton", "latinhypercube", "random"),
        "the init names documented by scipy.optimize.differential_evolution",
    ),
}
"""Names removed from this module in 0.4.3, each mapped to the function building its value and
the replacement its deprecation warning names."""


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    build, replacement = _DEPRECATED[name]
    warnings.warn(
        f"better_optimize.constants.{name} is deprecated and will be removed in a future "
        f"release. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build()
