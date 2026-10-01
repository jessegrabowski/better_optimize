import ast
import inspect
import textwrap

from collections.abc import Callable
from typing import Any

from scipy.optimize import (
    _cobyla_py,
    _lbfgsb_py,
    _minimize,
    _optimize,
    _slsqp_py,
    _tnc,
    _trustregion_dogleg,
    _trustregion_exact,
    _trustregion_krylov,
    _trustregion_ncg,
)
from scipy.optimize._trustregion import _minimize_trust_region
from scipy.optimize._trustregion_constr import minimize_trustregion_constr

# Supplied positionally or by keyword by ``scipy.optimize.minimize`` itself, so they are
# never legal members of the ``options`` dict. ``grad`` is trust-constr's name for ``jac``.
SUPPLIED_BY_MINIMIZE = frozenset(
    {
        "fun",
        "func",
        "x0",
        "args",
        "jac",
        "grad",
        "hess",
        "hessp",
        "callback",
        "bounds",
        "constraints",
        "subproblem",
    }
)


def option_signature(function: Callable[..., Any]) -> dict[str, Any]:
    """The options a scipy minimizer accepts, mapped to their source defaults."""
    return {
        name: None if _is_deprecation_sentinel(parameter.default) else parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is not parameter.VAR_KEYWORD and name not in SUPPLIED_BY_MINIMIZE
    }


def _is_deprecation_sentinel(default: Any) -> bool:
    """scipy marks a deprecated option's default with a bare ``object()`` meaning "unset"."""
    return type(default) is object


def trust_region_signature(wrapper: Callable[..., Any]) -> dict[str, Any]:
    """A trust-region method's options: the shared driver's, overridden by the wrapper's.

    The four wrappers name almost nothing themselves and forward the rest to
    ``_minimize_trust_region``, so the reachable option set is the union of the two.
    """
    return option_signature(_minimize_trust_region) | option_signature(wrapper)


SCIPY_OPTIONS: dict[str, dict[str, Any]] = {
    "nelder-mead": option_signature(_optimize._minimize_neldermead),
    "powell": option_signature(_optimize._minimize_powell),
    "CG": option_signature(_optimize._minimize_cg),
    "BFGS": option_signature(_optimize._minimize_bfgs),
    "Newton-CG": option_signature(_optimize._minimize_newtoncg),
    "L-BFGS-B": option_signature(_lbfgsb_py._minimize_lbfgsb),
    "TNC": option_signature(_tnc._minimize_tnc),
    "COBYLA": option_signature(_cobyla_py._minimize_cobyla),
    "SLSQP": option_signature(_slsqp_py._minimize_slsqp),
    "trust-constr": option_signature(minimize_trustregion_constr._minimize_trustregion_constr),
    "dogleg": trust_region_signature(_trustregion_dogleg._minimize_dogleg),
    "trust-ncg": trust_region_signature(_trustregion_ncg._minimize_trust_ncg),
    "trust-exact": trust_region_signature(_trustregion_exact._minimize_trustregion_exact),
    "trust-krylov": trust_region_signature(_trustregion_krylov._minimize_trust_krylov),
}


def _is_setdefault(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "setdefault"
    )


def tol_targets(method: str) -> tuple[str, ...]:
    """The options ``minimize(tol=...)`` fills for `method`.

    Parsed from the ``if tol is not None:`` block of ``scipy.optimize.minimize`` rather than
    transcribed, so that a change to scipy's mapping shows up as a failing test.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(_minimize.minimize)))
    (block,) = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "tol is not None"
    )

    targets: list[str] = []
    for branch in block.body:
        if not isinstance(branch, ast.If):
            continue

        compared = ast.unparse(branch.test)
        if f"'{method.lower()}'" not in compared:
            continue

        targets.extend(
            statement.value.args[0].value for statement in branch.body if _is_setdefault(statement)
        )

    return tuple(targets)
