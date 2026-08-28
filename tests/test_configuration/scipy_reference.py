import ast
import inspect
import textwrap

from collections.abc import Callable
from typing import Any

from scipy.optimize import _lbfgsb_py, _minimize, _optimize, _tnc

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
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is not parameter.VAR_KEYWORD and name not in SUPPLIED_BY_MINIMIZE
    }


SCIPY_OPTIONS: dict[str, dict[str, Any]] = {
    "CG": option_signature(_optimize._minimize_cg),
    "BFGS": option_signature(_optimize._minimize_bfgs),
    "L-BFGS-B": option_signature(_lbfgsb_py._minimize_lbfgsb),
    "TNC": option_signature(_tnc._minimize_tnc),
}

# Options scipy accepts that we deliberately do not expose, with the reason.
OMITTED_OPTIONS: dict[str, dict[str, str]] = {
    "L-BFGS-B": {
        "disp": "deprecated no-op, removed in scipy 1.18",
        "iprint": "deprecated no-op, removed in scipy 1.18",
    },
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
