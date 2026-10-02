import warnings

import numpy as np
import pytest

from scipy.optimize import root
from scipy.optimize._nonlin import KrylovJacobian

from better_optimize.configuration.root_jac_options import (
    KRYLOV_SOLVERS,
    AndersonJacOptions,
    BroydenJacOptions,
    DiagonalJacOptions,
    ExcitingMixingJacOptions,
    KrylovJacOptions,
)

ALL = [
    BroydenJacOptions,
    AndersonJacOptions,
    DiagonalJacOptions,
    ExcitingMixingJacOptions,
    KrylovJacOptions,
]


def residual(x):
    return np.array([x[0] ** 2 - 1.0, x[1] - 2.0])


@pytest.mark.parametrize(
    "options, unset",
    [
        (BroydenJacOptions, "alpha"),
        (AndersonJacOptions, "alpha"),
        (DiagonalJacOptions, "alpha"),
        (ExcitingMixingJacOptions, "alpha"),
        (KrylovJacOptions, "rdiff"),
    ],
)
def test_an_unset_option_is_omitted(options, unset):
    """Named per type, because `alpha` is not a key krylov has and the assertion would
    pass on it for the wrong reason."""
    assert unset not in options().as_dict()


def test_an_option_the_caller_set_is_emitted():
    assert BroydenJacOptions(alpha=0.5, max_rank=3).as_dict()["max_rank"] == 3


@pytest.mark.parametrize(
    "options, method",
    [
        (BroydenJacOptions(max_rank=3), "broyden2"),
        (AndersonJacOptions(M=3), "anderson"),
        (DiagonalJacOptions(alpha=0.1), "diagbroyden"),
        (ExcitingMixingJacOptions(alphamax=0.5), "excitingmixing"),
        (KrylovJacOptions(inner_maxiter=5), "krylov"),
    ],
)
def test_scipy_accepts_what_each_type_emits(options, method):
    """A key the jacobian class does not take raises from its constructor, so this cannot
    pass by being ignored."""
    result = root(
        residual,
        np.array([0.5, 0.5]),
        method=method,
        options={"jac_options": options.as_dict(), "maxiter": 5},
    )

    assert np.all(np.isfinite(result.x))


def test_an_unknown_krylov_solver_is_refused():
    with pytest.raises(ValueError, match="method must be one of"):
        KrylovJacOptions(method="nope")


def test_an_inner_option_the_solver_does_not_take_is_refused():
    """scipy warns and ignores it, which is the failure this package exists to replace."""
    with pytest.raises(ValueError, match=r"inner_options \['restart'\]"):
        KrylovJacOptions(inner_options={"restart": 5})


@pytest.mark.parametrize(
    "inner, field", [("maxiter", "inner_maxiter"), ("M", "inner_M"), ("outer_k", "outer_k")]
)
def test_an_inner_option_that_would_override_a_field_is_refused(inner, field):
    """scipy writes ``inner_<name>`` onto the inner keyword ``<name>``, on top of what the
    field of the same name set, so emitting both discards the field without a word."""
    with pytest.raises(ValueError, match=rf"override the fields \['{field}'\]"):
        KrylovJacOptions(inner_maxiter=20, inner_options={inner: 5})


def test_an_inner_option_naming_no_field_is_still_accepted():
    """`inner_m` is a real lgmres parameter that no field controls, and the prefix alone
    cannot tell it apart from one that collides."""
    assert KrylovJacOptions(inner_options={"inner_m": 4}).as_dict()["inner_inner_m"] == 4


def test_the_accepted_inner_options_follow_the_chosen_solver():
    """`restart` belongs to gmres and not to lgmres, so the check cannot be a fixed list."""
    assert (
        KrylovJacOptions(method="gmres", inner_options={"restart": 5}).as_dict()["inner_restart"]
        == 5
    )


def test_an_inner_option_reaches_scipy_without_a_warning():
    options = KrylovJacOptions(inner_options={"rtol": 1e-9}).as_dict()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        root(
            residual,
            np.array([0.5, 0.5]),
            method="krylov",
            options={"jac_options": options, "maxiter": 5},
        )


@pytest.mark.parametrize("name", sorted(KRYLOV_SOLVERS))
def test_every_named_krylov_solver_is_one_scipy_dispatches_to(name):
    """scipy resolves the name against its own table, so a key spelled differently here
    would be passed through as a callable and fail at solve time."""
    jacobian = KrylovJacobian(**KrylovJacOptions(method=name).as_dict())

    assert jacobian.method is KRYLOV_SOLVERS[name]


def test_an_array_option_is_not_shared_between_runs():
    preconditioner = np.eye(2)
    options = KrylovJacOptions(inner_M=preconditioner)

    options.as_dict()["inner_M"][0, 0] = 99.0

    assert options.inner_M[0, 0] == 1.0
