from functools import partial
from typing import get_args

import numpy as np
import pytest

from numpy.testing import assert_allclose
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult
from scipy.sparse.linalg import LinearOperator

from better_optimize.constants import minimize_method
from better_optimize.minimize import minimize
from better_optimize.utilities import ToggleableProgress

all_methods = list(get_args(minimize_method))
no_grad_methods = ["nelder-mead", "powell", "CG", "BFGS", "L-BFGS-B"]
grad_methods = ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]
hess_methods = ["trust-krylov", "trust-ncg", "trust-exact", "trust-constr", "Newton-CG"]
hessp_methods = ["trust-krylov", "trust-ncg", "trust-constr", "Newton-CG"]


def rosen(x, a, b) -> float:
    """The Rosenbrock function"""
    return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b


def rosen_grad(x, a, b):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]

    grad = np.zeros_like(x)
    grad[0] = -4 * a * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    grad[-1] = 2 * a * (x[-1] - x[-2] ** 2)
    grad[1:-1] = 2 * a * (xm - xm_m1**2) - 4 * a * (xm_p1 - xm**2) * xm - 2 * (1 - xm)

    return grad


def rosen_hess(x, a, b):
    x = np.asarray(x)

    H = np.diag(-4 * a * x[:-1], 1) - np.diag(4 * a * x[:-1], -1)

    diagonal = np.zeros_like(x)
    diagonal[0] = 12 * a * x[0] ** 2 - 4 * a * x[1] + 2
    diagonal[-1] = 2 * a
    diagonal[1:-1] = 2 * a + 2 + 12 * a * x[1:-1] ** 2 - 4 * a * x[2:]

    H = H + np.diag(diagonal)
    return H


def rosen_hess_p(x, p, a, b):
    x = np.asarray(x)

    Hp = np.zeros_like(x)
    Hp[0] = (12 * a * x[0] ** 2 - 4 * a * x[1] + 2) * p[0] - 4 * a * x[0] * p[1]
    Hp[1:-1] = (
        -4 * a * x[:-2] * p[:-2]
        + (2 * a + 2 + 12 * a * x[1:-1] ** 2 - 4 * a * x[2:]) * p[1:-1]
        - 4 * a * x[1:-1] * p[2:]
    )
    Hp[-1] = -4 * a * x[-2] * p[-2] + 2 * a * p[-1]

    return Hp


def rosen_fused(x, a, b):
    y = sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b

    dy = np.zeros_like(x)
    dy[0] = -4 * a * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    dy[1:-1] = (
        2 * a * (x[1:-1] - x[:-2] ** 2)
        - 4 * a * (x[2:] - x[1:-1] ** 2) * x[1:-1]
        - 2 * (1 - x[1:-1])
    )
    dy[-1] = 2 * a * (x[-1] - x[-2] ** 2)

    return y, dy


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_rosen(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(partial(rosen, a=100, b=0), x0, method=method, tol=1e-8, maxiter=5000)

    assert isinstance(res, OptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_rosen_with_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen, x0, method=method, args=(0.5, 1.0), tol=1e-20, maxiter=5000)

    assert isinstance(res, OptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 1.0)


@pytest.mark.parametrize("method", grad_methods, ids=grad_methods)
def test_rosen_with_jac(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method=method,
        jac=partial(rosen_grad, a=100, b=0),
        tol=1e-20,
    )
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", grad_methods, ids=grad_methods)
def test_rosen_with_jac_and_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen, x0, args=(100, 0), method=method, jac=rosen_grad, tol=1e-20)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", grad_methods, ids=grad_methods)
def test_fused_rosen_with_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen_fused, x0, method=method, args=(0.5, 1.0), tol=1e-20)

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 1.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hess_methods, ids=hess_methods)
def test_rosen_with_hess(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method=method,
        jac=partial(rosen_grad, a=100, b=0),
        hess=partial(rosen_hess, a=100, b=0),
        tol=1e-20,
        maxiter=10000,
    )

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hess_methods, ids=hess_methods)
def test_rosen_with_hess_and_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        rosen,
        x0,
        args=(100, 0),
        method=method,
        jac=rosen_grad,
        hess=rosen_hess,
        tol=1e-20,
        maxiter=10000,
    )

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hessp_methods, ids=hessp_methods)
def test_rosen_with_hessp(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method=method,
        jac=partial(rosen_grad, a=100, b=0),
        hessp=partial(rosen_hess_p, a=100, b=0),
        tol=1e-20,
    )

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hessp_methods, ids=hessp_methods)
def test_rosen_with_hessp_and_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        rosen, x0, args=(100, 0), method=method, jac=rosen_grad, hessp=rosen_hess_p, tol=1e-20
    )

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


def test_constrained_rosen():
    def cons_f(x):
        return [x[0] ** 2 + x[1], x[0] ** 2 - x[1]]

    def cons_J(x):
        return [[2 * x[0], 1], [2 * x[0], -1]]

    def cons_H(x, v):
        return v[0] * np.array([[2, 0], [0, 0]]) + v[1] * np.array([[2, 0], [0, 0]])

    bounds = Bounds([0, -0.5], [1.0, 2.0])
    linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

    x0 = np.array([0.5, 0])

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method="trust-constr",
        jac=partial(rosen_grad, a=100, b=0),
        hess=partial(rosen_hess, a=100, b=0),
        constraints=[linear_constraint, nonlinear_constraint],
        bounds=bounds,
        tol=1e-20,
    )

    assert_allclose(res.x, np.array([0.41494531, 0.17010937]), atol=1e-5, rtol=1e-5)

    def rosen_hess_linop(x):
        def matvec(p):
            return rosen_hess_p(x, p)

        return LinearOperator((2, 2), matvec=matvec)

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method="trust-constr",
        jac=partial(rosen_grad, a=100, b=0),
        hess=partial(rosen_hess, a=100, b=0),
        constraints=[linear_constraint, nonlinear_constraint],
        options={"verbose": 1},
        bounds=bounds,
        tol=1e-20,
    )
    assert_allclose(res.x, np.array([0.41494531, 0.17010937]), atol=1e-5, rtol=1e-5)


def test_minimize_with_external_progressbar():
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    progress = ToggleableProgress()
    task1 = progress.add_task("task1")
    task2 = progress.add_task("task2", f_value=0.0)

    res = minimize(
        partial(rosen, a=100, b=0),
        x0,
        method="L-BFGS-B",
        tol=1e-8,
        maxiter=5000,
        progressbar=progress,
        progress_task=task2,
    )

    assert isinstance(res, OptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)

    assert task1 == 0
    assert task2 > 0
