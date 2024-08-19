from typing import get_args

import numpy as np
import pytest

from numpy.testing import assert_allclose
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult
from scipy.sparse.linalg import LinearOperator

from better_optimize.constants import minimize_method
from better_optimize.minimize import minimize

all_methods = list(get_args(minimize_method))
no_grad_methods = ["nelder-mead", "powell", "CG", "BFGS", "L-BFGS-B"]
grad_methods = ["CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"]
hess_methods = ["trust-krylov", "trust-ncg", "trust-exact", "trust-constr", "Newton-CG"]
hessp_methods = ["trust-krylov", "trust-ncg", "trust-constr", "Newton-CG"]


def rosen(x) -> float:
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rosen_with_args(x: np.ndarray, a: float, b: float) -> float:
    """The Rosenbrock function with additional arguments"""
    return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]

    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)

    return der


def rosen_hess(x):
    x = np.asarray(x)

    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)

    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]

    H = H + np.diag(diagonal)
    return H


def rosen_hess_p(x, p):
    x = np.asarray(x)

    Hp = np.zeros_like(x)
    Hp[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = (
        -400 * x[:-2] * p[:-2]
        + (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1]
        - 400 * x[1:-1] * p[2:]
    )
    Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]

    return Hp


def rosen_fused_args(x, a, b):
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

    res = minimize(rosen, x0, method=method, tol=1e-8)

    assert isinstance(res, OptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_rosen_with_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen_with_args, x0, method=method, args=(0.5, 1.0), tol=1e-20, maxiter=1000)

    assert isinstance(res, OptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 1.0)


@pytest.mark.parametrize("method", grad_methods, ids=grad_methods)
def test_rosen_with_jac(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen, x0, method=method, jac=rosen_der, tol=1e-20)
    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", grad_methods, ids=grad_methods)
def test_fused_rosen_with_args(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen_fused_args, x0, method=method, args=(0.5, 1.0), tol=1e-20)

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 1.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hess_methods, ids=hess_methods)
def test_rosen_with_hess(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(
        rosen, x0, method=method, jac=rosen_der, hess=rosen_hess, tol=1e-20, maxiter=10000
    )

    assert_allclose(res.x, np.ones(5), atol=1e-5, rtol=1e-5)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", hessp_methods, ids=hessp_methods)
def test_rosen_with_hessp(method: minimize_method):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen, x0, method=method, jac=rosen_der, hessp=rosen_hess_p, tol=1e-20)

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
        rosen,
        x0,
        method="trust-constr",
        jac=rosen_der,
        hess=rosen_hess,
        constraints=[linear_constraint, nonlinear_constraint],
        bounds=bounds,
    )

    assert_allclose(res.x, np.array([0.41494531, 0.17010937]))

    def rosen_hess_linop(x):
        def matvec(p):
            return rosen_hess_p(x, p)

        return LinearOperator((2, 2), matvec=matvec)

    res = minimize(
        rosen,
        x0,
        method="trust-constr",
        jac=rosen_der,
        hess=rosen_hess_linop,
        constraints=[linear_constraint, nonlinear_constraint],
        options={"verbose": 1},
        bounds=bounds,
    )
    assert_allclose(res.x, np.array([0.41494531, 0.17010937]))

    res = minimize(
        rosen,
        x0,
        method="trust-constr",
        jac=rosen_der,
        hessp=rosen_hess_p,
        constraints=[linear_constraint, nonlinear_constraint],
        options={"verbose": 1},
        bounds=bounds,
    )
    assert_allclose(res.x, np.array([0.41494531, 0.17010937]))

    res = minimize(
        rosen,
        x0,
        method="trust-constr",
        jac=rosen_der,
        hessp=rosen_hess_p,
        constraints=[linear_constraint, nonlinear_constraint],
        options={"verbose": 1},
        bounds=bounds,
    )
    assert_allclose(res.x, np.array([0.41494531, 0.17010937]))
