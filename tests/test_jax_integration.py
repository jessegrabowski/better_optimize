import numpy as np
import pytest

from numpy.testing import assert_allclose

from better_optimize.basinhopping import basinhopping
from better_optimize.minimize import minimize
from better_optimize.root import root

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(scope="module")
def jax_1d_quadratic():
    def f(x):
        return jnp.sum((x - 1.0) ** 2)

    return jax.value_and_grad(f)


def test_minimize_with_jax_array(jax_1d_quadratic):
    x0 = np.array([2.0])

    res = minimize(jax_1d_quadratic, x0, method="L-BFGS-B")

    assert res.success
    assert_allclose(res.x, np.array([1.0]))

    f_val, _ = jax_1d_quadratic(res.x)
    assert isinstance(f_val, jax.Array)
    assert_allclose(np.asarray(f_val), 0.0, atol=1e-6, rtol=1e-6)


def test_root_with_jax_array():
    def f(x):
        return jnp.array(x) - 2.0

    x0 = np.array([0.0])

    res = root(f, x0, method="hybr")

    assert res.success
    assert_allclose(res.x, np.array([2.0]), atol=1e-4, rtol=1e-4)
    f_val = f(res.x)
    assert isinstance(f_val, jax.Array)
    assert_allclose(np.asarray(f_val), 0.0, atol=1e-6, rtol=1e-6)


def test_basinhopping_with_jax_array(jax_1d_quadratic):
    x0 = np.array([3.0])

    res = basinhopping(
        jax_1d_quadratic,
        x0=x0,
        minimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-6},
        niter=20,
        progressbar=False,
    )

    assert res.success
    assert_allclose(res.x, np.array([1.0]), atol=1e-3, rtol=1e-3)

    val, grad = jax_1d_quadratic(res.x)
    assert isinstance(val, jax.Array)
    assert isinstance(grad, jax.Array)
    assert_allclose(np.asarray(val), 0.0, atol=1e-5, rtol=1e-5)
    assert_allclose(np.asarray(grad), np.array([0.0]), atol=1e-5, rtol=1e-5)
