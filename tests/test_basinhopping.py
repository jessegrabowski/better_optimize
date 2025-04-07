import numpy as np

from better_optimize.basinhopping import basinhopping


def func1d(x):
    f = np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * np.sin(14.5 * x - 0.3) + 2.0 * x + 0.2)
    return f, df


def func2d_nograd(x):
    f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f


def func2d(x):
    f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    df = np.zeros(2)
    df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2.0 * x[0] + 0.2
    df[1] = 2.0 * x[1] + 0.2
    return f, df


def func2d_easyderiv(x):
    f = 2.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + 2.0 * x[1] ** 2 - 6.0 * x[0]
    df = np.zeros(2)
    df[0] = 4.0 * x[0] + 2.0 * x[1] - 6.0
    df[1] = 2.0 * x[0] + 4.0 * x[1]

    return f, df


def test_basinhopping_1d():
    res = basinhopping(
        func1d,
        x0=[1.0],
        minimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )
    assert res.success is True
    np.testing.assert_allclose(res.x, -0.195, atol=1e-3, rtol=1e-3)


def test_basinhopping_2d():
    res = basinhopping(
        func2d,
        x0=[1.0, 1.0],
        minimizer_kwargs={"method": "L-BFGS-B", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )

    assert res.success is True
    np.testing.assert_allclose(res.x, np.array([-0.195, -0.1]), atol=1e-3, rtol=1e-3)


def test_basinhopping_nograd():
    res = basinhopping(
        func2d,
        x0=[1.0, 1.0],
        minimizer_kwargs={"method": "nelder-mead", "tol": 1e-8},
        niter=100,
        progressbar=True,
    )

    assert res.success is True
    np.testing.assert_allclose(res.x, np.array([-0.195, -0.1]), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_basinhopping_1d()
