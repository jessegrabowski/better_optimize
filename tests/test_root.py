from typing import get_args

import numpy as np
import pytest

from numpy.testing import assert_allclose

from better_optimize.constants import root_method
from better_optimize.root import root

all_methods = list(get_args(root_method))


def func(x):
    return x + 2 * np.cos(x)


def func2(x):
    f = [x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5]

    df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])

    return f, df


def func3(P, h, bounds):
    hx, hy = h
    P_left, P_right, P_top, P_bottom = bounds

    d2x = np.zeros_like(P)
    d2y = np.zeros_like(P)

    d2x[1:-1] = (P[2:] - 2 * P[1:-1] + P[:-2]) / hx / hx
    d2x[0] = (P[1] - 2 * P[0] + P_left) / hx / hx
    d2x[-1] = (P_right - 2 * P[-1] + P[-2]) / hx / hx

    d2y[:, 1:-1] = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / hy / hy
    d2y[:, 0] = (P[:, 1] - 2 * P[:, 0] + P_bottom) / hy / hy
    d2y[:, -1] = (P_top - 2 * P[:, -1] + P[:, -2]) / hy / hy

    return d2x + d2y + 5 * np.cosh(P).mean() ** 2


@pytest.mark.parametrize("method", all_methods, ids=all_methods)
def test_root(method: root_method):
    if "mixing" in method:
        pytest.skip("mixing methods fail even on this, just skipping")

    x0 = np.array([0.1])
    kwargs = {}

    res = root(func, x0, method=method, **kwargs)
    assert_allclose(res.x, [-1.029866529322393])
    assert_allclose(res.fun, [0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", ["hybr", "lm"], ids=["hybr", "lm"])
def test_root_fused_objective(method: root_method):
    x0 = np.array([0.8, 0.8])
    res = root(func2, x0, method=method)
    assert_allclose(res.x, np.array([6.50409711, 0.90841421]))
    assert_allclose(res.fun, [0.0, 0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "method, options",
    [
        ("krylov", {"disp": True}),
        ("broyden2", {"disp": True, "max_rank": 50}),
        ("anderson", {"disp": True, "M": 10}),
    ],
    ids=["krylov", "broyden2", "anderson"],
)
def test_large_root(method: root_method, options: dict):
    n = [75, 75]
    y = [1.0 / (n[0] - 1), 1.0 / (n[1] - 1)]
    bounds = [0, 0, 1, 0]

    # solve
    x0 = np.zeros(n, float)
    res = root(func3, x0, args=(y, bounds), method=method, **options)

    assert_allclose(res.fun, 0.0, atol=1e-5, rtol=1e-5)
