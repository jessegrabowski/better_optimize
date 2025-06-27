import sys

from functools import partial
from typing import get_args

import numpy as np
import pytest

from numpy.testing import assert_allclose

from better_optimize.constants import root_method
from better_optimize.root import root
from better_optimize.utilities import LRUCache1

all_methods = list(get_args(root_method))


def func(x, a, b):
    return a * x + b * np.cos(x)


def func2(x, a, b):
    f = [a * x[0] * np.cos(x[1]) - 4, x[1] * x[0] - b * x[1] - 5]

    return np.array(f)


def func2_jac(x, a, b):
    df = np.array([[a * np.cos(x[1]), -a * x[0] * np.sin(x[1])], [x[1], x[0] - b]])

    return df


def func2_fused(x, a, b):
    return func2(x, a, b), func2_jac(x, a, b)


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

    res = root(partial(func, a=1, b=2), x0, method=method, **kwargs)
    assert_allclose(res.x, [-1.029866529322393])
    assert_allclose(res.fun, [0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", all_methods, ids=all_methods)
def test_root_with_args(method: root_method):
    if "mixing" in method:
        pytest.skip("mixing methods fail even on this, just skipping")

    x0 = np.array([0.1])
    kwargs = {}

    res = root(func, args=(1, 2), x0=x0, method=method, **kwargs)
    assert_allclose(res.x, [-1.029866529322393])
    assert_allclose(res.fun, [0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", ["hybr", "lm"], ids=["hybr", "lm"])
def test_root_with_jac(method: root_method):
    x0 = np.array([0.8, 0.8])
    res = root(partial(func2, a=1, b=1), x0, jac=partial(func2_jac, a=1, b=1), method=method)

    assert_allclose(res.x, np.array([6.50409711, 0.90841421]))
    assert_allclose(res.fun, [0.0, 0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", ["hybr", "lm"], ids=["hybr", "lm"])
def test_root_with_jac_and_args(method: root_method):
    x0 = np.array([0.8, 0.8])
    res = root(func2, x0, jac=func2_jac, args=(1, 1), method=method)

    assert_allclose(res.x, np.array([6.50409711, 0.90841421]))
    assert_allclose(res.fun, [0.0, 0.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", ["hybr", "lm"], ids=["hybr", "lm"])
def test_root_fused_objective(method: root_method, monkeypatch):
    cache_holder = {}

    def accessible_LRUCache1(*args, **kwargs):
        cache = LRUCache1(*args, **kwargs)
        cache_holder["cache"] = cache
        return cache

    root_mod = sys.modules["better_optimize.root"]
    with monkeypatch.context() as c:
        c.setattr(root_mod, "LRUCache1", accessible_LRUCache1)

        x0 = np.array([0.8, 0.8])
        res = root(partial(func2_fused, a=1, b=1), x0, method=method)

    assert_allclose(res.x, np.array([6.50409711, 0.90841421]))
    assert_allclose(res.fun, [0.0, 0.0], atol=1e-8, rtol=1e-8)

    cache = cache_holder.get("cache")

    assert cache is not None
    assert (cache.cache_hits + cache.cache_misses) > 0
    assert cache.value_and_grad_calls > 0
    assert cache.hess_calls == 0  # Hessian not used by root


@pytest.mark.parametrize(
    "method, options",
    [
        ("krylov", {"disp": False}),
        ("broyden2", {"disp": False, "max_rank": 50}),
        ("anderson", {"disp": False, "M": 10}),
    ],
    ids=["krylov", "broyden2", "anderson"],
)
def test_large_root(method: root_method, options: dict):
    n = [75, 75]
    y = [1.0 / (n[0] - 1), 1.0 / (n[1] - 1)]
    bounds = [0, 0, 1, 0]

    # solve
    x0 = np.zeros(n, float)
    res = root(func3, x0, args=(y, bounds), method=method, maxiter=10000, **options)

    assert_allclose(res.fun, 0.0, atol=1e-5, rtol=1e-5)
