import argparse

from functools import partial

import numpy as np

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import root as scipy_root

from better_optimize.wrapper import ObjectiveWrapper, optimizer_early_stopping_wrapper

N_EXEC = 0
RAISED = 0


def f(x: np.ndarray):
    global N_EXEC
    global RAISED
    N_EXEC += 1

    if N_EXEC < 10 or RAISED:
        return x**2

    else:
        RAISED = 1
        raise KeyboardInterrupt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="store_true")
    parser.add_argument("--method", type=str)

    args = parser.parse_args()
    root = args.root
    method = args.method

    objective = ObjectiveWrapper(f=f, root=root)

    f_optim = partial(
        scipy_root if root else scipy_minimize,
        fun=objective,
        x0=np.array([1.0]),
        method=method,
        callback=objective.callback if not root else None,
    )

    res = optimizer_early_stopping_wrapper(f_optim)

    return res


if __name__ == "__main__":
    res = main()
    print(res)
