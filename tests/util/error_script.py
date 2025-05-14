import argparse
import json

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

    objective = ObjectiveWrapper(f=f, root=root, progressbar=False)

    f_optim = partial(
        scipy_root if root else scipy_minimize,
        fun=objective,
        x0=np.array([1.0]),
        method=method,
        callback=objective.callback if not root else None,
    )

    res = optimizer_early_stopping_wrapper(f_optim)

    # Save result as a serializable JSON
    output = {
        "x": res.x.tolist(),
        "fun": res.fun if not root else res.fun.tolist(),
        "success": res.success,
        "message": res.message,
    }
    output = json.dumps(output)

    return output


if __name__ == "__main__":
    res = main()
    print(res)
