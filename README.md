# Better Optimization!

`better_optimize` is a friendlier front-end to scipy's `optimize.minimize` and `optimize.root` functions. Features
include:

- Progress bar!
- Early stopping!
- Better propagation of common arguments (`maxiters`, `tol`)!

## Installation

To install `better_optimize`, simply use conda:

```bash
conda install -c conda-forge better_optimize
```

Or, if you prefer pip:

```bash
pip install better_optimize
```

## What does `better_optimize` provide over basic scipy?

### 1. Progress Bars

All optimization routines in `better_optimize` can display a rich, informative progress bar using the [rich](https://github.com/Textualize/rich) library. This includes:

- Iteration counts, elapsed time, and objective values.
- Gradient and Hessian norms (when available).
- Separate progress bars for global (basinhopping) and local (minimizer) steps.
- Toggleable display for headless or script environments.

### 2. Flat and Generalized Keyword Arguments

- No more nested `options` dictionaries! You can pass `tol`, `maxiter`, and other common options directly as top-level keyword arguments.
- `better_optimize` automatically sorts and promotes these arguments to the correct place for each optimizer.
- Generalizes argument handling: always provides `tol` and `maxiter` (or their equivalents) to the optimizer, even if you forget.

### 3. Argument Checking and Validation

- Automatic checking of provided gradient (`jac`), Hessian (`hess`), and Hessian-vector (`hessp`) functions.
- Warns if you provide unnecessary or unused arguments for a given method.
- Detects and handles fused objective functions (e.g., functions returning `(loss, grad)` or `(loss, grad, hess)` tuples).
- Ensures that the correct function signatures and return types are used for each optimizer.

### 4. LRUCache1 for Fused Functions

- Provides an `LRUCache1` utility to cache the results of expensive objective/gradient/Hessian computations.
- Especially useful for triple-fused functions that return value, gradient, and Hessian together, avoiding redundant computation.
- Totally invisible -- just pass a function with 3 return values. Seamlessly integrated into the optimization workflow.

### 5. Robust Basin-Hopping with Failure Tolerance

- Enhanced `basinhopping` implementation allows you to continue even if the local minimizer fails.
- Optionally accepts and stores failed minimizer results if they improve the global minimum.
- Useful for noisy or non-smooth objective functions where local minimization may occasionally fail.

---

## Example Usage

### Simple Example

```python
from better_optimize import minimize

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

result = minimize(
    rosenbrock,
    x0=[-1, 2],
    method="L-BFGS-B",
    tol=1e-6,
    maxiter=1000,
    progressbar=True,  # Show a rich progress bar!
)
```

```shell
  Minimizing                                         Elapsed   Iteration   Objective    ||grad||
 ──────────────────────────────────────────────────────────────────────────────────────────────────
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0:00:00   721/721     0.34271757   0.92457651
```

The result object is a standard `OptimizeResult` from `scipy.optimize`, so there are no surprises there!

### Triple-Fused Function using Pytensor

```python
from better_optimize import minimize
import pytensor.tensor as pt
from pytensor import function
import numpy as np

x = pt.vector('x')
value = pt.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
grad = pt.grad(value, x)
hess = pt.hessian(value, x)

fused_fn = function([x], [value, grad, hess])
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

result = minimize(
    fused_fn, # No need to set flags separately, `better_optimize` handles it!
    x0=x0,
    method="Newton-CG",
    tol=1e-6,
    maxiter=1000,
    progressbar=True,  # Show a rich progress bar!
)
```

Many sub-computations are repeated between the objective, gradient, and hessian functions. Scipy allows you to pass a
fused value_and_grad function, but `better_optimize` also lets you pass a triple-fused value_grad_and_hess function.
This avoids redundant computation and speeds up the optimization process.


## Contributing

We welcome contributions! If you find a bug, have a feature request, or want to improve the documentation, please open
an issue or submit a pull request on GitHub.
