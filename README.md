# Better Optimization!

`better_optimize` is a friendlier front-end to scipy's `optimize.minimize` and `optimize.root` functions. Features
include:

- Progress bar!
- Early stopping!
- Better propagation of common arguments (`maxiters`, `tol`)!
- A typed, documented configuration object for every method, so swapping optimizers is a one-argument change!

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
- Generalizes argument handling: always provides `tol` and `maxiter` (or their equivalents) to the optimizer, even if you forget. `root` methods disagree about whether that budget is called `maxiter`, `maxfev` or `maxfun`, and you can pass `maxiter` to any of them.
- Seven of the ten `root` methods take part of their configuration nested inside `jac_options`, which scipy splats into the jacobian it builds. Pass it as a mapping or as the typed object for that jacobian (`BroydenJacOptions`, `KrylovJacOptions`, and so on). An option that belongs there raises if you pass it at the top level, where scipy would warn and drop it.

### 3. Typed Configuration Objects

Every method scipy supports has a dataclass whose fields are exactly the options that method accepts, typed, defaulted, and documented with scipy's own prose. `help(BFGSConfig)` tells you what `c2` does without a trip to `scipy.optimize.show_options`.

- Pass one wherever a method name works: `minimize(f, x0, method=BFGSConfig(gtol=1e-8))`. The string form keeps working unchanged.
- A misspelled option raises `TypeError` naming it, rather than reaching scipy and being dropped with a warning nobody reads.
- Fields and defaults are checked against scipy's own function signatures by test, so a scipy release that moves a default fails a test instead of drifting quietly.
- One `minimize` call reaches local minimizers, basinhopping and differential evolution alike, because the configuration carries which solver runs it.

### 4. Argument Checking and Validation

- Automatic checking of provided gradient (`jac`), Hessian (`hess`), and Hessian-vector (`hessp`) functions.
- Warns if you provide unnecessary or unused arguments for a given method.
- Detects and handles fused objective functions (e.g., functions returning `(loss, grad)` or `(loss, grad, hess)` tuples).
- Ensures that the correct function signatures and return types are used for each optimizer.

### 5. LRUCache1 for Fused Functions

- Provides an `LRUCache1` utility to cache the results of expensive objective/gradient/Hessian computations.
- Especially useful for triple-fused functions that return value, gradient, and Hessian together, avoiding redundant computation.
- Totally invisible -- just pass a function with 3 return values. Seamlessly integrated into the optimization workflow.

### 6. Robust Basin-Hopping with Failure Tolerance

- Enhanced `basinhopping` implementation allows you to continue even if the local minimizer fails.
- Optionally accepts and stores failed minimizer results if they improve the global minimum.
- Useful for noisy or non-smooth objective functions where local minimization may occasionally fail.

### 7. Uniform Callback API

scipy passes callbacks a different argument for almost every method: `callback(xk)`, `callback(xk, state)`,
`callback(intermediate_result)`, or `callback(x, f)` for root finding. `better_optimize` normalizes them to a single
`callback(res)`, where `res` is an `OptimizeResult` with the current `res.x`, `res.fun`, and `res.nit` -- plus `res.jac`
when a gradient is available, and solver-specific fields like `res.accept` (basinhopping) or `res.convergence`
(differential evolution).

- The return value is ignored, so a callback can report a value (an ELBO, a logged loss) without stopping the run.
- Raise `StopOptimization` to stop early; the best result so far is returned with `success=False`.
- The same callback works across `minimize`, `root`, `basinhopping`, and `differential_evolution`. `hybr` and `lm` are
  the exception -- scipy never calls a callback for them, so it is ignored with a warning.

---

## Example Usage

### Simple Example

```python
import numpy as np
from better_optimize import minimize
from better_optimize.configuration import LBFGSBConfig

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

result = minimize(
    rosenbrock,
    x0=np.array([-1.0, 2.0]),
    method=LBFGSBConfig(tol=1e-6, maxiter=1000),
    progressbar=True,  # Show a rich progress bar!
)
```

```shell
  Minimizing                                         Elapsed   Iteration   Objective    ||grad||
 ──────────────────────────────────────────────────────────────────────────────────────────────────
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0:00:00   721/721     0.34271757   0.92457651
```

The result object is a standard `OptimizeResult` from `scipy.optimize`, so there are no surprises there!

### Swapping Optimizers

A configuration object says which solver runs it, so trying a different optimizer family is a change to one argument rather than a rewrite. Nothing below changes but the loop variable.

```python
import numpy as np
from better_optimize import minimize
from better_optimize.configuration import (
    BasinHoppingConfig,
    DifferentialEvolutionConfig,
    LBFGSBConfig,
)

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

x0 = np.array([-1.0, 2.0])
bounds = [(-5.0, 5.0), (-5.0, 5.0)]

for config in [
    LBFGSBConfig(maxiter=500),
    BasinHoppingConfig(niter=25, minimizer_config=LBFGSBConfig()),
    DifferentialEvolutionConfig(popsize=20, rng=0),
]:
    result = minimize(rosenbrock, x0, method=config, bounds=bounds, progressbar=False)
    print(f"{config.method_name:25s} {result.fun:.3e}")
```

```shell
L-BFGS-B                  9.191e-12
basinhopping              3.696e-12
differential_evolution    4.980e-30
```

`L-BFGS-B` is a local minimizer, `basinhopping` restarts one from perturbed points, and `differential_evolution` is a population method that never calls the local minimizer at all. They take different arguments, and scipy reaches them through different entry points. The configuration object is what absorbs that difference.

### scipy Backwards Compatibility

For users who have already memorized every combination of algorithm and option, none of the above is required. Method names still work, and so do flat keywords -- including the ones scipy keeps inside its `options` dictionary, which `better_optimize` gathers up for you. If you memorized that part too, pass `options` yourself and it is merged and checked the same way.

```python
import numpy as np
from better_optimize import minimize
from better_optimize.configuration import LBFGSBConfig

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

x0 = np.array([-1.0, 2.0])

# A configuration object
minimize(rosenbrock, x0, method=LBFGSBConfig(ftol=1e-6, gtol=1e-6, maxiter=1000, maxfun=1000))

# Flat keywords, including ftol and gtol, which scipy keeps in `options`
minimize(rosenbrock, x0, method="L-BFGS-B", ftol=1e-6, gtol=1e-6, maxiter=1000)

# Or the options dictionary itself
minimize(rosenbrock, x0, method="L-BFGS-B",
         options={"ftol": 1e-6, "gtol": 1e-6, "maxiter": 1000, "maxfun": 1000})
```

Those three are the same call and return the same answer. The flat form additionally spreads a top-level `maxiter` across every budget the method has, which for `L-BFGS-B` means `maxfun` -- that is why the first and third forms name it and the second does not.

All three are checked the same way, which is the part that is not backwards compatible. A misspelled option raises `TypeError` before the run starts, with a suggestion:

```python
minimize(rosenbrock, x0, method="L-BFGS-B", options={"gtoll": 1e-6})
# TypeError: LBFGSBConfig.__init__() got an unexpected keyword argument 'gtoll'. Did you mean 'gtol'?
```

scipy would have accepted that call, run to completion, and mentioned the dropped option in an `OptimizeWarning` afterwards.

### Callbacks and Early Stopping

Pass a `callback` and it runs after each iteration with an `OptimizeResult`. Its return value is ignored; raise
`StopOptimization` to stop and return the best result so far.

```python
import numpy as np
from better_optimize import minimize, StopOptimization
from better_optimize.configuration import LBFGSBConfig

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

history = []

def callback(res):
    history.append(res.fun)  # res.x, res.fun, res.nit; res.jac when available
    if res.fun < 1e-8:
        raise StopOptimization

result = minimize(
    rosenbrock,
    x0=np.array([-1.0, 2.0]),
    method=LBFGSBConfig(),
    callback=callback,
)
```

The same callback works for `root` (`res.fun` is the residual vector), `basinhopping` (`res.accept`), and
`differential_evolution` (`res.convergence`).

### Triple-Fused Function using Pytensor

```python
from better_optimize import minimize
from better_optimize.configuration import NewtonCGConfig
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
    method=NewtonCGConfig(tol=1e-6, maxiter=1000),
    progressbar=True,  # Show a rich progress bar!
)
```

Many sub-computations are repeated between the objective, gradient, and hessian functions. Scipy allows you to pass a
fused value_and_grad function, but `better_optimize` also lets you pass a triple-fused value_grad_and_hess function.
This avoids redundant computation and speeds up the optimization process.

### Global Optimization with Differential Evolution

`differential_evolution` searches a bounded region with a population instead of descending from a starting point, which suits objectives with many local minima and no useful gradient. It takes `bounds` rather than `x0`, and gets the same progress bar and callback treatment as everything else.

```python
import numpy as np
from better_optimize import differential_evolution

def rastrigin(x):
    return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))

result = differential_evolution(
    rastrigin,
    bounds=[(-5.12, 5.12)] * 5,
    popsize=20,
    rng=0,
)

print(result.x)    # [-0. -0. -0. -0. -0.]
print(result.fun)  # 0.0
```

Rastrigin has a local minimum at roughly every integer lattice point, so a gradient method started anywhere but the middle converges to the wrong one.

### Chaining Optimizers with `sequential_optimize`

It is often quicker to get close with a robust method and finish with a fast one. `sequential_optimize` runs a list of stages, forwarding each stage's best point into the next stage's starting point, and reports what every stage did.

```python
import numpy as np
from better_optimize import sequential_optimize
from better_optimize.configuration import BFGSConfig, NelderMeadConfig

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

result = sequential_optimize(
    rosenbrock,
    x0=np.array([-1.2, 1.0]),
    stages=[NelderMeadConfig(maxiter=200), BFGSConfig(gtol=1e-10)],
)

print(result.x)           # [0.99999552 0.99999104]
print(result.best_stage)  # 1
result.summary()          # Rich table of every stage, with the winner starred
```

A stage can also be a dict carrying its own `solver` and keyword arguments, which is what you need when a stage wants its own `bounds`, `constraints`, or starting point. By default the chain stops at the first stage that fails. Pass `on_failure="continue"` to run the rest anyway and keep the best result from those that worked.

### Parallel Optimization from Multiple Starting Points

Real-world objectives often have multiple local minima. A common workaround is to throw many random starting points at
the optimizer and keep the best result. `better_optimize` makes this painless with `multi_optimize`:

```python
import numpy as np
from better_optimize import minimize, multi_optimize
from better_optimize.configuration import LBFGSBConfig

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

result = multi_optimize(
    solver=minimize,
    solver_kwargs=dict(f=rosenbrock, method=LBFGSBConfig(tol=1e-10)),
    x0=np.zeros(5),
    n_runs=16,
    init_strategy="uniform",
    bounds=(-5, 5),
    backend="loky",
    n_jobs=-1,
    seed=42,
    progressbar=True,
)

print(result.best)       # Best OptimizeResult
print(result.x_best)     # Best parameter vector
print(result.fun_best)   # Best objective value
result.summary()          # Rich table of all runs, ranked
```

> The `loky` (process) backend needs an `if __name__ == "__main__":` guard when run as a script on macOS or Windows.
> Notebooks and the `threading`/`sequential` backends don't.

`multi_optimize` works with **any** solver that follows the `(x0, **kwargs) -> OptimizeResult` signature -- that includes
`minimize`, `root`, `basinhopping`, or your own custom wrapper. It just calls `solver(x0=x0_i, **solver_kwargs)` for
each starting point; it never inspects the solver internals.

A few highlights:

- **Initialization strategies** -- `"uniform"`, `"normal"`, `"sobol"`, `"lhs"`, or pass your own callable. Bounded
  strategies (`uniform`, `sobol`, `lhs`) require a `bounds` argument; `"normal"` perturbs around `x0` with a
  configurable `init_scale`. Or just pass an explicit `list[np.ndarray]` as `x0` and skip the generation entirely.
- **Parallel backends** -- `"sequential"` (for debugging), `"loky"` (CPU-bound work, default), or `"threading"`
  (GIL-releasing code). Under the hood this is `joblib`, so the usual `n_jobs=-1` convention works.
- **BLAS thread control** -- When many workers each spawn a full BLAS/OpenMP thread pool, you get noisy-neighbor
  over-subscription. The `blas_cores` argument (default `"auto"`) caps the total thread budget so workers don't
  fight over the same cores.

The returned `MultiStartResult` gives you `best`, `top_k(k)`, `ranked()`, `success_rate`, a `summary()` table, and
`to_dataframe()` for further analysis.


## Contributing

We welcome contributions! If you find a bug, have a feature request, or want to improve the documentation, please open
an issue or submit a pull request on GitHub.
