class StopOptimization(StopIteration):
    """Raise from inside a callback to request a graceful early stop.

    Subclasses :class:`StopIteration` so that both SciPy's callback machinery and
    ``better_optimize``'s own early-stopping wrapper halt on it. When raised, the solver returns
    the best point seen so far with ``success=False``.
    """


class OptimizationInterrupted(KeyboardInterrupt):
    """Raised when a solver is stopped by ``KeyboardInterrupt`` (Ctrl-C).

    Subclasses :class:`KeyboardInterrupt`, so an interrupt still stops any loop that calls a
    solver. The partial result is on :attr:`result`, of the same type the solver returns.
    For a nested solver (basinhopping, sequential_optimize, multi_optimize) it describes the
    outer search, and the inner solver's partial result is on ``__cause__.result``.

    Parameters
    ----------
    result : OptimizeResult, SequentialResult, or MultiStartResult
        The partial result at the moment of the interrupt.
    message : str
        What was interrupted, and how far it got.
    """

    def __init__(self, result, message: str):
        super().__init__(message)
        self.result = result

    def __reduce__(self):
        return type(self), (self.result, *self.args)


__all__ = ["OptimizationInterrupted", "StopOptimization"]
