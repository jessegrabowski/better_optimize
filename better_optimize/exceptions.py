class StopOptimization(StopIteration):
    """Raise from inside a callback to request a graceful early stop.

    Subclasses :class:`StopIteration` so that both SciPy's callback machinery and
    ``better_optimize``'s own early-stopping wrapper halt on it. When raised, the solver returns
    the best point seen so far with ``success=False``.
    """


__all__ = ["StopOptimization"]
