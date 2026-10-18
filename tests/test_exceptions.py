import pickle

from scipy.optimize import OptimizeResult

from better_optimize import OptimizationInterrupted


def test_interrupt_survives_pickling():
    # loky workers send exceptions back to the parent pickled.
    interrupt = OptimizationInterrupted(OptimizeResult(x=[1.0], fun=2.0), "stopped")
    restored = pickle.loads(pickle.dumps(interrupt))

    assert isinstance(restored, KeyboardInterrupt)
    assert restored.result.fun == 2.0
    assert str(restored) == "stopped"
