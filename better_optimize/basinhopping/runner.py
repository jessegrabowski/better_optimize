from scipy.optimize._basinhopping import BasinHoppingRunner, Storage


class AllowFailureStorage(Storage):
    """
    Subclass of Storage that allows the minimizer to fail, but still updates the global minimum
    if the new point is better than the current global minimum.
    """

    def update(self, minres):
        if minres.fun < self.minres.fun:
            self._add(minres)
            return True
        else:
            return False


class AllowFailureBasinHoppingRunner(BasinHoppingRunner):
    """
    A subclass of BasinHoppingRunner that allows the minimizer to fail, but still updates the
    global minimum if the new point is better than the current global minimum.
    """

    def __init__(
        self, x0, minimizer, step_taking, accept_tests, accept_on_minimizer_fail=False, disp=False
    ):
        super().__init__(x0, minimizer, step_taking, accept_tests, disp=disp)

        if accept_on_minimizer_fail:
            self.storage = AllowFailureStorage(self.storage.minres)


__all__ = ["AllowFailureBasinHoppingRunner", "AllowFailureStorage"]
