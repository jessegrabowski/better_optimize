import logging

from better_optimize._version import get_versions
from better_optimize.basinhopping import basinhopping
from better_optimize.differential_evolution import differential_evolution
from better_optimize.minimize import minimize
from better_optimize.multi_optimize import multi_optimize
from better_optimize.root import root
from better_optimize.sequential_optimize import sequential_optimize

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


__version__ = get_versions()["version"]
__all__ = [
    "basinhopping",
    "differential_evolution",
    "minimize",
    "multi_optimize",
    "root",
    "sequential_optimize",
]
