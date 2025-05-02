import logging

from better_optimize._version import get_versions
from better_optimize.basinhopping import basinhopping
from better_optimize.minimize import minimize
from better_optimize.root import root

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


__version__ = get_versions()["version"]
__all__ = ["minimize", "root", "basinhopping"]
