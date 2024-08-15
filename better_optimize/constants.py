from typing import Literal

minimize_method = Literal[
    'nelder-mead',
    'powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov'
]

MODE_KWARGS = {
    "nelder-mead": {"uses_grad": False, "uses_hess": False, "uses_hessp": False},
    "powell": {"uses_grad": False, "uses_hess": False, "uses_hessp": False},
    "CG": {"uses_grad": True, "uses_hess": False, "uses_hessp": False},
    "BFGS": {"uses_grad": True, "uses_hess": False, "uses_hessp": False},
    "Newton-CG": {"uses_grad": True, "uses_hess": True, "uses_hessp": True},
    "L-BFGS-B": {"uses_grad": True, "uses_hess": False, "uses_hessp": False},
    "TNC": {"uses_grad": True, "uses_hess": False, "uses_hessp": False},
    "COBYLA": {"uses_grad": False, "uses_hess": False, "uses_hessp": False},
    "SLSQP": {"uses_grad": True, "uses_hess": False, "uses_hessp": False},
    "trust-constr": {"uses_grad": True, "uses_hess": True, "uses_hessp": True},
    "dogleg": {"uses_grad": True, "uses_hess": True, "uses_hessp": False},
    "trust-ncg": {"uses_grad": True, "uses_hess": True, "uses_hessp": True},
    "trust-exact": {"uses_grad": True, "uses_hess": True, "uses_hessp": False},
    "trust-krylov": {"uses_grad": True, "uses_hess": True, "uses_hessp": True},
}
