from typing import Literal

minimize_method = Literal[
    "nelder-mead",
    "powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]

root_method = Literal[
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
    "krylov",
    "df-sane",
]

TOLERANCES = ["xtol", "ftol", "gtol", "fatol", "xatol"]

MINIMIZE_MODE_KWARGS = {
    "nelder-mead": {
        "uses_grad": False,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "disp",
            "maxiter",
            "maxfev",
            "return_all",
            "initial_simplex",
            "xatol",
            "fatol",
            "adaptive",
            "bounds",
        ],
    },
    "powell": {
        "uses_grad": False,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": ["disp", "xtol", "ftol", "maxiter", "maxfev", "direc", "return_all"],
    },
    "CG": {
        "uses_grad": True,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "disp",
            "maxiter",
            "gtol",
            "norm",
            "eps",
            "return_all",
            "finite_diff_rel_step",
            "c1",
            "c2",
        ],
    },
    "BFGS": {
        "uses_grad": True,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "disp",
            "maxiter",
            "gtol",
            "norm",
            "eps",
            "return_all",
            "finite_diff_rel_step",
            "xrtol",
            "c1",
            "c2",
            "hess_inv0",
        ],
    },
    "Newton-CG": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": True,
        "valid_options": ["disp", "xtol", "maxiter", "eps", "return_all", "c1", "c2"],
    },
    "L-BFGS-B": {
        "uses_grad": True,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "disp",
            "maxcor",
            "ftol",
            "gtol",
            "eps",
            "maxfun",
            "maxiter",
            "iprint",
            "maxls",
            "finite_diff_rel_step",
        ],
    },
    "TNC": {
        "uses_grad": True,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "eps",
            "scale",
            "offset",
            "disp",
            "maxCGit",
            "eta",
            "stepmx",
            "accuracy",
            "minfev",
            "ftol",
            "xtol",
            "gtol",
            "rescale",
            "finite_diff_rel_step",
            "maxfun",
        ],
    },
    "COBYLA": {
        "uses_grad": False,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": ["rhobeg", "tol", "disp", "maxiter", "catol"],
    },
    "SLSQP": {
        "uses_grad": True,
        "uses_hess": False,
        "uses_hessp": False,
        "valid_options": [
            "ftol",
            "eps",
            "disp",
            "maxiter",
            "finite_diff_rel_step",
        ],
    },
    "trust-constr": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": True,
        "valid_options": [
            "gtol",
            "xtol",
            "barrier_tol",
            "sparse_jacobian",
            "initial_tr_radius",
            "initial_constr_penalty",
            "initial_barrier_parameter",
            "initial_barrier_tolerance",
            "factorization_method",
            "finite_diff_rel_step",
            "maxiter",
            "verbose",
            "disp",
        ],
    },
    "dogleg": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": False,
        "valid_options": ["initial_trust_radius", "max_trust_radius", "eta", "gtol"],
    },
    "trust-ncg": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": True,
        "valid_options": ["initial_trust_radius", "max_trust_radius", "eta", "gtol"],
    },
    "trust-exact": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": False,
        "valid_options": ["initial_trust_radius", "max_trust_radius", "eta", "gtol"],
    },
    "trust-krylov": {
        "uses_grad": True,
        "uses_hess": True,
        "uses_hessp": True,
        "valid_options": ["inexact"],
    },
}

root_method = Literal[
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
    "krylov",
    "df-sane",
]

ROOT_MODE_KWARGS = {
    "hybr": {
        "uses_jac": True,
        "valid_options": ["col_deriv", "xtol", "maxfev", "band", "eps", "factor", "diag"],
    },
    "lm": {
        "uses_jac": True,
        "valid_options": ["col_deriv", "ftol", "xtol", "gtol", "maxiter", "eps", "factor", "diag"],
    },
    "broyden1": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "max_rank",
            "jac_options",
        ],
        "jac_options": ["alpha", "reduction_method", "to_retain"],
    },
    "broyden2": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "max_rank",
            "jac_options",
        ],
        "jac_options": ["alpha", "reduction_method", "to_retain"],
    },
    "anderson": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "jac_options",
        ],
        "jac_options": ["alpha", "M", "w0"],
    },
    "linearmixing": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "jac_options",
        ],
        "jac_options": ["alpha"],
    },
    "diagbroyden": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "jac_options",
        ],
        "jac_options": ["alpha"],
    },
    "excitingmixing": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "jac_options",
        ],
        "jac_options": ["alpha", "alphamax"],
    },
    "krylov": {
        "uses_jac": False,
        "valid_options": [
            "nit",
            "disp",
            "maxiter",
            "ftol",
            "fatol",
            "xtol",
            "xatol",
            "tol_norm",
            "line_search",
            "jac_options",
        ],
        "jac_options": [
            "rdiff",
            "method",
            "inner_M",
            "inner_tol",
            "inner_maxiter",
            "outer_k",
        ],
    },
    "df-sane": {
        "uses_jac": False,
        "valid_options": [
            "ftol",
            "fatol",
            "fnorm",
            "maxfev",
            "disp",
            "eta_strategy",
            "sigma_eps",
            "sigma_0",
            "M",
            "line_search",
        ],
    },
}
