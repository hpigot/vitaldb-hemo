"""Output-error identification of the Windkessel models from aortic flow and pressure.

Ported from WindkesselIdentification.jl 2.0 (identify, identifyglobal), with scipy in place of
Optim: the local fit is Gauss-Newton with trust regions (scipy.optimize.least_squares) instead
of Newton with an exact Hessian, so it reaches the same minimum by a different path. Both fits
minimise the sum of squared errors between measured pressure y and the pressure simulated from
the measured flow u. This module depends on numpy and scipy only.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .windkessel import simulate, solvex0, wk4


@dataclass
class Fit:
    theta: np.ndarray  # all parameters, free and fixed, [Rp, C, Rc, L][:n]
    x0: np.ndarray  # initial state: estimated, periodic or as given
    mse: float  # mean squared pressure error [mmHg^2]
    result: optimize.OptimizeResult  # from scipy, for nfev, status, message, ...


def _problem(theta0, u, h, model, free, x0):
    """Initial optimisation vector p0, the map p -> (theta, x0), and the number of free theta.

    p holds the free theta entries in order, then x0 when x0 == "estimate".
    """
    theta0 = np.asarray(theta0, float)
    free = np.ones(len(theta0), bool) if free is None else np.asarray(free, bool)
    if len(free) != len(theta0):
        raise ValueError("free must have the same length as theta0")
    estimate = isinstance(x0, str) and x0 == "estimate"
    periodic = isinstance(x0, str) and x0 == "periodic"
    if isinstance(x0, str) and not (estimate or periodic):
        raise ValueError('x0 must be "estimate", "periodic" or an array')
    nf = int(free.sum())

    def unpack(p):
        theta = theta0.copy()
        theta[free] = p[:nf]
        if periodic:
            return theta, solvex0(theta, u, h, model)
        return theta, (p[nf:] if estimate else np.asarray(x0, float))

    p0 = theta0[free]
    if estimate:
        p0 = np.r_[p0, solvex0(theta0, u, h, model)]
    return p0, unpack, nf


def identify(theta0, u, y, h, model=wk4, free=None, x0="estimate", positive=True):
    """Fit model parameters to flow u [mL/s] and pressure y [mmHg] sampled every h [s].

    theta0: initial guess [Rp, C, Rc, L] (first 3 for wk3, first 2 for wk2); entries not in
        free stay fixed at these values.
    free: boolean mask over theta0, e.g. [1, 1, 1, 0] fixes L; default all free.
    x0: "estimate" fits the initial state too, starting from the periodic state of theta0;
        "periodic" recomputes it from periodicity at every step (u must be whole beats);
        an array fixes it.
    positive: keep the free theta entries >= 0 (x0 is never bounded).
    Local: converges to the minimum nearest theta0. See identify_global.
    """
    p0, unpack, nf = _problem(theta0, u, h, model, free, x0)
    y = np.asarray(y, float)

    def residuals(p):
        return simulate(*unpack(p), u, h, model) - y

    lower = np.r_[np.full(nf, 0.0 if positive else -np.inf), np.full(len(p0) - nf, -np.inf)]
    res = optimize.least_squares(
        residuals, p0, bounds=(lower, np.inf), x_scale="jac", ftol=1e-12, xtol=1e-12
    )
    return Fit(*unpack(res.x), float(np.mean(res.fun**2)), res)


def identify_global(theta0, u, y, h, model=wk4, free=None, x0="estimate", positive=True):
    """Same problem as identify(), solved with the derivative-free Nelder-Mead simplex.

    Less sensitive to a poor theta0, as its first simplex spans a large region and it needs
    no gradient, but much slower and with no guarantee of the global minimum. A good use is to
    start identify() from its result.
    """
    p0, unpack, nf = _problem(theta0, u, h, model, free, x0)
    y = np.asarray(y, float)

    def sse(p):
        with np.errstate(divide="ignore", invalid="ignore"):  # simplex vertices on theta = 0
            s = np.sum((simulate(*unpack(p), u, h, model) - y) ** 2)
        return s if np.isfinite(s) else np.inf

    bounds = [(0.0 if positive else None, None)] * nf + [(None, None)] * (len(p0) - nf)
    res = optimize.minimize(
        sse,
        p0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 100_000, "maxfev": 100_000, "xatol": 1e-10, "fatol": 1e-10},
    )
    return Fit(*unpack(res.x), float(res.fun / len(y)), res)
