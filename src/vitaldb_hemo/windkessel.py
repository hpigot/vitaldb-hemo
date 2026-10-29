"""2-, 3- and parallel 4-element Windkessel models from aortic flow to aortic pressure.

Ported from the Julia package WindkesselIdentification.jl 2.0. Units throughout: pressure in
mmHg, flow in mL/s, time in s, so that Rp, Rc [mmHg s/mL], C [mL/mmHg], L [mmHg s^2/mL].
theta = [Rp, C, Rc, L]; wk3 uses the first 3 entries and wk2 the first 2.

Each model is continuous-time and diagonal, x' = a x + b u, y = c x + d u, returned as the
arrays (a, b, c, d). State x1 is the volume on C [mL]; in wk4, x2 = L times the flow through L
[mmHg s]. Discretised with a zero-order hold (flow constant over each sampling period h), each
state is a first-order recursion x[k+1] = ad x[k] + bd u[k], which scipy.signal.lfilter runs
in C. This module depends on numpy and scipy only.
"""

import numpy as np
from scipy import signal


def wk4(theta):
    """Parallel 4-element Windkessel (Burattini, Westerhof, Stergiopulos), theta = [Rp, C, Rc, L]."""
    rp, c, rc, ll = theta[:4]
    return (
        np.array([-1 / (c * rp), -rc / ll]),
        np.array([1.0, rc]),
        np.array([1 / c, -rc / ll]),
        rc,
    )


def wk3(theta):
    """3-element Windkessel, theta = [Rp, C, Rc]: wk4 with L = inf."""
    rp, c, rc = theta[:3]
    return np.array([-1 / (c * rp)]), np.array([1.0]), np.array([1 / c]), rc


def wk2(theta):
    """2-element Windkessel, theta = [Rp, C]: wk4 with Rc = 0 (L then has no effect)."""
    rp, c = theta[:2]
    return np.array([-1 / (c * rp)]), np.array([1.0]), np.array([1 / c]), 0.0


def zoh(m, h):
    """Exact zero-order-hold discretisation of a diagonal model m = (a, b, c, d): (ad, bd, c, d)."""
    a, b, c, d = m
    ad = np.exp(a * h)
    return ad, (ad - 1) / a * b, c, d


def statespace(theta, h=None, model=wk4):
    """The model as a scipy.signal.StateSpace: continuous if h is None, else ZOH-discretised.

    For analysis (scipy.signal.bode, freqresp, ...); simulate() is faster for fitting.
    """
    a, b, c, d = zoh(model(theta), h) if h else model(theta)
    m = np.diag(a), b[:, None], c[None, :], np.array([[d]])
    return signal.StateSpace(*m, dt=h) if h else signal.StateSpace(*m)


def simulate(theta, x0, u, h, model=wk4, states=False):
    """Pressure [mmHg] from flow u [mL/s] sampled every h [s], from initial state x0.

    With states=True also returns X, len(u) x n: row k is the state at sample k, before u[k].
    """
    ad, bd, c, d = zoh(model(theta), h)
    # x[k+1] = ad x[k] + bd u[k] as the filter bd z^-1 / (1 - ad z^-1); zi = x0 makes x[0] = x0
    x = np.array(
        [signal.lfilter([0, bi], [1, -ai], u, zi=[xi])[0] for ai, bi, xi in zip(ad, bd, x0)]
    )
    y = c @ x + d * np.asarray(u)
    return (y, x.T) if states else y


def solvex0(theta, u, h, model=wk4):
    """Initial state for which the response to u is periodic, x(n) = x(0) with n = len(u).

    With s the state reached from x(0) = 0 after n steps, x(0) = s / (1 - ad^n) per state.
    Use it when u is one or more whole beats.
    """
    ad, bd, _, _ = zoh(model(theta), h)
    s = [signal.lfilter([0, bi], [1, -ai], u, zi=[0.0])[1][0] for ai, bi in zip(ad, bd)]
    return np.array(s) / (1 - ad ** len(u))
