# 4. Units and Windkessel parametrisation

Date: 2026-10-29

## Context

The Windkessel models and the fitter are ported from the Julia package WindkesselIdentification.jl, and the port is tested against its reference outputs. Its version 1 used flow in L/min, which still appears in some of its raw data files.

## Decision

- Pressure in mmHg, flow in mL/s, time in s throughout, so that `Rp` and `Rc` are in mmHg s/mL, `C` in mL/mmHg and `L` in mmHg s^2/mL.
- `theta = [Rp, C, Rc, L]` in that order for every model; the 3- and 2-element models use the first 3 and 2 entries.
- Each model is diagonal: `x' = a x + b u`, `y = c x + d u`, with state x1 the volume on `C` [mL] and, in the 4-element model, x2 = `L` times the flow through `L`. It is discretised exactly with a zero-order hold, so each state is a first-order recursion.

## Consequences

- The Python and Julia packages agree to 1e-12 on simulation and to five digits on fits, and their parameter vectors can be compared directly.
- The initial state has a physical meaning (a volume), so a periodic or estimated initial state can be checked for plausibility.
- Fitting is done in the physical parameters with positivity bounds, not in logarithms; a fit in log parameters is a small change if the conditioning calls for it.
