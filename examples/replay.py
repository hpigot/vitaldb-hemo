"""Replay a case: the arterial pressure as it arrives, and a stroke volume estimate next to the CardioQ.

    uv run python examples/replay.py 19                    # the Liljestrand-Zander baseline
    uv run python examples/replay.py 19 my_estimate.csv    # your estimate: columns t0 [s], est [mL]
    uv run python examples/replay.py 19 --speed 120 --save replay.gif --minutes 30

speed is seconds of case per second of replay. With an estimate file, est is matched to the
usable windows by t0; windows without a value are left blank.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from vitaldb_hemo import cardioq, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("case", type=int)
ap.add_argument("estimate", nargs="?", help="CSV with columns t0, est")
ap.add_argument("--speed", type=float, default=60.0)
ap.add_argument("--save", help="write an animated GIF instead of showing a window")
ap.add_argument("--minutes", type=float, help="replay only this many minutes of case time")
a = ap.parse_args()

c = cardioq.load(a.case)
w = c.windows.query("ok").assign(case=a.case).reset_index(drop=True)
if a.estimate:
    w = w.merge(pd.read_csv(a.estimate)[["t0", "est"]], on="t0", how="left")
    label = a.estimate
else:
    lz = w.pp / (w.sbp + w.dbp)
    w["est"] = evaluate.calibrated(w, lz).per_case.k.iloc[0] * lz
    label = "Liljestrand-Zander, calibrated on the first 15 min"

dt = 0.05  # s between frames
t_start, t_end = w.t0.iloc[0] + 10, w.t1.iloc[-1]
if a.minutes:
    t_end = min(t_end, t_start + 60 * a.minutes)
times = np.arange(t_start, t_end, a.speed * dt)

fig, (ax_p, ax_sv) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 1.4]})
(line_p,) = ax_p.plot([], [], color="C3", lw=1)
ax_p.set(xlim=(-10, 0), ylim=(np.nanpercentile(c.p, 0.5) - 5, np.nanpercentile(c.p, 99.5) + 5))
ax_p.set(xlabel="seconds before now", ylabel="arterial pressure [mmHg]")
h = (w.t0 - t_start) / 3600
ax_sv.set(xlim=(0, (t_end - t_start) / 3600), ylim=(0, 1.3 * np.nanmax([w.sv.max(), w.est.max()])))
ax_sv.set(xlabel="hours", ylabel="stroke volume [mL]")
(dev,) = ax_sv.plot([], [], ".", ms=2, color="0.4", label="CardioQ")
(est,) = ax_sv.plot([], [], ".", ms=2, color="C0", label=label)
now = ax_sv.axvline(0, color="k", lw=0.8)
ax_sv.legend(loc="upper left", fontsize=8, markerscale=4)
fig.tight_layout()


def frame(t):
    tt, p, _ = c.segment(t - 10, t)
    line_p.set_data(tt - t, p)
    seen = w.t1 <= t
    dev.set_data(h[seen], w.sv[seen])
    est.set_data(h[seen], w.est[seen])
    now.set_xdata([(t - t_start) / 3600] * 2)
    ax_p.set_title(f"case {a.case}, {(t - t_start) / 3600:.2f} h", fontsize=10)
    return line_p, dev, est, now


anim = FuncAnimation(fig, frame, frames=times, interval=1000 * dt, blit=False)
if a.save:
    anim.save(a.save, writer=PillowWriter(fps=round(1 / dt)))
else:
    plt.show()
