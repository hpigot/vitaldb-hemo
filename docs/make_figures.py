"""Regenerate the figures in docs/img from VitalDB data (downloads on first run).

uv run python docs/make_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from vitaldb_hemo import beats, cardioq, data, evaluate, identify, windkessel

OUT = Path(__file__).parent / "img"
CREDIT = "Data: VitalDB (Lee et al. 2022), CC BY 4.0"
COLORS = {"missing": "0.6", "range": "C3", "zero": "C2", "flat": "C1"}


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color="0.4")


def devices():
    rows = [
        ("CardioQ/FLOW + SV + SNUADC/ART", ["CardioQ/FLOW", "CardioQ/SV", "SNUADC/ART"]),
        ("CardioQ/FLOW", ["CardioQ/FLOW"]),
        ("Vigilance/CO", ["Vigilance/CO"]),
        ("Vigileo/SV", ["Vigileo/SV"]),
        ("EV1000/SV", ["EV1000/SV"]),
        ("SNUADC/ART", ["SNUADC/ART"]),
    ]
    n = [len(data.find_cases(tracks)) for _, tracks in rows]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh([r[0] for r in rows], n, color=["C0"] + ["0.65"] * (len(rows) - 1))
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(f"cases (of {len(set().union(*data.tracks().values()))})")
    ax.set_title("VitalDB cases with the tracks: data.find_cases", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / "find_cases.png", dpi=120)


def envelope(x, fs, w=10.0):
    k = int(w * fs)
    b = x[: len(x) // k * k].reshape(-1, k)
    with warnings.catch_warnings(action="ignore"):
        return np.arange(len(b)) * w / 60, np.nanmin(b, 1), np.nanmax(b, 1)


def shade(ax, flags, fs, t0=0.0, scale=60):
    for name, m in flags.items():
        d = np.diff(np.r_[0, m.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(
                (t0 + a / fs) / scale, (t0 + b / fs) / scale, color=COLORS[name], alpha=0.3, lw=0
            )


def case_overview(caseid):
    art, fs = data.load_wave(caseid, "SNUADC/ART")
    flow, fs_f = data.load_wave(caseid, "CardioQ/FLOW")
    t_sv, sv = data.load_numeric(caseid, "CardioQ/SV")
    flags = data.artefact_flags(art, fs)
    print(f"case {caseid}: {len(art) / fs / 3600:.1f} h; flagged fraction of ART:")
    for name, m in flags.items():
        print(f"  {name:8s} {m.mean():6.1%}")
    print(f"  {'any':8s} {np.any(list(flags.values()), axis=0).mean():6.1%}")

    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    t, lo, hi = envelope(art, fs)
    ax[0].fill_between(t, lo, hi, color="C0", lw=0)
    ax[0].set_ylim(-60, 320)
    ax[0].set_ylabel("SNUADC/ART [mmHg]\n500 Hz", fontsize=8)
    shade(ax[0], flags, fs)
    for name, c in COLORS.items():
        ax[0].plot([], [], color=c, alpha=0.5, lw=6, label=name)
    ax[0].legend(title="artefact_flags", fontsize=7, title_fontsize=7, ncol=4, loc="upper left")
    t, lo, hi = envelope(flow, fs_f)
    ax[1].fill_between(t, lo, hi, color="C0", lw=0)
    ax[1].set_ylabel("CardioQ/FLOW [cm/s]\n180 Hz", fontsize=8)
    ax[2].plot(t_sv / 60, sv, ".", ms=1.5)
    ax[2].set_ylabel("CardioQ/SV [mL]\nper beat", fontsize=8)
    ax[2].set_xlabel("time from start of case file [min]")
    ax[0].set_title(f"Case {caseid}: 10 s min-max envelopes, ART artefacts shaded", fontsize=10)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / f"case{caseid}_overview.png", dpi=120)
    return art, fs, flags


def zoom(caseid, art, fs, flags, t0, w, what, slug):
    a, b = int(t0 * fs), int((t0 + w) * fs)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(a, b) / fs, art[a:b], lw=0.7, color="k")
    shade(ax, {k: m[a:b] for k, m in flags.items()}, fs, t0, scale=1)
    for name, c in COLORS.items():
        ax.plot([], [], color=c, alpha=0.5, lw=6, label=name)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_xlabel("time from start of case file [s]")
    ax.set_ylabel("SNUADC/ART [mmHg]")
    ax.set_title(f"Case {caseid}: {what}, as flagged by artefact_flags", fontsize=10)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / f"case{caseid}_{slug}.png", dpi=120)


def beat_features(caseid, t0=7200, w=6):
    art, fs = data.load_wave(caseid, "SNUADC/ART")
    p, fs = data.resample(art, fs, 125), 125.0
    bad = np.any(list(data.artefact_flags(p, fs).values()), axis=0)
    on = beats.onsets(p, fs)
    f = beats.features(p, fs, on, bad)
    hr_t, hr = data.load_numeric(caseid, "Solar8000/HR")
    print(f"case {caseid}: {len(f)} beats, {f.ok.mean():.1%} ok")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    a, b = int(t0 * fs), int((t0 + w) * fs)
    ax[0].plot(np.arange(a, b) / fs, p[a:b], "k", lw=0.8)
    g = f[(f.t >= t0) & (f.t < t0 + w - 1.5)]
    o = (g.t * fs).round().astype(int)
    ax[0].plot(g.t, p[o], "v", color="C2", label="onset")
    ax[0].hlines(g.sbp, g.t, g.t + 60 / g.hr, color="C3", lw=1, label="sbp")
    ax[0].hlines(g["map"], g.t, g.t + 60 / g.hr, color="C0", lw=1, label="map")
    ax[0].hlines(g.dbp, g.t, g.t + 60 / g.hr, color="C1", lw=1, label="dbp")
    ax[0].legend(fontsize=7, ncol=4, loc="center right")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("SNUADC/ART [mmHg]")
    ax[0].set_title("beats.onsets and beats.features", fontsize=10)
    ok = f[f.ok]
    ax[1].plot(ok.t / 60, ok["map"], ".", ms=1, color="C0", label="map, ok beats")
    ax[1].plot(ok.t / 60, ok.hr, ".", ms=1, color="C2", label="hr from onsets")
    ax[1].plot(hr_t / 60, hr, "-", lw=0.6, color="k", label="Solar8000/HR (monitor)")
    ax[1].set_ylim(30, 150)
    ax[1].set_xlabel("time from start of case file [min]")
    ax[1].set_ylabel("mmHg, 1/min")
    ax[1].legend(fontsize=7, markerscale=6, loc="upper right")
    ax[1].set_title(f"Case {caseid}: {ok.shape[0]} ok beats of {len(f)}", fontsize=10)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / f"case{caseid}_beats.png", dpi=120)


REASONS = {
    "": ("ok", "C2"),
    "missing": ("missing", "0.6"),
    "pressure": ("pressure artefact", "C3"),
    "saturated": ("flow saturated", "C1"),
    "device": ("no device SV", "C4"),
    "beats": ("unpaired beats", "C5"),
    "gain": ("flow off gain", "C9"),
}


def cardioq_windows(caseid, t0):
    c = cardioq.load(caseid)
    w = c.windows
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [1, 1.6]})
    for i, (r, (label, color)) in enumerate(REASONS.items()):
        m = w.reason == r
        ax[0].broken_barh(
            list(zip(w.t0[m] / 60, (w.t1 - w.t0)[m] / 60)), (-i - 0.4, 0.8), color=color
        )
    ax[0].set_yticks(
        -np.arange(len(REASONS)),
        [f"{lab} ({(w.reason == r).sum()})" for r, (lab, _) in REASONS.items()],
        fontsize=7,
    )
    ax[0].set_xlabel("time from start of case file [min]")
    ax[0].set_title(
        f"Case {caseid}: cardioq.load windows, {w.ok.sum() * 10 / 3600:.1f} h ok of "
        f"{len(w) * 10 / 3600:.1f} h",
        fontsize=10,
    )
    t, p, q = c.segment(t0, t0 + 5)
    ax[1].plot(t, p, "C3", lw=1)
    ax[1].set_ylabel("p: CardioQ/ABP [mmHg]", color="C3")
    ax[1].set_xlabel("time [s]")
    b = ax[1].twinx()
    b.plot(t, q, "C0", lw=1)
    b.set_ylabel(f"q: flow [mL/s] = {c.gain:.2f} x CardioQ/FLOW", color="C0")
    r = w[(w.t0 <= t0) & (w.t1 > t0)].iloc[0]
    ax[1].set_title(
        f"One ok window: device SV {r.sv:.0f} mL, HR {r.hr:.0f}/min, flow leads pressure by "
        f"{r.delay * 1000:.0f} ms",
        fontsize=10,
    )
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / f"case{caseid}_windows.png", dpi=120)


def case_set():
    ids = cardioq.cases()
    hours = {cid: cardioq.load(cid).windows.reason.value_counts() * 10 / 3600 for cid in ids}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, cid in enumerate(ids):
        left = 0.0
        for r, (_, color) in REASONS.items():
            h = hours[cid].get(r, 0.0)
            ax.barh(i, h, left=left, color=color, height=0.8)
            left += h
    ax.set_yticks(
        range(len(ids)),
        [f"{c} (dev)" if c in cardioq.DEVELOPMENT else str(c) for c in ids],
        fontsize=8,
    )
    for lab in ax.get_yticklabels():
        if "dev" in lab.get_text():
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("hours")
    ax.legend(
        handles=[Patch(color=c, label=lab) for lab, c in REASONS.values()],
        fontsize=7,
        loc="lower right",
    )
    total = sum(h.get("", 0.0) for h in hours.values())
    ax.set_title(
        f"cardioq.cases(): {len(ids)} cases, {total:.0f} h of ok windows; bold: DEVELOPMENT",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / "cardioq_cases.png", dpi=120)


def development_windows():
    """One row per usable window of the development cases: device SV, pulse pressure, 3-element fit."""
    rows = []
    for cid in cardioq.DEVELOPMENT:
        c = cardioq.load(cid)
        f = beats.features(c.p, c.fs, beats.onsets(c.p, c.fs))
        pp = f[f.ok].groupby(f.t // 10 * 10).pp.mean()
        for r in c.windows[c.windows.ok].itertuples():
            _, p, q = c.segment(r.t0, r.t1)
            k = round(r.delay * c.fs)
            fit = identify.identify(
                [1.0, 1.5, 0.05], q[: len(q) - k], p[k:], 1 / c.fs, windkessel.wk3
            )
            rows.append((cid, r.t0, r.sv, pp.get(r.t0, np.nan), *fit.theta))
    return pd.DataFrame(rows, columns=["case", "t0", "sv", "pp", "rp", "c", "rc"])


def fits_over_time(d):
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for i, (cid, g) in enumerate(d.groupby("case")):
        m = g.set_index("t0")[["rp", "c", "rc"]].rolling(60, center=True, min_periods=20).median()
        for a, col in zip(ax, ["rp", "c", "rc"]):
            a.plot(g.t0 / 3600, g[col], ".", ms=1, color=f"C{i}", alpha=0.25)
            a.plot(m.index / 3600, m[col], color=f"C{i}", lw=1.5, label=f"case {cid}")
    for a, lab, top in zip(ax, ["Rp [mmHg s/mL]", "C [mL/mmHg]", "Rc [mmHg s/mL]"], [2.5, 10, 0.3]):
        a.set_ylabel(lab, fontsize=8)
        a.set_ylim(0, top)
    ax[0].legend(fontsize=7, ncol=4, loc="upper right")
    ax[2].set_xlabel("time from start of case file [h]")
    ax[0].set_title(
        "3-element fits on every usable window of the development cases (dots), 10 min medians (lines)",
        fontsize=10,
    )
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / "development_fits.png", dpi=120)


def baseline_agreement(d):
    """Pulse-pressure baseline: SV = k PP, k from the first 5 usable minutes, 1 min means."""
    parts = []
    for cid, g in d.dropna().sort_values("t0").groupby("case"):
        k = (g.sv / g.pp).iloc[:30].mean()
        g = g.iloc[30:].assign(est=k * g.pp.iloc[30:], minute=g.t0.iloc[30:] // 60)
        parts.append(g.groupby("minute")[["sv", "est"]].mean().assign(case=cid))
    e = pd.concat(parts)
    a = evaluate.agreement(e.est, e.sv, e.case)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (cid, g) in enumerate(e.groupby("case")):
        ax.plot((g.est + g.sv) / 2, g.est - g.sv, ".", ms=3, color=f"C{i}", label=f"case {cid}")
    for v, ls in [(a.bias, "-"), (a.loa[0], "--"), (a.loa[1], "--")]:
        ax.axhline(v, color="k", lw=1, ls=ls)
    ax.text(ax.get_xlim()[1], a.loa[1], " upper LoA", va="bottom", ha="right", fontsize=7)
    ax.text(ax.get_xlim()[1], a.loa[0], " lower LoA", va="top", ha="right", fontsize=7)
    ax.set_xlabel("mean of estimate and device SV [mL]")
    ax.set_ylabel("estimate - device SV [mL]")
    ax.set_title(
        f"Pulse-pressure baseline: bias {a.bias:+.0f} mL, limits {a.loa[0]:.0f} to {a.loa[1]:+.0f} mL, "
        f"percentage error {a.percentage_error:.0f} %",
        fontsize=9,
    )
    ax.legend(fontsize=7, markerscale=2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    credit(fig)
    fig.tight_layout()
    fig.savefig(OUT / "development_baseline.png", dpi=120)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    devices()
    art, fs, flags = case_overview(19)
    zoom(19, art, fs, flags, 1450, 45, "line zeroing and flush", "flush")
    zoom(19, art, fs, flags, 4600, 45, "NIBP cuff inflated on the arterial-line arm", "cuff")
    case_overview(1602)
    beat_features(19)
    cardioq_windows(19, 7200)
    case_set()
    d = development_windows()
    fits_over_time(d)
    baseline_agreement(d)
