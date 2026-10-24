"""Regenerate the figures in docs/img from VitalDB data (downloads on first run).

uv run python docs/make_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vitaldb_hemo import data

OUT = Path(__file__).parent / "img"
CREDIT = "Data: VitalDB (Lee et al. 2022), CC BY 4.0"
COLORS = {"missing": "0.6", "range": "C3", "zero": "C2", "flat": "C1"}


def credit(fig):
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color="0.4")


def devices():
    idx = data.track_index()
    rows = [
        ("CardioQ/FLOW + SV + SNUADC/ART", ["CardioQ/FLOW", "CardioQ/SV", "SNUADC/ART"]),
        ("CardioQ/FLOW", ["CardioQ/FLOW"]),
        ("Vigilance/CO", ["Vigilance/CO"]),
        ("Vigileo/SV", ["Vigileo/SV"]),
        ("EV1000/SV", ["EV1000/SV"]),
        ("SNUADC/ART", ["SNUADC/ART"]),
    ]
    n = [len(data.find_cases(tracks, idx)) for _, tracks in rows]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh([r[0] for r in rows], n, color=["C0"] + ["0.65"] * (len(rows) - 1))
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(f"cases (of {idx.caseid.nunique()})")
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    devices()
    art, fs, flags = case_overview(19)
    zoom(19, art, fs, flags, 1450, 45, "line zeroing and flush", "flush")
    zoom(19, art, fs, flags, 4600, 45, "NIBP cuff inflated on the arterial-line arm", "cuff")
    case_overview(1602)
