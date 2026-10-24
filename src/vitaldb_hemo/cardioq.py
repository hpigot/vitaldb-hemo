"""The CardioQ case set: aortic flow (oesophageal Doppler) and arterial pressure, cleaned.

Pressure comes from CardioQ/ABP, the arterial-line signal as relayed through the CardioQ
monitor, because it shares the monitor's clock with CardioQ/FLOW: flow leads it by a
steady 50-70 ms of pulse transit. SNUADC/ART is the same pressure at 500 Hz, but its offset
to the CardioQ clock wanders by several hundred ms within minutes, so do not pair it with
FLOW.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import beats, data

TRACKS = ["CardioQ/FLOW", "CardioQ/SV", "SNUADC/ART"]

# Cases with the tracks that still cannot be used (checked 2026-10-24)
EXCLUDED = {
    1018: "CardioQ/ABP is all zero: the arterial line was not connected to the monitor",
    3311: "CardioQ/ABP is all zero: the arterial line was not connected to the monitor",
    3980: "device SV and HR invalid (HR mostly 0, SV up to 720 mL)",
}

# Frozen for tuning methods and for examples; evaluate on the other cases. Chosen for many
# usable hours, a steady flow-to-pressure delay and gain, and a spread of HR, SV and MAP.
DEVELOPMENT = [19, 468, 963, 1602]


@dataclass
class Case:
    caseid: int
    fs: float  # Hz, both signals
    p: np.ndarray  # arterial pressure [mmHg], CardioQ/ABP; NaN in dropouts
    q: np.ndarray  # aortic flow [mL/s]; NaN in dropouts and where FLOW saturates
    gain: float  # mL/s per CardioQ/FLOW unit
    windows: pd.DataFrame  # see windows()

    def segment(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, p, q) from t0 to t1 seconds."""
        a, b = round(t0 * self.fs), round(t1 * self.fs)
        return np.arange(a, b) / self.fs, self.p[a:b], self.q[a:b]


def cases() -> list[int]:
    """The 24 usable cases with CardioQ flow, stroke volume and an arterial line."""
    return [c for c in data.find_cases(TRACKS) if c not in EXCLUDED]


def load(caseid: int, length: float = 10.0) -> Case:
    """Load a case: pressure, flow scaled to mL/s, and its table of windows."""
    p, fs = data.load_wave(caseid, "CardioQ/ABP")
    v, _ = data.load_wave(caseid, "CardioQ/FLOW")
    n = min(len(p), len(v))
    p, v = p[:n], v[:n]
    w, gain = windows(
        p,
        v,
        fs,
        data.load_numeric(caseid, "CardioQ/SV"),
        data.load_numeric(caseid, "CardioQ/HR"),
        length,
    )
    q = gain * np.where(_saturated(v), np.nan, v)
    return Case(caseid, fs, p, q, gain, w)


def _saturated(v):
    return v >= 0.95 * np.nanmax(v)


def windows(
    p: np.ndarray,
    v: np.ndarray,
    fs: float,
    sv: tuple[np.ndarray, np.ndarray],
    hr: tuple[np.ndarray, np.ndarray],
    length: float = 10.0,
    tolerance: float = 0.2,
) -> tuple[pd.DataFrame, float]:
    """Split a case into windows of length seconds and judge each one.

    p: pressure [mmHg] and v: raw CardioQ/FLOW, both at fs on the same clock.
    sv, hr: (t, values) of the device's CardioQ/SV [mL] and CardioQ/HR [1/min].
    Returns the table and the case gain [mL/s per FLOW unit]: the median over usable
    windows of device flow (sv * hr / 60) over mean FLOW. Scaled this way, each beat's
    volume matches the device's SV, which is its nomogram estimate of total stroke
    volume, not of descending-aortic flow alone.

    Columns: t0, t1 [s]; beats (pressure onsets); map [mmHg]; hr, sv: device means;
    delay [s]: median lag from flow onset to pressure onset (about +-20 ms, as onsets are
    placed on a 16 Hz low-pass); ratio: window gain over case
    gain; ok; reason: the first check that failed, in this order:
    missing (NaN in p or v), pressure (data.artefact_flags on p), saturated (FLOW at its
    ceiling), device (no valid device SV or HR), beats (a pressure onset
    without a flow onset 0-250 ms before it; extra flow onsets from Doppler spikes in
    diastole are allowed), gain (ratio off by more than tolerance: probe out of focus).
    """
    k = int(length * fs)
    starts = np.arange(0, len(p) - k + 1, k)
    bad_p = np.any(list(data.artefact_flags(p, fs).values()), axis=0)
    sat = _saturated(v)
    on_p = beats.onsets(p, fs)
    on_v = beats.onsets(v, fs)
    max_delay = 0.25  # s, longest plausible flow-to-pressure transit
    rows = []
    for a in starts:
        b = a + k
        t0, t1 = a / fs, b / fs
        ps, vs = p[a:b], v[a:b]
        op = on_p[(on_p >= a) & (on_p < b)]
        fv = on_v[np.maximum(np.searchsorted(on_v, op) - 1, 0)]  # flow onset of each beat
        lag = (op - fv) / fs
        paired = len(op) > 1 and ((lag > 0) & (lag <= max_delay)).all()
        m_sv = (sv[0] >= t0) & (sv[0] < t1)
        m_hr = (hr[0] >= t0) & (hr[0] < t1)
        dev_sv = sv[1][m_sv].mean() if m_sv.any() else np.nan
        dev_hr = hr[1][m_hr].mean() if m_hr.any() else np.nan
        if np.isnan(ps).any() or np.isnan(vs).any():
            reason = "missing"
        elif bad_p[a:b].any():
            reason = "pressure"
        elif sat[a:b].any():
            reason = "saturated"
        elif not (m_sv.any() and m_hr.any()) or (sv[1][m_sv] <= 0).any() or dev_hr <= 0:
            reason = "device"
        elif not paired:
            reason = "beats"
        else:
            reason = ""
        rows.append(
            {
                "t0": t0,
                "t1": t1,
                "beats": len(op),
                "map": np.nanmean(ps) if not np.isnan(ps).all() else np.nan,
                "hr": dev_hr,
                "sv": dev_sv,
                "delay": np.median(lag) if len(lag) else np.nan,
                # whole beats only, so a window that cuts a beat does not bias the mean
                "flow_gain": dev_sv * dev_hr / 60 / np.mean(v[fv[0] : fv[-1]])
                if reason == ""
                else np.nan,
                "reason": reason,
            }
        )
    w = pd.DataFrame(rows)
    gain = float(np.nanmedian(w.flow_gain))
    w["ratio"] = w.pop("flow_gain") / gain
    w.loc[(w.reason == "") & (np.abs(w.ratio - 1) > tolerance), "reason"] = "gain"
    w["ok"] = w.reason == ""
    return w, gain
