"""Beat onsets in arterial pressure (Zong et al. 2003) and per-beat features."""

import numpy as np
import pandas as pd
from scipy import signal


def _lowpass(p: np.ndarray, fs: float) -> np.ndarray:
    b, a = signal.butter(2, min(16 / (fs / 2), 0.99))
    return signal.filtfilt(b, a, np.nan_to_num(p))


def onsets(p: np.ndarray, fs: float, refractory: float = 0.25) -> np.ndarray:
    """Sample indices of pulse onsets in arterial pressure p [mmHg] at fs [Hz].

    Zong W, Heldt T, Moody GB, Mark RG. An open-source algorithm to detect onset of
    arterial blood pressure pulses. Computers in Cardiology 2003;30:259-262.
    Low-pass filter at 16 Hz, slope sum function (SSF: sum of upward slopes over 128 ms),
    threshold at 60 % of an adaptive pulse height. The onset is the last sample within 2 %
    of the upstroke height above the minimum in the 150 ms before each threshold crossing;
    the paper instead searches back to where the SSF starts rising, which noise in diastole
    drags up to 100 ms early. Works for flow too, whose diastole is flat.
    NaNs are treated as 0, so beats in artefacts are detected too: drop them with the ok
    column of features().
    """
    y = _lowpass(p, fs)
    du = np.maximum(np.diff(y, prepend=y[0]), 0)
    ssf = np.convolve(du, np.ones(int(0.128 * fs)))[: len(y)]
    height = 3 * ssf[: int(10 * fs)].mean()  # paper: 3 x mean SSF of the first 10 s
    look, chunk, gap = int(0.15 * fs), int(2 * fs), int(refractory * fs)
    out, i = [], 0
    while i < len(y) - look:
        hits = np.flatnonzero(ssf[i : i + chunk] > 0.6 * height)
        if not len(hits):
            height /= 2  # no beat for 2 s (pause, artefact): lower the threshold
            i += chunk
            continue
        j = i + hits[0]
        top = ssf[j : j + look].max()
        seg = y[max(j - look, 0) : j + 1]
        low = seg.min() + 0.02 * (seg[-1] - seg.min())  # flat diastole (flow): take its end
        out.append(j - int(np.argmax(seg[::-1] <= low)))
        height += (top - height) / 4  # paper updates from each pulse; averaged for spikes
        i = j + gap
    return np.array(out, dtype=int)


def features(
    p: np.ndarray, fs: float, on: np.ndarray, bad: np.ndarray | None = None
) -> pd.DataFrame:
    """One row per beat (onset to next onset) of arterial pressure p [mmHg] at fs [Hz].

    t: onset time [s]; sbp, dbp, map, pp [mmHg]; hr [1/min]; dpdt_max [mmHg/s], from the
    16 Hz low-pass signal. ok is False for beats with NaNs, a heart rate outside 30-200/min (missed or extra onsets), or any sample in bad
    (e.g. the union of data.artefact_flags).
    """
    a, b = on[:-1], on[1:]
    n = b - a
    pmax = np.maximum.reduceat(p, a)[: len(a)]
    pmin = np.minimum.reduceat(p, a)[: len(a)]

    def csum(x):  # area between onsets by differences of a running sum
        return np.concatenate([[0], np.cumsum(x)])

    c = csum(np.nan_to_num(p)) / fs
    mean = (c[b] - c[a]) * fs / n
    hr = 60 * fs / n
    dpdt = np.maximum.reduceat(np.gradient(_lowpass(p, fs)) * fs, a)[: len(a)]
    nan = csum(np.isnan(p))
    ok = (nan[b] == nan[a]) & (hr >= 30) & (hr <= 200)
    if bad is not None:
        cb = csum(bad)
        ok &= cb[b] == cb[a]
    return pd.DataFrame(
        {
            "t": a / fs,
            "sbp": pmax,
            "dbp": pmin,
            "map": mean,
            "pp": pmax - pmin,
            "hr": hr,
            "dpdt_max": dpdt,
            "ok": ok,
        }
    )
