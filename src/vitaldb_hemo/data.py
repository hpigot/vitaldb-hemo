"""Find, download and load VitalDB cases; resample waveforms; flag arterial-pressure artefacts.

Times are seconds from the start of the case file, which all tracks of a case share.
"""

import http.client
import os
import shutil
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb
from scipy import ndimage, signal

API = "https://api.vitaldb.net"
DATASET_VERSION = "1.0.1"  # pinned so frozen case lists stay reproducible
CACHE = Path(os.environ.get("VITALDB_HEMO_CACHE", Path.home() / ".cache" / "vitaldb-hemo"))


def _download(url: str, path: Path, tries: int = 30) -> None:
    """Fetch url to path, resuming dropped transfers with HTTP Range requests.

    VitalDB's server often drops long transfers part-way, and vitaldb reads a truncated
    file without complaint, so path only appears once the full length has arrived.
    """
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    total = -1
    for _ in range(tries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 206:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                else:  # range ignored: start over
                    have, total = 0, int(r.headers["Content-Length"])
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f)
        except urllib.error.HTTPError as e:
            if e.code != 416:
                raise
            part.unlink()  # stale part file from an interrupted run
            continue
        except (OSError, http.client.HTTPException):
            pass
        if part.exists() and part.stat().st_size == total:
            part.replace(path)
            return
    raise OSError(f"download of {url} failed after {tries} attempts")


def track_index() -> pd.DataFrame:
    """All tracks of all cases (columns caseid, tname, tid), cached in CACHE."""
    path = CACHE / DATASET_VERSION / "trks.csv.gz"
    if not path.exists():
        _download(f"{API}/trks", path)
    return pd.read_csv(path, compression="gzip")


def find_cases(tracks: list[str], index: pd.DataFrame | None = None) -> list[int]:
    """Case ids that have every track in tracks (exact names such as "CardioQ/FLOW")."""
    idx = track_index() if index is None else index
    return sorted(set.intersection(*(set(idx.caseid[idx.tname == t]) for t in tracks)))


def case_file(caseid: int) -> Path:
    """Local path of a case's .vital file, downloaded on first use."""
    path = CACHE / DATASET_VERSION / f"{caseid}.vital"
    if not path.exists():
        _download(f"{API}/{DATASET_VERSION}/{caseid}.vital", path)
    return path


def _track(caseid: int, name: str):
    vf = vitaldb.VitalFile(str(case_file(caseid)), [name])
    trk = vf.find_track(name)
    if trk is None:
        raise KeyError(f"case {caseid} has no track {name}")
    return vf, trk


def load_wave(caseid: int, name: str) -> tuple[np.ndarray, float]:
    """Waveform track at its native rate: (x, fs), sample k at time k / fs. Gaps are NaN."""
    vf, trk = _track(caseid, name)
    if trk.type != 1:
        raise ValueError(f"{name} is not a waveform track; use load_numeric")
    return vf.get_track_samples(name, 1 / trk.srate).astype(float), float(trk.srate)


def load_numeric(caseid: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Numeric track (one value per beat or per update): (t, v) at irregular times."""
    vf, trk = _track(caseid, name)
    if trk.type != 2:
        raise ValueError(f"{name} is not a numeric track; use load_wave")
    t = np.array([r["dt"] for r in trk.recs]) - vf.dtstart
    v = np.array([r["val"] for r in trk.recs], dtype=float)
    return t, v


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Anti-aliased resampling (polyphase FIR); NaN gaps stay NaN.

    vitaldb's own resampling picks the nearest sample, which aliases when downsampling.
    """
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    nan = np.isnan(x)
    if nan.all():
        return np.full(-(-len(x) * ratio.numerator // ratio.denominator), np.nan)
    i = np.arange(len(x))
    # ponytail: gaps are bridged linearly before filtering, so a few samples next to a
    # gap carry filter ringing; widen the NaN mask if that ever matters
    y = signal.resample_poly(np.interp(i, i[~nan], x[~nan]), ratio.numerator, ratio.denominator)
    src = np.minimum(np.round(np.arange(len(y)) * fs_in / fs_out).astype(int), len(x) - 1)
    y[nan[src]] = np.nan
    return y


def artefact_flags(
    p: np.ndarray,
    fs: float,
    window: float = 2.0,
    min_std: float = 5.0,
    limits: tuple[float, float] = (0.0, 300.0),
    max_mean: float = 200.0,
    pad: float = 2.0,
) -> dict[str, np.ndarray]:
    """Boolean masks of artefacts in an arterial pressure waveform p [mmHg] at fs [Hz].

    missing: NaN samples.
    range:   samples outside limits, or a window mean above max_mean (junk values,
             transients when the line is opened, flush peaks).
    zero:    window mean within 10 mmHg of zero (transducer zeroing, disconnected line).
    flat:    no pulse: the standard deviation over the window, after 100 ms smoothing,
             is below min_std (flush plateaus, clipping, a fully damped line, and a
             blood-pressure cuff inflated on the same arm, which in VitalDB shows as about
             20 s of exponential decay before each NIBP reading); not zero.
    range, zero and flat are widened by pad seconds on each side to cover transitions.
    The defaults were set on cases 19 and 1602, where a normal pulse has a std of 7 mmHg
    or more.
    """
    k = max(int(window * fs), 1)
    q = np.nan_to_num(p)
    level = ndimage.uniform_filter1d(q, k)
    d = ndimage.uniform_filter1d(q, max(int(0.1 * fs), 1))  # suppresses mains hum and noise
    std = np.sqrt(np.maximum(ndimage.uniform_filter1d(d * d, k) - level**2, 0))
    zero = np.abs(level) < 10
    grow = np.ones(2 * int(pad * fs) + 1, bool)

    def widen(m):
        return ndimage.binary_dilation(m, grow) if pad > 0 else m

    return {
        "missing": np.isnan(p),
        "range": widen((q < limits[0]) | (q > limits[1]) | (level > max_mean)),
        "zero": widen(zero),
        "flat": widen((std < min_std) & ~zero),
    }
