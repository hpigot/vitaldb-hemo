"""Agreement between an estimate and a reference, e.g. stroke volume against CardioQ/SV.

Bland JM, Altman DG. Agreement between methods of measurement with multiple observations
per individual. J Biopharm Stat 2007;17:571-582.
Critchley LA, Critchley JA. A meta-analysis of studies using bias and precision statistics
to compare cardiac output measurement techniques. J Clin Monit Comput 1999;15:85-91.
Critchley LA, Lee A, Ho AM. A critical review of the ability of continuous cardiac output
monitors to measure trends in cardiac output. Anesth Analg 2010;111:1180-1192.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Agreement:
    bias: float  # mean of est - ref
    sd: float  # standard deviation of est - ref (between and within subjects)
    loa: tuple[float, float]  # 95 % limits of agreement, bias -+ 1.96 sd
    percentage_error: float  # 1.96 sd / mean(ref), in %; under 30 % is the usual criterion
    n: int  # number of pairs


def agreement(est, ref, subject=None) -> Agreement:
    """Bland-Altman bias, limits of agreement and percentage error of est against ref.

    subject: one label per pair (e.g. the caseid). With several pairs per subject, the
    differences within a subject are correlated, and the plain standard deviation would
    make the limits too narrow; the sd is then the sum of the between- and
    within-subject variances (Bland and Altman 2007, true value varying).
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    d = est - ref
    sd = float(np.std(d, ddof=1))
    if subject is not None:
        _, g, m = np.unique(subject, return_inverse=True, return_counts=True)
        n, big_n = len(m), len(d)
        if 1 < n < big_n:  # one subject, or one pair each: the plain sd is right
            means = np.bincount(g, d) / m
            msb = np.sum(m * (means - d.mean()) ** 2) / (n - 1)
            msw = np.sum((d - means[g]) ** 2) / (big_n - n)
            lam = (big_n**2 - np.sum(m**2)) / ((n - 1) * big_n)
            sd = float(np.sqrt(max((msb - msw) / lam, 0.0) + msw))
    bias = float(d.mean())
    pe = float(100 * 1.96 * sd / ref.mean())
    return Agreement(bias, sd, (bias - 1.96 * sd, bias + 1.96 * sd), pe, len(d))


def concordance(est, ref, subject=None, exclusion=0.15) -> tuple[float, int]:
    """Four-quadrant concordance: how often est and ref change in the same direction.

    Changes are relative, from each value to the next of the same subject. Pairs where
    both changes are smaller than exclusion (default 15 %) are left out, as their direction
    is mostly noise. Returns (rate, number of pairs used); above 0.9 is the usual criterion.
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    de, dr = np.diff(est) / est[:-1], np.diff(ref) / ref[:-1]
    keep = np.maximum(np.abs(de), np.abs(dr)) >= exclusion
    if subject is not None:
        s = np.asarray(subject)
        keep &= s[1:] == s[:-1]
    n = int(keep.sum())
    rate = float(np.mean(np.sign(de[keep]) == np.sign(dr[keep]))) if n else np.nan
    return rate, n


@dataclass
class Calibrated:
    per_case: pd.DataFrame  # case, k, bias, percentage_error, concordance, pairs, minutes
    agreement: Agreement  # pooled over cases, repeated measures
    concordance: tuple[float, int]  # pooled over cases: (rate, pairs)
    minutes: pd.DataFrame  # case, t [s], hours since calibration, sv, est, map, dmap, error


def calibrated(table: pd.DataFrame, proxy, minutes: float = 15.0, step: int = 5) -> Calibrated:
    """Calibrate SV = k proxy on the first minutes of usable windows of each case; evaluate the rest.

    table: usable windows with columns case, t0, t1, sv (device) and map, as cardioq.table gives.
    proxy: a stroke volume proxy, a quantity that SV is assumed proportional to, as a column
    name of table or an array with one value per row (NaN where the estimator has none, e.g.
    while a filter warms up). Per case, k is the median of sv / proxy over the
    windows that make up the first minutes of usable time; those windows are left out of the
    evaluation. The rest is averaged over whole minutes before comparing, and the concordance
    uses changes over step minutes. In minutes: dmap = ln(MAP / median MAP of the calibration)
    and error = ln(est / sv), for plotting the error against the change in MAP or the time.
    """
    d = table.assign(proxy=table[proxy] if isinstance(proxy, str) else np.asarray(proxy, float))
    rows, parts, conc = [], [], []
    for case, g in d.sort_values(["case", "t0"]).groupby("case", sort=False):
        cal = (g.t1 - g.t0).cumsum() <= 60 * minutes
        k = float(np.nanmedian(g.sv[cal] / g.proxy[cal]))
        e = g[~cal].assign(est=k * g.proxy[~cal], m=g.t0[~cal] // 60)
        m = e.groupby("m")[["t0", "sv", "est", "map"]].mean().dropna()
        if len(m) < 2:
            continue
        m = m.assign(
            case=case,
            hours=(m.t0 - g.t1[cal].max()) / 3600,
            dmap=np.log(m["map"] / g["map"][cal].median()),
            error=np.log(m.est / m.sv),
        ).rename(columns={"t0": "t"})
        a = agreement(m.est, m.sv)
        c = concordance(m.est.values[::step], m.sv.values[::step], exclusion=0.15)
        conc.append(m.iloc[::step])
        rows.append((case, k, a.bias, a.percentage_error, c[0], c[1], len(m)))
        parts.append(m)
    per_case = pd.DataFrame(
        rows, columns=["case", "k", "bias", "percentage_error", "concordance", "pairs", "minutes"]
    )
    mm = pd.concat(parts, ignore_index=True)
    cc = pd.concat(conc, ignore_index=True)
    return Calibrated(
        per_case,
        agreement(mm.est, mm.sv, mm.case),
        concordance(cc.est, cc.sv, cc.case),
        mm[["case", "t", "hours", "sv", "est", "map", "dmap", "error"]],
    )
