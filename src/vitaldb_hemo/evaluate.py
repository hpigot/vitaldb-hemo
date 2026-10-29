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
