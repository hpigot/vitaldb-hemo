import numpy as np
import pandas as pd

from vitaldb_hemo import evaluate


def test_agreement_single_measurements():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    a = evaluate.agreement(ref + [0.1, 0.1, -0.1, 0.3], ref)
    sd = np.sqrt(0.08 / 3)  # differences 0.1 +- [0, 0, -0.2, 0.2]
    assert np.isclose(a.bias, 0.1) and np.isclose(a.sd, sd)
    assert np.allclose(a.loa, [0.1 - 1.96 * sd, 0.1 + 1.96 * sd])
    assert np.isclose(a.percentage_error, 100 * 1.96 * sd / 2.5)
    assert a.n == 4


def test_agreement_repeated_measurements():
    # each subject has a constant offset: no within-subject scatter, so the spread of the
    # differences is the spread of the subject means, sd sqrt(2), not the naive sqrt(1.2)
    ref = np.full(6, 50.0)
    est = ref + [1, 1, 1, -1, -1, -1]
    subject = [7, 7, 7, 9, 9, 9]
    assert np.isclose(evaluate.agreement(est, ref).sd, np.sqrt(1.2))
    a = evaluate.agreement(est, ref, subject)
    assert np.isclose(a.bias, 0) and np.isclose(a.sd, np.sqrt(2))
    # one measurement per subject: the same as ignoring subjects
    b = evaluate.agreement(est[[0, 3]], ref[[0, 3]], [7, 9])
    assert np.isclose(b.sd, evaluate.agreement(est[[0, 3]], ref[[0, 3]]).sd)


def test_concordance():
    ref = np.array([100, 120, 90, 91, 80, 100.0])  # changes +20 %, -25 %, +1 %, -12 %, +25 %
    est = np.array([100, 110, 115, 114, 95, 110.0])  # +10 %, +4.5 %, -0.9 %, -17 %, +16 %
    rate, n = evaluate.concordance(est, ref, exclusion=0.05)
    assert n == 4 and rate == 0.75  # +1 % vs -0.9 % is excluded; -25 % vs +4.5 % disagree
    # no change is taken from one subject to the next: 91 -> 80 is dropped
    rate, n = evaluate.concordance(est, ref, subject=[1, 1, 1, 1, 2, 2], exclusion=0.05)
    assert n == 3 and np.isclose(rate, 2 / 3)


def test_calibrated_scale_exclusion_and_pooling():
    # two cases of 2 h of 10 s windows; the drive is SV / 3 (case 1) and SV / 5 (case 2)
    t0 = np.arange(720) * 10.0
    rows = []
    for case, k in ((1, 3.0), (2, 5.0)):
        sv = 70 + 10 * np.sin(t0 / 900)
        rows.append(
            pd.DataFrame(
                {"case": case, "t0": t0, "t1": t0 + 10, "sv": sv, "map": 80.0, "x": sv / k}
            )
        )
    table = pd.concat(rows, ignore_index=True)
    r = evaluate.calibrated(table, "x")
    assert np.allclose(r.per_case.k, [3, 5]) and np.allclose(r.per_case.percentage_error, 0)
    assert r.minutes.hours.min() >= 0 and len(r.minutes) == 2 * (120 - 15)  # calibration left out
    # a drive that overestimates by 10 % after the calibration shows up as a bias of 7 mL
    table.loc[table.t0 >= 900, "x"] *= 1.1
    r = evaluate.calibrated(table, "x")
    assert np.allclose(r.minutes.error, np.log(1.1)) and abs(r.agreement.bias - 7) < 0.1
    assert r.agreement.n == len(r.minutes) and r.concordance[1] >= 0
