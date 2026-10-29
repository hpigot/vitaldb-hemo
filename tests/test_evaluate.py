import numpy as np

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
