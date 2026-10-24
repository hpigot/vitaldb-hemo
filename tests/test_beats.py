import numpy as np

from vitaldb_hemo import beats

FS = 125.0


def synthetic(rr, dbp=60.0, pp=40.0, noise=0.0, seed=0):
    """Pressure beats with onsets at known samples: a 0.1 s upstroke, then a decay."""
    on = np.round(np.concatenate([[0.5], 0.5 + np.cumsum(rr)[:-1]]) * FS).astype(int)
    n = on[-1] + int(rr[-1] * FS)
    p = np.empty(n)
    for k, (a, b) in enumerate(zip(on, np.r_[on[1:], n])):
        t = np.arange(b - a) / FS
        up = np.minimum(t / 0.1, 1)
        shape = (1 - np.cos(np.pi * up)) / 2 * np.exp(-np.maximum(t - 0.1, 0) / 0.6)
        p[a:b] = dbp + pp * shape
    p[: on[0]] = dbp + pp * np.exp(-1.5)
    return p + np.random.default_rng(seed).normal(0, noise, n), on


def test_onsets_on_clean_and_noisy_beats():
    rr = np.random.default_rng(1).uniform(0.6, 1.2, 60)
    for noise in [0.0, 2.0]:
        p, true = synthetic(rr, noise=noise)
        on = beats.onsets(p, FS)
        assert len(on) == len(true), noise
        assert np.abs(on - true).max() <= 0.02 * FS, noise  # within 20 ms


def test_missed_beat_gives_no_false_onset():
    rr = np.full(40, 0.8)
    rr[20] = 2.4  # a long pause, as after a dropped beat
    p, true = synthetic(rr, noise=1.0)
    on = beats.onsets(p, FS)
    assert len(on) == len(true) and np.abs(on - true).max() <= 0.02 * FS


def test_features():
    p, true = synthetic(np.full(10, 0.8))
    bad = np.zeros(len(p), bool)
    bad[true[4] + 10] = True
    f = beats.features(p, FS, true, bad)
    assert len(f) == 9
    assert np.allclose(f.hr, 75, atol=1) and np.allclose(f.dbp[1:], 60, atol=0.5)
    assert np.allclose(f.sbp, 100, atol=0.5) and np.allclose(f.pp[1:], 40, atol=1)
    assert f.ok.tolist() == [True] * 4 + [False] + [True] * 4
    assert (f.dpdt_max > 400).all() and (f["map"] > 60).all() and (f["map"] < 100).all()
