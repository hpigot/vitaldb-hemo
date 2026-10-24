import numpy as np

from vitaldb_hemo import cardioq

FS = 180.0


def test_windows_recover_gain_and_delay_and_reject_bad_windows():
    gain, delay, rr, n = 1.6, 0.06, 0.8, int(120 * FS)  # 12 windows of 10 s
    t = np.arange(n) / FS
    phase = (t - 0.3) % rr  # beats start at 0.3 s
    v = np.where(phase < 0.3, 200 * np.sin(np.pi * phase / 0.3), 0.0)  # FLOW, raw units
    tp = (t - 0.3 - delay) % rr  # pressure follows flow by delay
    p = 60 + 40 * np.where(tp < 0.1, np.sin(np.pi * tp / 0.2), np.exp(-(tp - 0.1) / 0.5))

    sv_beat = gain * 200 * 0.3 * 2 / np.pi  # gain x integral of one flow pulse
    t_dev = np.arange(1, 120, 3.7)
    sv = (t_dev, np.full(len(t_dev), sv_beat))
    hr = (t_dev, np.full(len(t_dev), 60 / rr))

    v[int(32 * FS) : int(38 * FS)] *= 1.5  # window 3: flow out of proportion
    p[int(52 * FS) : int(53 * FS)] = np.nan  # window 5: dropout
    v[int(75 * FS)] = 5000  # window 7: saturated FLOW sample

    w, g = cardioq.windows(p, v, FS, sv, hr)
    assert abs(g / gain - 1) < 0.02
    assert w.reason.tolist() == ["", "", "", "gain", "", "missing", "", "saturated"] + [""] * 4
    assert np.allclose(w.delay[w.ok], delay, atol=0.02)  # onsets on 16 Hz low-pass
    assert (w.beats[w.ok] >= 12).all()


def test_case_lists():
    usable = cardioq.cases()
    assert len(usable) == 24 and not set(usable) & set(cardioq.EXCLUDED)
    assert set(cardioq.DEVELOPMENT) <= set(usable)
