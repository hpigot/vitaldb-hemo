import http.client
import io

import numpy as np
import pytest

from vitaldb_hemo import data

INDEX = {
    "SNUADC/ART": frozenset({19, 20, 1602}),
    "CardioQ/FLOW": frozenset({19, 1602, 3000}),
    "CardioQ/SV": frozenset({19, 1602}),
    "EV1000/SV": frozenset({20}),
}


def test_find_cases():
    assert data.find_cases(["SNUADC/ART"], INDEX) == [19, 20, 1602]
    assert data.find_cases(["CardioQ/FLOW", "CardioQ/SV", "SNUADC/ART"], INDEX) == [19, 1602]
    assert data.find_cases(["CardioQ/FLOW", "EV1000/SV"], INDEX) == []
    with pytest.raises(KeyError, match="CardioQ/FLOW"):
        data.find_cases(["CardioQ/Flow"], INDEX)


def test_bundled_track_list():
    assert len(data.tracks()) == 196
    cardioq = data.find_cases(["CardioQ/FLOW", "CardioQ/SV", "SNUADC/ART"])
    assert len(cardioq) == 27 and cardioq[:2] == [19, 468] and 1602 in cardioq


def test_download_resumes_dropped_transfers(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 400
    drops = iter([1000, 30000])  # the server cuts the first two transfers short
    ranges = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self, start):
            self.headers = {"Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}"}
            self.cut = next(drops, len(payload))
            super().__init__(payload[start : self.cut])

        def read(self, n=-1):
            chunk = super().read(n)
            if not chunk and self.cut < len(payload):
                raise http.client.IncompleteRead(b"")
            return chunk

    def urlopen(req, timeout):
        start = int(req.headers["Range"][len("bytes=") : -1])
        ranges.append(start)
        return Response(start)

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    out = tmp_path / "x.vital"
    data._download("http://example/x.vital", out)
    assert out.read_bytes() == payload
    assert ranges == [0, 1000, 30000]
    assert not (tmp_path / "x.vital.part").exists()


def test_resample_keeps_waveform_and_gaps():
    fs_in, fs_out, f = 500.0, 180.0, 3.0
    t = np.arange(0, 10, 1 / fs_in)
    x = np.sin(2 * np.pi * f * t)
    x[2000:2500] = np.nan  # 4.0-5.0 s
    y = data.resample(x, fs_in, fs_out)
    ty = np.arange(len(y)) / fs_out
    assert len(y) == 1800
    gap = (ty >= 4.0) & (ty < 5.0)
    assert np.isnan(y[gap]).all() and not np.isnan(y[~gap]).any()
    away = ((ty > 0.5) & (ty < 3.5)) | ((ty > 5.5) & (ty < 9.5))
    assert np.abs(y[away] - np.sin(2 * np.pi * f * ty[away])).max() < 1e-3


def test_resample_removes_content_above_new_nyquist():
    fs_in = 500.0
    t = np.arange(0, 10, 1 / fs_in)
    y = data.resample(np.sin(2 * np.pi * 100 * t), fs_in, 125.0)  # 100 Hz > 62.5 Hz
    assert np.abs(y[100:-100]).max() < 0.05  # nearest-sample picking would alias to 25 Hz


def test_artefact_flags():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    rng = np.random.default_rng(0)
    p = 80 + 20 * np.sin(2 * np.pi * 1.2 * t)  # 60-100 mmHg at 72 bpm
    p[1000:1500] = rng.normal(0, 4, 500)  # noisy zeroing, 10-15 s
    p[2500:3000] = 175 - np.linspace(0, 3, 500)  # flush plateau with drift, 25-30 s
    p[3300:3460] = 250.0  # flush peak shorter than the window, 33-34.6 s
    p[4200] = -200.0  # transient, 42 s
    p[5000:5100] = np.nan  # dropout, 50-51 s
    f = data.artefact_flags(p, fs)

    def flagged(m, a, b):
        return m[int(a * fs) : int(b * fs)]

    assert flagged(f["zero"], 10, 15).all() and not f["zero"][2000:].any()
    assert flagged(f["flat"], 25, 30).all() and not f["flat"][:2000].any()
    assert not flagged(f["range"], 18, 31).any()  # 175 mmHg is a plausible level
    assert flagged(f["range"], 33, 34.6).all()
    assert flagged(f["range"], 41, 43).all()
    assert flagged(f["missing"], 50, 51).all() and f["missing"].sum() == 100
    clean = ~np.any(list(f.values()), axis=0)
    for a, b in [(1, 7.5), (18, 22.5), (37, 39.5), (45, 49.5), (52, 59)]:
        assert flagged(clean, a, b).all(), (a, b)
