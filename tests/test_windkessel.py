import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from vitaldb_hemo import identify, windkessel
from vitaldb_hemo.windkessel import wk2, wk3, wk4

REF = Path(__file__).parent / "reference"  # from WindkesselIdentification.jl test/reference
H = 0.005
THETA = np.array([0.79, 1.22, 0.056, 0.0051])  # Stergiopulos et al. 1999, type A beat
START = np.array([0.85, 1.5, 0.05, 0.006])
MODELS = {"wk2": (wk2, 2), "wk3": (wk3, 3), "wk4": (wk4, 4)}
Q, P = np.loadtxt(REF / "stergiopulos1999_beat.csv", delimiter=",", skiprows=3).T


def csv(name):
    """Header row and data rows of a reference file, skipping its # comment lines."""
    rows = [r.split(",") for r in (REF / name).read_text().splitlines() if not r.startswith("#")]
    return rows[0], rows[1:]


def fits():
    return {r[0]: np.array(r[1:], float) for r in csv("fits.csv")[1]}


def test_simulation_matches_julia():
    names, rows = csv("simulation.csv")
    ref = dict(zip(names, np.array(rows, float).T))
    assert np.allclose(ref["q"], Q)
    for name, (model, n) in MODELS.items():
        x0 = windkessel.solvex0(THETA[:n], Q, H, model)
        p = windkessel.simulate(THETA[:n], x0, Q, H, model)
        assert np.allclose(p, ref[f"p_{name}"], rtol=1e-12), name


def test_zoh_and_periodic_state():
    sys_d = windkessel.statespace(THETA, H)
    c = windkessel.statespace(THETA)
    ad, bd, *_ = signal.cont2discrete((c.A, c.B, c.C, c.D), H, method="zoh")
    assert np.allclose(sys_d.A, ad) and np.allclose(sys_d.B, bd)
    x0 = windkessel.solvex0(THETA, Q, H)
    _, x = windkessel.simulate(THETA, x0, np.r_[Q, Q[:1]], H, states=True)
    assert np.allclose(x[-1], x0)  # back where it started after one beat


def test_fits_match_julia():
    for name, (model, n) in MODELS.items():
        ref = fits()[name]
        f = identify.identify(START[:n], Q, P, H, model, x0="periodic")
        assert np.allclose(f.theta, ref[:n], rtol=1e-5), name
        assert np.isclose(f.mse, ref[-1], rtol=1e-8), name
        assert np.allclose(f.x0, ref[4 : 4 + len(f.x0)], rtol=1e-5), name


def test_fit_options():
    ref = fits()["wk3"]
    # estimated x0: the periodic fit is one of its candidates, so it does at least as well
    f = identify.identify(START[:3], Q, P, H, wk3)
    assert f.mse < ref[-1]
    # Rp fixed at its fitted value: the others are recovered
    f = identify.identify(np.r_[ref[0], START[1:3]], Q, P, H, wk3, free=[0, 1, 1], x0="periodic")
    assert f.theta[0] == ref[0] and np.allclose(f.theta, ref[:3], rtol=1e-5)
    # Nelder-Mead from a poor start
    f = identify.identify_global([2.0, 0.5, 0.2], Q, P, H, wk3, x0="periodic")
    assert np.allclose(f.theta, ref[:3], rtol=1e-3)


def test_models_do_not_import_clinical_code():
    code = "import sys, vitaldb_hemo.identify; assert not {'vitaldb', 'vitaldb_hemo.data'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
