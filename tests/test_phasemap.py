"""Two-crystal synthetic test for gixfit.phasemap: spots from crystal A (strained NPB, contact (1 0 1)) and crystal B
(strained alpha-MoO3, contact (0 1 0)) plus random junk spots in one pattern.  The multi-phase fit starts from the
CIF cells and scans the contact planes of both crystals; phase_map must then put each spot in the right crystal.

python tests/test_phasemap.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gixfit.crystal import Cell, read_aims, read_cif
from gixfit.geometry import Geometry
from gixfit.fit import FitConfig
from gixfit.multi import MultiConfig
from gixfit.phasemap import fit_phases, phase_map, summary, UNASSIGNED
from test_synthetic import make_peaks, AIMS, PUB
from test_multi import concat

MOO3 = os.path.join(ROOT, "library", "MoO3_alpha_Pbnm_Kihlborg1963_COD9014282.cif")


def main():
    t0 = time.time()
    A = read_aims(AIMS, drop=("Mo", "O")).with_cell(PUB); A.name = "NPB"
    B = read_cif(MOO3); B.name = "MoO3"
    trueA = Cell(10.3077 * 1.012, 11.3545 * 0.992, 14.4778 * 1.004, 82.339 + 0.6, 77.657 - 0.8, 75.526 + 0.4)
    b0 = B.cell
    trueB = Cell(b0.a * 0.99, b0.b * 1.01, b0.c * 1.005, b0.alpha, b0.beta, b0.gamma)
    geom = Geometry(x0=577, y0=643, alpha_i_deg=0.1)
    shape = (1043, 981)
    mask = np.ones(shape, bool); mask[595:] = False
    pa = make_peaks(A.with_cell(trueA), (1, 0, 1), geom, shape, n_strong=35, n_spurious=12, seed=1)
    pb = make_peaks(B.with_cell(trueB), (0, 1, 0), geom, shape, n_strong=15, n_spurious=0, seed=2)
    nA, nB = 35, len(pb)
    truth = ["NPB"] * nA + ["junk"] * (len(pa) - nA) + ["MoO3"] * nB
    pk = concat(pa, pb)
    cfg = FitConfig(q_range=(0.35, 2.4), qmax=2.5, n_strong=20)
    mc = MultiConfig(max_components=2, planes_maxidx=1, n_null=0, workers=1)
    res, hist, forced = fit_phases([A, B], geom, pk, mask, cfg, mc, log=print)
    print("model:", res.label(), " forced:", forced)
    rows = phase_map(res, pk)
    print("labels:", {k: v["n"] for k, v in summary(rows).items()})
    got = {r["spot"]: r for r in rows}
    ok = wrong = shared = missing = 0
    for i, t in enumerate(truth):
        r = got.get(i)
        if r is None:
            continue                                  # outside the fitted q range
        lab = r["label"]
        if t == "junk":
            ok += lab == UNASSIGNED
            wrong += lab != UNASSIGNED
        elif lab == t:
            ok += 1
        elif r["n_candidates"] > 1 and t in lab:
            shared += 1                               # honestly flagged as ambiguous
        elif lab == UNASSIGNED:
            missing += 1
        else:
            wrong += 1
            print(f"  WRONG spot {i}: truth {t}, label {lab}, q {r['q']:.3f}, chi {r['chi_deg']:.1f}")
    n = ok + wrong + shared + missing
    print(f"correct {ok}/{n}, flagged shared {shared}, not found {missing}, wrong {wrong}  ({time.time() - t0:.0f} s)")
    comps = {res.phase_names[p]: tuple(c) for p, c in res.components}
    assert comps.get("NPB") == (1, 0, 1), f"NPB contact plane not found: {comps}"
    assert comps.get("MoO3") in ((0, 1, 0),), f"MoO3 contact plane not found: {comps}"
    assert wrong <= 0.05 * n, "too many spots given to the wrong crystal"
    assert ok >= 0.8 * n, "too few spots labelled correctly"
    print("PHASE MAP OK")


if __name__ == "__main__":
    main()
