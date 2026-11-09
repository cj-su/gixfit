"""Synthetic recovery test: fake peaks from a KNOWN strained NPB cell (+ noise + spurious peaks), start the
refinement from the published cell, check the true cell / contact plane are recovered.

python tests/test_synthetic.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gixfit.crystal import Cell, read_cif, PARAM_NAMES
from gixfit.geometry import Geometry
from gixfit.pattern import fiber_pattern, group_reflections, group_intensity
from gixfit.peaks import PeakTable
from gixfit.fit import FitConfig, Problem

AIMS = os.path.join(ROOT, "library", "NPB_published_cell_DFT_atoms.cif")
PUB = Cell(10.3077, 11.3545, 14.4778, 82.339, 77.657, 75.526)


def require_npb():
    """The NPB test structure is collaborator-derived and not in the public repository: skip, don't fail."""
    if not os.path.exists(AIMS):
        print(f"SKIP: {os.path.relpath(AIMS, ROOT)} is not distributed with the repository.")
        sys.exit(0)


def make_peaks(struct, contact, geom, shape, n_strong=45, n_spurious=15, noise_px=0.3, seed=1):
    rng = np.random.default_rng(seed)
    refl = fiber_pattern(struct, contact, qmax=1.9)
    col, row = geom.q_to_pixel(refl.qxy, refl.qz)
    acc = np.isfinite(col) & (col > 20) & (col < shape[1] - 20) & (row > 20) & (row < 590) & (refl.q > 0.35)
    groups = group_reflections(refl, 0.008, 0.003, acc)
    gI = group_intensity(refl, groups)
    keep = np.argsort(-gI)[:n_strong]
    cols, rows, areas = [], [], []
    for j in keep:
        g = groups[j]
        x, z = g.position(refl)
        c, r = geom.q_to_pixel(x, z)
        cols.append(c + rng.normal(0, noise_px)); rows.append(r + rng.normal(0, noise_px)); areas.append(1e5 * gI[j])
    for _ in range(n_spurious):                      # unindexed junk (other phases, substrate, artefacts)
        cols.append(rng.uniform(600, 950)); rows.append(rng.uniform(80, 580)); areas.append(rng.uniform(1e3, 3e4))
    n = len(cols)
    z = np.zeros(n)
    return PeakTable(np.array(cols), np.array(rows), z + noise_px, z + noise_px, z + 1.5, z + 1.2, z,
                     np.array(areas), np.array(areas), z, z + 50)


def main():
    require_npb()
    st = read_cif(AIMS).with_cell(PUB)
    true = Cell(10.3077 * 1.015, 11.3545 * 0.99, 14.4778 * 1.005, 82.339 + 0.8, 77.657 - 1.2, 75.526 + 0.5)
    geom = Geometry(x0=577, y0=643, alpha_i_deg=0.1)
    shape = (1043, 981)
    mask = np.ones(shape, bool); mask[595:] = False
    pk = make_peaks(st.with_cell(true), (1, 0, 1), geom, shape)
    cfg = FitConfig(q_range=(0.35, 1.9), n_strong=20)
    print(f"{len(pk)} synthetic peaks (45 true + 15 spurious)")
    print("true   ", true)
    out = []
    for contact in [(1, 0, 1), (1, 0, 0), (0, 1, 1), (0, 0, 1)]:
        r = Problem(st, contact, geom, pk, mask, cfg).run()
        m = r.metrics
        out.append((m["fom"], contact, r))
        print(f"{contact}: matched {m['n_match']}/{m['n_obs']} fom {m['fom']:.3f} rms CIF {m['rms_dq_cif_pct']:.2f}% "
              f"-> fit {m['rms_dq_fit_pct']:.2f}%  rho {m['spearman_I']:.2f}")
    best = max(out, key=lambda t: t[0])
    r = best[2]
    print("best contact", best[1])
    print("fit    ", r.cell)
    print("error  ", " ".join(f"{n}:{v:+.4f}(+-{e:.4f})" for n, v, e in
                             zip(PARAM_NAMES, r.cell.array() - true.array(), r.cell_err)))
    dev = np.abs(r.cell.array() - true.array())
    assert best[1] == (1, 0, 1), "wrong contact plane"
    assert np.all(dev[:3] < 0.05) and np.all(dev[3:] < 0.3), "cell not recovered"
    print("SYNTHETIC RECOVERY OK")


if __name__ == "__main__":
    main()
