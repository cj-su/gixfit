"""Two-texture synthetic test: spots from ONE strained NPB cell in two orientations, (1 0 1) and (0 0 1), plus junk.
The greedy multi-component fit (started from the published cell, scanning every contact plane with |index| <= 2)
must find both textures, recover the shared cell, and put the spots in the right component; the chance-level
test (same pipeline on chi-shuffled surrogates) must call k = 2 significant.

python tests/test_multi.py            (add --no-null to skip the surrogate runs)
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gixfit.crystal import Cell, read_aims, PARAM_NAMES
from gixfit.geometry import Geometry
from gixfit.peaks import PeakTable
from gixfit.fit import FitConfig
from gixfit.multi import MultiConfig, decompose, null_test_multi
from test_synthetic import make_peaks, AIMS, PUB


def concat(*tables):
    keys = PeakTable.__dataclass_fields__
    return PeakTable(**{k: np.concatenate([getattr(t, k) for t in tables]) for k in keys})


def main(run_null=True):
    t0 = time.time()
    st = read_aims(AIMS, drop=("Mo", "O")).with_cell(PUB)
    st.name = "NPB"
    true = Cell(10.3077 * 1.015, 11.3545 * 0.99, 14.4778 * 1.005, 82.339 + 0.8, 77.657 - 1.2, 75.526 + 0.5)
    geom = Geometry(x0=577, y0=643, alpha_i_deg=0.1)
    shape = (1043, 981)
    mask = np.ones(shape, bool); mask[595:] = False
    A = make_peaks(st.with_cell(true), (1, 0, 1), geom, shape, n_strong=35, n_spurious=12, seed=1)
    B = make_peaks(st.with_cell(true), (0, 0, 1), geom, shape, n_strong=25, n_spurious=0, seed=2)
    B.area *= 0.6; B.amp *= 0.6
    nA = 35
    # drop B spots that coincide with an A spot (indistinguishable; would make the ground truth ambiguous)
    keepB = [i for i in range(len(B)) if np.min(np.hypot(A.col - B.col[i], A.row - B.row[i])) > 3]
    B = B.subset(np.array(keepB))
    pk = concat(A, B)
    truth = np.array([0] * nA + [-1] * (len(A) - nA) + [1] * len(B))       # 0 = (101), 1 = (001), -1 = junk
    cfg = FitConfig(q_range=(0.35, 1.9), n_strong=20)
    mcfg = MultiConfig(max_components=3, planes_maxidx=2, prescreen_top=6, n_null=8, null_modes=("chi",))
    print(f"{len(pk)} spots: {nA} from (1 0 1), {len(B)} from (0 0 1), {len(A) - nA} junk")
    hist = decompose([st], geom, pk, mask, cfg, mcfg, log=print)
    r2 = hist[1]
    found = [c for _, c in r2.components]
    print("k=2 components:", found)
    print("true   ", true)
    print("fit    ", r2.cells[0])
    print("error  ", " ".join(f"{n}:{v:+.4f}(+-{e:.4f})" for n, v, e in
                             zip(PARAM_NAMES, r2.cells[0].array() - true.array(), r2.cell_errs[0])))
    # spot -> component agreement
    comp_of = {row["obs"]: (0 if r2.components[row["component"]][1] == (1, 0, 1) else 1) for row in r2.pairs}
    real = np.nonzero(truth >= 0)[0]
    right = sum(1 for i in real if comp_of.get(int(i)) == truth[i])
    junk_idx = sum(1 for i in np.nonzero(truth < 0)[0] if int(i) in comp_of)
    print(f"true spots indexed in the right component: {right}/{len(real)};  junk spots indexed: {junk_idx}/{(truth < 0).sum()}")
    foms = [h.metrics["fom"] for h in hist]
    print("fom_k:", " ".join(f"{f:.3f}" for f in foms))
    assert set(found) == {(1, 0, 1), (0, 0, 1)}, "textures not found"
    dev = np.abs(r2.cells[0].array() - true.array())
    assert np.all(dev[:3] < 0.05) and np.all(dev[3:] < 0.3), "shared cell not recovered"
    assert right >= 0.85 * len(real), "spots not assigned to the right texture"
    assert foms[1] > foms[0] + 0.1, "second texture did not improve the fom"
    if run_null:
        nt = null_test_multi([st], geom, pk, mask, cfg, mcfg, hist, log=print)
        k2 = nt["chi"][1]
        assert k2["z"] > 3 and k2["inc_z"] > 3, "k=2 not significant against the chi-shuffled null"
    print(f"MULTI-TEXTURE RECOVERY OK ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(run_null="--no-null" not in sys.argv)
