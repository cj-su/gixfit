"""python tests/test_core.py   (plain asserts; no pytest needed)"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gixfit.crystal import Cell, read_cif
from gixfit.geometry import Geometry
from gixfit.pattern import fiber_qcoords

NPB_DIR = os.path.join(os.path.dirname(ROOT), "giwaxs_20260831 NPB MoO3")


def test_cell_d():
    c = Cell(10.3077, 11.3545, 14.4778, 82.339, 77.657, 75.526)       # published NPB cell
    ref = {(0, 0, 1): 14.093, (0, 1, 0): 10.955, (1, 0, 0): 9.803, (0, 1, 1): 9.026, (1, 0, 1): 8.867,
           (0, 1, -1): 8.315, (1, 1, 0): 8.310, (1, 1, 1): 7.981, (1, 0, -1): 7.420, (0, 0, 2): 7.046}
    for hkl, d in ref.items():
        assert abs(c.d(hkl) - d) < 2e-3, (hkl, c.d(hkl), d)
    # handoff 5.7: (110) under (101) texture -> q 0.756, chi 62.0 deg
    qxy, qz, q = fiber_qcoords(c, np.array([[1, 1, 0]]), (1, 0, 1))
    chi = np.degrees(np.arctan2(qxy, qz))[0]
    assert abs(q[0] - 0.756) < 1e-3 and abs(chi - 62.0) < 0.3, (q, chi)
    print("cell / d-spacings / fibre chi OK")


def test_geometry_roundtrip():
    for ai, ac in ((0.1, None), (0.05, None), (0.1, 0.102)):
        g = Geometry(x0=577, y0=643, alpha_i_deg=ai, alpha_c_deg=ac)
        rows, cols = np.mgrid[20:600:37, 20:960:41]
        qxy, qz = g.pixel_to_q(cols, rows)
        c2, r2 = g.q_to_pixel(qxy, qz)
        ok = np.isfinite(c2)
        assert ok.mean() > 0.9
        assert np.nanmax(np.abs(c2 - cols)[ok]) < 1e-6 and np.nanmax(np.abs(r2 - rows)[ok]) < 1e-6
    print("geometry round trip OK")


def test_vs_legacy():
    """exact mapping vs the old pipeline's per-axis formula along the q_z axis (column = beam centre)."""
    g = Geometry(x0=577, y0=643, alpha_i_deg=0.10)
    rows = np.arange(100, 600)
    qxy, qz = g.pixel_to_q(np.full(rows.shape, 577.0), rows)
    legacy = 4 * np.pi * np.sin(0.5 * np.arctan(0.172 * (643 - rows) / 259.0)) / 0.9184
    # |q| must agree exactly (it only depends on the scattering angle); the legacy formula puts all of it
    # into q_z, the exact one splits off the Ewald-curvature q_xy (missing wedge) -> report the size
    assert np.max(np.abs(np.hypot(qxy, qz) - legacy)) < 1e-9
    print(f"  specular column: |q| identical; q_z exact - legacy max {np.max(np.abs(qz - legacy)):.4f} A^-1, "
          f"q_xy there up to {np.max(np.abs(qxy)):.4f} A^-1 (Ewald curvature)")
    # off-axis: q_xy differs (Ewald curvature); report the size
    cols = np.arange(600, 960)
    qxy, qz = g.pixel_to_q(cols, np.full(cols.shape, 300.0))
    lq = 4 * np.pi * np.sin(0.5 * np.arctan(0.172 * (cols - 577) / 259.0)) / 0.9184
    lz = 4 * np.pi * np.sin(0.5 * np.arctan(0.172 * (643 - 300) / 259.0)) / 0.9184
    print(f"  off-axis (row 300): |q| exact vs legacy max diff "
          f"{np.max(np.abs(np.hypot(qxy, qz) - np.hypot(lq, lz))):.4f} A^-1")


def test_structure_factor_vs_pymatgen():
    from pymatgen.core import Structure as PS
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    cif = os.path.join(NPB_DIR, "reference_cif", "MoO3_alpha_Pbnm_Kihlborg1963_COD9014282.cif")
    st = read_cif(cif, b_iso=0.0)
    ps = PS.from_file(cif)
    pat = XRDCalculator(wavelength=0.9184).get_pattern(ps, two_theta_range=(5, 40))
    # compare |F|^2 x multiplicity x LP via pymatgen's own hkl lists
    k = 2 * np.pi / 0.9184
    mine, ref = [], []
    for tt, I, hk in zip(pat.x, pat.y, pat.hkls):
        th = np.radians(tt / 2)
        lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
        tot = 0.0
        for h in hk:
            hkl = np.array(h["hkl"])[None]
            tot += st.f_squared(hkl)[0] * h["multiplicity"]
        mine.append(tot * lp); ref.append(I)
    mine = np.array(mine) / max(mine) * 100
    ref = np.array(ref)
    c = np.corrcoef(mine, ref)[0, 1]
    print(f"  |F|^2 vs pymatgen XRD intensities (alpha-MoO3, {len(ref)} lines): r = {c:.4f}")
    assert c > 0.99


if __name__ == "__main__":
    test_cell_d()
    test_geometry_roundtrip()
    test_vs_legacy()
    test_structure_factor_vs_pymatgen()
    print("ALL OK")
