"""Beam centre + SDD from the ITO (222) Debye ring (q = 2.151 A^-1, In2O3 bixbyite, a = 10.118 A).

For each azimuth sector the ring radius is located on the radial profile (Gaussian + linear background); a circle
is fitted to the ring points -> (x0, y0, R).  With q fixed, R gives the SDD.  The ring is only seen above the sample
horizon, so the vertical centre is less well constrained than the horizontal one (both uncertainties are printed).

python examples/ito_ring_calibration.py <image.tiff> [x0 y0 sdd]
"""
import sys
import numpy as np
from scipy.optimize import least_squares, curve_fit

sys.path.insert(0, __file__.rsplit("examples", 1)[0])
from gixfit.peaks import load_image

LAM, PIX = 0.9184, 0.172
Q_ITO222 = 2 * np.pi / (10.118 / np.sqrt(12))      # 2.151 A^-1


def ring_points(D, x0, y0, sdd, q=Q_ITO222, half_width_px=25, nsect=72):
    r_nom = sdd * np.tan(2 * np.arcsin(q * LAM / (4 * np.pi))) / PIX
    rows, cols = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    dx, dy = cols - x0, y0 - rows
    r = np.hypot(dx, dy)
    phi = np.degrees(np.arctan2(dx, dy))                 # 0 = up, +90 = right
    good = (D > 0) & (np.abs(r - r_nom) < half_width_px) & (dy > 15)
    pts = []
    for a0 in np.linspace(-90, 90, nsect, endpoint=False):
        m = good & (phi >= a0) & (phi < a0 + 180 / nsect)
        if m.sum() < 60:
            continue
        rb = np.round(r[m]).astype(int)
        lo = rb.min()
        I = np.bincount(rb - lo, weights=D[m]) / np.maximum(np.bincount(rb - lo), 1)
        rr = np.arange(len(I)) + lo
        if len(rr) < 15:
            continue
        f = lambda x, A, c, s, b0, b1: A * np.exp(-0.5 * ((x - c) / s) ** 2) + b0 + b1 * (x - r_nom)
        try:
            p, cov = curve_fit(f, rr, I, p0=[I.max() - np.median(I), rr[np.argmax(I)], 4, np.median(I), 0],
                               maxfev=4000)
        except Exception:
            continue
        err = np.sqrt(np.diag(cov))[1]
        if p[0] <= 0 or not (2 < abs(p[2]) < 15) or not np.isfinite(err) or err > 1.5 or p[0] < 5 * np.sqrt(abs(p[3]) + 1) / 5:
            continue
        a = np.radians(a0 + 90 / nsect)
        pts.append((x0 + p[1] * np.sin(a), y0 - p[1] * np.cos(a), err, a0))
    return np.array(pts)


def fit_circle(pts, x0, y0):
    x, y, e = pts[:, 0], pts[:, 1], pts[:, 2]
    res = least_squares(lambda p: (np.hypot(x - p[0], y - p[1]) - p[2]) / e,
                        [x0, y0, np.median(np.hypot(x - x0, y - y0))], loss="soft_l1")
    J = res.jac
    chi2r = 2 * res.cost / max(len(x) - 3, 1)
    err = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) * max(chi2r, 1))
    return res.x, err


if __name__ == "__main__":
    path = sys.argv[1]
    x0, y0, sdd = (float(v) for v in sys.argv[2:5]) if len(sys.argv) >= 5 else (458.0, 637.0, 259.0)
    D = load_image(path)
    for it in range(3):
        pts = ring_points(D, x0, y0, sdd)
        (xc, yc, R), (ex, ey, eR) = fit_circle(pts, x0, y0)
        sdd_new = R * PIX / np.tan(2 * np.arcsin(Q_ITO222 * LAM / (4 * np.pi)))
        print(f"iter {it}: {len(pts)} ring points, azimuth {pts[:, 3].min():.0f}..{pts[:, 3].max():.0f} deg | "
              f"x0 = {xc:.2f} +- {ex:.2f}, y0 = {yc:.2f} +- {ey:.2f} (0-indexed), R = {R:.2f} +- {eR:.2f} px "
              f"-> SDD = {sdd_new:.2f} mm")
        x0, y0, sdd = xc, yc, sdd_new
    # ring q with the nominal geometry, for reference
    Rn = np.hypot(pts[:, 0] - 458.0, pts[:, 1] - 637.0)
    qn = 4 * np.pi / LAM * np.sin(0.5 * np.arctan(Rn * PIX / 259.0))
    print(f"nominal geometry (x0 458, y0 637, SDD 259): ring q = {qn.mean():.4f} +- {qn.std():.4f} (expected {Q_ITO222:.4f})")
