"""Image loading, masking and 2D peak finding on the RAW detector image (pixel space, no resampling).

detect -> difference of Gaussians (mask-aware normalised convolution), significance = DoG / sqrt(background),
          scaled by its robust (MAD) spread; local maxima above ``threshold``.
fit    -> each candidate: rotated 2D Gaussian + plane background on a (2h+1)^2 patch, Poisson weights.
          Gives sub-pixel centre with 1-sigma errors, widths, integrated intensity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares


# ---------------------------------------------------------------------- image
def load_image(path, despike=True, dead_pixels=()):
    import tifffile
    D = tifffile.imread(str(path)).astype(np.float64)
    for r, c in dead_pixels:
        D[r, c] = np.mean([D[r + 1, c + 1], D[r + 1, c - 1], D[r - 1, c + 1], D[r - 1, c - 1]])
    if despike:
        # isolated hot pixels / cosmic rays locked into max-stitched frames
        med = ndi.median_filter(D, size=3)
        bad = (D > med + 80.0 + 6.0 * np.sqrt(np.clip(med, 0, None))) & (D > 0)
        D[bad] = med[bad]
    return D


def build_mask(D, edge=15, nonpositive=True, rects=(), grow=2):
    """True = good pixel.  rects = [(row0,row1,col0,col1), ...] to exclude (beam stop, specular, ...)."""
    m = np.ones(D.shape, bool)
    if nonpositive:
        m &= D > 0
    if edge:
        m[:edge] = m[-edge:] = False
        m[:, :edge] = m[:, -edge:] = False
    for r0, r1, c0, c1 in rects:
        m[r0:r1, c0:c1] = False
    if grow:
        m = ~ndi.binary_dilation(~m, iterations=grow)
    return m


def _nconv(D, m, s):
    num = ndi.gaussian_filter(np.where(m, D, 0.0), s)
    den = ndi.gaussian_filter(m.astype(float), s)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / den, np.nan)


# ---------------------------------------------------------------------- detection
@dataclass
class PeakTable:
    col: np.ndarray
    row: np.ndarray
    ecol: np.ndarray
    erow: np.ndarray
    sig_major: np.ndarray      # px
    sig_minor: np.ndarray
    angle: np.ndarray          # rad, orientation of the major axis in (col,row)
    amp: np.ndarray
    area: np.ndarray           # integrated counts
    bg: np.ndarray
    snr: np.ndarray            # detection significance

    def __len__(self):
        return len(self.col)

    def subset(self, idx):
        return PeakTable(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def detect(D, mask, s_small=1.5, s_large=8.0, threshold=6.0, min_sep=4, max_peaks=400):
    g1 = _nconv(D, mask, s_small)
    g2 = _nconv(D, mask, s_large)
    dog = g1 - g2
    z = dog / np.sqrt(np.clip(g2, 1.0, None))
    good = mask & np.isfinite(z)
    zz = z[good]
    med = np.median(zz)
    mad = 1.4826 * np.median(np.abs(zz - med))
    snr = np.where(good, (z - med) / mad, -np.inf)
    mx = ndi.maximum_filter(np.nan_to_num(snr, neginf=-1e9), size=2 * min_sep + 1)
    cand = (snr == mx) & (snr > threshold) & good
    r, c = np.nonzero(cand)
    order = np.argsort(-snr[r, c])[:max_peaks]
    return r[order], c[order], snr[r, c][order]


def _gauss2d(p, X, Y):
    A, x0, y0, sx, sy, th, b0, bx, by = p
    ct, st = np.cos(th), np.sin(th)
    xr = (X - x0) * ct + (Y - y0) * st
    yr = -(X - x0) * st + (Y - y0) * ct
    return A * np.exp(-0.5 * (xr ** 2 / sx ** 2 + yr ** 2 / sy ** 2)) + b0 + bx * (X - x0) + by * (Y - y0)


def fit_peak(D, mask, r0, c0, half=8, max_shift=3.0):
    R0, R1 = max(r0 - half, 0), min(r0 + half + 1, D.shape[0])
    C0, C1 = max(c0 - half, 0), min(c0 + half + 1, D.shape[1])
    Y, X = np.mgrid[R0:R1, C0:C1]
    P = D[R0:R1, C0:C1]
    M = mask[R0:R1, C0:C1]
    if M.sum() < 30:
        return None
    X, Y, P = X[M].astype(float), Y[M].astype(float), P[M]
    w = 1.0 / np.sqrt(np.clip(P, 1.0, None))
    b0 = np.percentile(P, 20)
    p0 = [max(D[r0, c0] - b0, 1.0), c0, r0, 1.8, 1.8, 0.0, b0, 0.0, 0.0]
    lo = [0, c0 - max_shift, r0 - max_shift, 0.5, 0.5, -np.pi, -np.inf, -np.inf, -np.inf]
    hi = [np.inf, c0 + max_shift, r0 + max_shift, half, half, np.pi, np.inf, np.inf, np.inf]
    try:
        res = least_squares(lambda p: (_gauss2d(p, X, Y) - P) * w, p0, bounds=(lo, hi), x_scale="jac")
    except Exception:
        return None
    p = res.x
    dof = max(len(P) - len(p), 1)
    chi2r = 2 * res.cost / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * max(chi2r, 1.0)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return None
    A, x0, y0, sx, sy, th = p[:6]
    if not (A > 0 and np.isfinite(err[1]) and np.isfinite(err[2])):
        return None
    if sx < sy:
        sx, sy, th = sy, sx, th + np.pi / 2
    return dict(col=x0, row=y0, ecol=err[1], erow=err[2], sig_major=sx, sig_minor=sy,
                angle=(th + np.pi / 2) % np.pi - np.pi / 2, amp=A, area=2 * np.pi * A * sx * sy, bg=p[6],
                amp_err=err[0])


def find_peaks(D, mask, threshold=6.0, s_small=1.5, s_large=8.0, min_sep=4, half=8, max_peaks=400,
               min_amp_sigma=3.0, max_width_px=None, merge_px=2.5) -> PeakTable:
    r, c, s = detect(D, mask, s_small, s_large, threshold, min_sep, max_peaks)
    rows = []
    for ri, ci, si in zip(r, c, s):
        f = fit_peak(D, mask, int(ri), int(ci), half)
        if f is None or f["amp"] < min_amp_sigma * f["amp_err"]:
            continue
        if max_width_px and f["sig_major"] > max_width_px:
            continue
        f["snr"] = si
        rows.append(f)
    # merge duplicates (two detections converging on one peak)
    kept = []
    for f in sorted(rows, key=lambda d: -d["snr"]):
        if all(np.hypot(f["col"] - g["col"], f["row"] - g["row"]) > merge_px for g in kept):
            kept.append(f)
    keys = PeakTable.__dataclass_fields__
    if not kept:
        return PeakTable(**{k: np.zeros(0) for k in keys})
    return PeakTable(**{k: np.array([d[k] for d in kept]) for k in keys})


# ---------------------------------------------------------------------- q-space view of the image
def to_qgrid(D, mask, geom, qxy_lim=(-0.1, 2.0), qz_lim=(0.0, 2.0), dq=0.004):
    """resample the image onto a uniform (q_xy, q_z) grid by exact inverse mapping (nearest pixel)."""
    qxy = np.arange(qxy_lim[0], qxy_lim[1] + dq / 2, dq)
    qz = np.arange(qz_lim[0], qz_lim[1] + dq / 2, dq)
    QX, QZ = np.meshgrid(qxy, qz)
    col, row = geom.q_to_pixel(QX, QZ)
    ci = np.round(col)
    ri = np.round(row)
    ok = np.isfinite(ci) & np.isfinite(ri) & (ci >= 0) & (ri >= 0) & (ci < D.shape[1]) & (ri < D.shape[0])
    img = np.full(QX.shape, np.nan)
    ci, ri = ci[ok].astype(int), ri[ok].astype(int)
    vals = np.where(mask[ri, ci], D[ri, ci], np.nan)
    img[ok] = vals
    return qxy, qz, img
