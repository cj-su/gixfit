"""Predicted GIWAXS patterns for a fibre-textured film.

A film with contact plane (H K L) parallel to the substrate and random in-plane rotation (fibre texture)
puts every reciprocal-lattice vector G on a ring:  q_z = G . n,  q_xy = sqrt(|G|^2 - q_z^2),  n = G_HKL/|G_HKL|.
Each reflection appears at +q_xy and -q_xy on the detector.

Predicted intensity (relative, for ranking/visibility only):
    I ~ |F|^2 / sqrt(q_xy^2 + (|G| sin(mosaic))^2)
i.e. the fibre "Lorentz" factor (1/q_xy, ring circumference) regularised by the mosaic spread.  Polarisation,
absorption and the DWBA transmission factor are ignored - adequate for deciding which peaks should be visible,
not for quantitative intensity refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .crystal import Cell, Structure


def hkl_sphere(cell: Cell, qmax: float) -> np.ndarray:
    A = cell.direct_matrix()
    lim = [int(np.ceil(qmax * np.linalg.norm(A[i]) / (2 * np.pi))) for i in range(3)]
    h, k, l = np.mgrid[-lim[0]:lim[0] + 1, -lim[1]:lim[1] + 1, -lim[2]:lim[2] + 1]
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], 1)
    hkl = hkl[np.any(hkl != 0, 1)]
    q = np.linalg.norm(cell.g(hkl), axis=1)
    return hkl[q <= qmax]


def rotate_normal(n: np.ndarray, tilt_deg=(0.0, 0.0)) -> np.ndarray:
    """small tilt of the fibre axis away from G_HKL (two angles about two axes perpendicular to n)."""
    if not any(tilt_deg):
        return n
    u = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    t1, t2 = np.radians(tilt_deg)
    m = n + np.tan(t1) * u + np.tan(t2) * v
    return m / np.linalg.norm(m)


def fiber_qcoords(cell: Cell, hkl, contact, tilt_deg=(0.0, 0.0)):
    G = cell.g(hkl)
    n = cell.g(contact)
    n = rotate_normal(n / np.linalg.norm(n), tilt_deg)
    qz = G @ n
    q = np.linalg.norm(G, axis=1)
    qxy = np.sqrt(np.clip(q ** 2 - qz ** 2, 0, None))
    return qxy, qz, q


def wedge_project(qxy, qz, geom, mosaic_deg=3.0, n_sigma=2.0):
    """Reflections inside the GIWAXS missing wedge (|q_xy| below the Ewald-sphere limit, e.g. the specular
    (HKL) itself) are seen only through the mosaic spread, at the wedge edge.  Move those within n_sigma*mosaic
    of the edge onto it at the same |q| (where the arc meets the Ewald sphere); others are left (inaccessible)."""
    qxy = np.asarray(qxy, float)
    qz = np.asarray(qz, float)
    q = np.hypot(qxy, qz)
    k, a = geom.k, np.radians(geom.alpha_i_deg)
    qx = -q ** 2 / (2 * k * np.cos(a))
    for _ in range(3):
        qzz = np.sqrt(np.clip(q ** 2 - qx ** 2, 0, None))
        qx = (2 * k * np.sin(a) * qzz - q ** 2) / (2 * k * np.cos(a))
    w = np.abs(qx) * 1.002 + 1e-4
    near = (np.abs(qxy) < w) & (w - np.abs(qxy) <= q * np.sin(np.radians(n_sigma * mosaic_deg)))
    sgn = np.where(qxy < 0, -1.0, 1.0)
    qxy2 = np.where(near, sgn * w, qxy)
    qz2 = np.where(near, np.sqrt(np.clip(q ** 2 - w ** 2, 0, None)), qz)
    return qxy2, qz2


@dataclass
class Reflections:
    hkl: np.ndarray        # (n,3)
    side: np.ndarray       # +1 / -1 (sign of q_xy on the detector), 0 = on the q_z axis
    qxy: np.ndarray        # signed
    qz: np.ndarray
    q: np.ndarray
    F2: np.ndarray
    I: np.ndarray          # relative predicted intensity (max = 1)


_F2_CACHE: dict = {}


def _f2_cached(struct: Structure, hkl, qmax):
    """|F|^2 does not depend on the contact plane -> compute once per (structure, cell, qmax) when scanning planes."""
    key = (id(struct), tuple(np.round(struct.cell.array(), 6)), round(float(qmax), 6), len(hkl))
    hit = _F2_CACHE.get(key)
    if hit is not None and hit[0] is struct:
        return hit[1]
    if len(_F2_CACHE) > 64:
        _F2_CACHE.clear()
    F2 = struct.f_squared(hkl)
    _F2_CACHE[key] = (struct, F2)
    return F2


def fiber_pattern(struct: Structure, contact, qmax=2.0, mosaic_deg=3.0, tilt_deg=(0.0, 0.0),
                  qz_min=0.0, axis_tol=1e-3) -> Reflections:
    cell = struct.cell
    hkl = hkl_sphere(cell, qmax)
    qxy, qz, q = fiber_qcoords(cell, hkl, contact, tilt_deg)
    keep = qz >= qz_min - 1e-9
    # reflections lying exactly in the surface plane (q_z = 0): +G and -G coincide -> keep one
    first = np.array([next((v for v in r if v != 0), 0) for r in hkl])
    keep &= ~((np.abs(qz) < 1e-9) & (first < 0))
    F2_all = _f2_cached(struct, hkl, qmax)
    hkl, qxy, qz, q = hkl[keep], qxy[keep], qz[keep], q[keep]
    F2 = F2_all[keep]
    lor = 1.0 / np.sqrt(qxy ** 2 + (q * np.sin(np.radians(mosaic_deg))) ** 2)
    I = F2 * lor
    # normalise to the 10th strongest reflection (robust: the near-specular Lorentz term can make one
    # reflection dominate); values > 1 are allowed
    ref = np.sort(I)[-min(10, len(I))] if len(I) else 1.0
    I = I / ref if ref > 0 else I
    on_axis = qxy < axis_tol
    # mirror to both detector sides
    hkl2 = np.concatenate([hkl, hkl[~on_axis]])
    side = np.concatenate([np.where(on_axis, 0, 1), -np.ones((~on_axis).sum(), int)])
    qxy2 = np.concatenate([qxy, -qxy[~on_axis]])
    cat = lambda a: np.concatenate([a, a[~on_axis]])
    return Reflections(hkl2, side, qxy2, cat(qz), cat(q), cat(F2), cat(I))


def hkl_str(h) -> str:
    return "(" + " ".join(str(int(v)) for v in h) + ")"


@dataclass
class Group:
    """reflections that overlap at the experimental resolution (counted as one predicted peak)."""
    members: list           # indices into Reflections
    side: int
    label: str

    def position(self, refl: Reflections, qxy=None, qz=None):
        qxy = refl.qxy if qxy is None else qxy
        qz = refl.qz if qz is None else qz
        w = refl.I[self.members] + 1e-12
        return (np.sum(w * qxy[self.members]) / w.sum(), np.sum(w * qz[self.members]) / w.sum())


def group_reflections(refl: Reflections, merge_tol=0.008, min_rel=0.0, accessible=None) -> list:
    """greedy clustering, strongest first: a reflection joins the EARLIEST group whose seed (its strongest member)
    is closer than merge_tol on the same detector side.  ``accessible`` = bool mask of reflections on good pixels.
    Seeds are kept in a grid hash (cell = merge_tol), so this is O(n) instead of O(n x groups)."""
    keep = refl.I >= min_rel
    if accessible is not None:
        keep &= accessible
    idx = np.nonzero(keep)[0]
    order = idx[np.argsort(-refl.I[idx], kind="stable")]
    qx = refl.qxy.tolist(); qz = refl.qz.tolist(); sd = refl.side.tolist()
    inv = 1.0 / merge_tol
    tol2 = merge_tol * merge_tol
    grid: dict = {}
    groups, seeds = [], []
    for i in order.tolist():
        x, z, s = qx[i], qz[i], sd[i]
        cx, cz = int(np.floor(x * inv)), int(np.floor(z * inv))
        best = None
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for gi in grid.get((s, cx + dx, cz + dz), ()):
                    if best is not None and gi >= best:
                        continue
                    sx, sz = seeds[gi]
                    if (x - sx) ** 2 + (z - sz) ** 2 < tol2:
                        best = gi
        if best is None:
            grid.setdefault((s, cx, cz), []).append(len(groups))
            groups.append(Group([i], int(s), ""))
            seeds.append((x, z))
        else:
            groups[best].members.append(i)
    for g in groups:
        g.label = "/".join(hkl_str(refl.hkl[m]) for m in g.members[:3]) + ("/..." if len(g.members) > 3 else "")
    return groups


def group_intensity(refl: Reflections, groups) -> np.ndarray:
    return np.array([refl.I[g.members].sum() for g in groups])


def low_index_planes(maxidx=2):
    """unique contact planes (h k l) up to sign, gcd = 1."""
    from math import gcd
    out = []
    r = range(-maxidx, maxidx + 1)
    for h in r:
        for k in r:
            for l in r:
                if (h, k, l) == (0, 0, 0) or gcd(gcd(abs(h), abs(k)), abs(l)) != 1:
                    continue
                first = next(v for v in (h, k, l) if v != 0)
                if first > 0:
                    out.append((h, k, l))
    out.sort(key=lambda t: (sum(abs(v) for v in t), t))
    return out
