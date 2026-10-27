"""Grazing-incidence detector geometry: pixel <-> (q_xy, q_z), exact (no small-angle approximation).

Conventions
-----------
* Detector flat and perpendicular to the direct beam at distance ``sdd_mm``.
* ``x0, y0`` = position of the DIRECT beam on the detector, 0-indexed (column, row).
  Columns increase to the right (+q_xy), rows increase DOWNWARD (row < y0 is above the beam).
* Lab frame: beam along +X, Y horizontal, Z up.  The sample is tilted by alpha_i so that
  k_i(sample) = k (cos a_i, 0, -sin a_i); sample axes in the lab:
      e_x = (cos a_i, 0, sin a_i),  e_y = (0, 1, 0),  e_z = (-sin a_i, 0, cos a_i).
* q_xy is SIGNED (sign of q_y, i.e. left/right of the beam); q_z along the surface normal.

Optional refraction correction (``alpha_c_deg``): angles inside the film are taken as
sqrt(a^2 - a_c^2) (0 below a_c) for both the incident and exit beam (simple DWBA-free
Snell correction).  Off by default: compare films at the SAME incident angle instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import numpy as np


@dataclass
class Geometry:
    wavelength: float = 0.9184      # A
    sdd_mm: float = 259.0           # sample-detector distance
    pixel_mm: float = 0.172
    x0: float = 577.0               # direct-beam column (0-indexed)
    y0: float = 643.0               # direct-beam row (0-indexed)
    alpha_i_deg: float = 0.10
    alpha_c_deg: float | None = None   # film critical angle for refraction correction (None = off)

    @property
    def k(self) -> float:
        return 2.0 * np.pi / self.wavelength

    def with_(self, **kw) -> "Geometry":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ helpers
    def _axes(self):
        a = np.radians(self.alpha_i_deg)
        ex = np.array([np.cos(a), 0.0, np.sin(a)])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([-np.sin(a), 0.0, np.cos(a)])
        return ex, ey, ez

    def _refract_qz(self, kfz_unit):
        """kfz_unit = sin(alpha_f) (external).  returns q_z with Snell-corrected internal angles."""
        ac = np.radians(self.alpha_c_deg)
        ai = np.radians(self.alpha_i_deg)
        sin_i = np.sqrt(np.clip(np.sin(ai) ** 2 - np.sin(ac) ** 2, 0, None))
        s2 = kfz_unit ** 2 - np.sin(ac) ** 2
        sin_f = np.sign(kfz_unit) * np.sqrt(np.clip(s2, 0, None))
        return self.k * (sin_f + sin_i)

    # ------------------------------------------------------------------ forward
    def pixel_to_q(self, col, row):
        """col,row (any shape, may be fractional) -> (qxy_signed, qz).  A^-1."""
        col = np.asarray(col, float)
        row = np.asarray(row, float)
        Y = (col - self.x0) * self.pixel_mm
        Z = (self.y0 - row) * self.pixel_mm
        X = np.full_like(Y, self.sdd_mm)
        R = np.sqrt(X * X + Y * Y + Z * Z)
        k = self.k
        kf = np.stack([X / R, Y / R, Z / R], axis=-1) * k
        q = kf - np.array([k, 0.0, 0.0])
        ex, ey, ez = self._axes()
        qx, qy, qz = q @ ex, q @ ey, q @ ez
        if self.alpha_c_deg:
            qz = self._refract_qz((kf @ ez) / k)
        qxy = np.where(qy < 0, -1.0, 1.0) * np.sqrt(qx * qx + qy * qy)
        return qxy, qz

    # ------------------------------------------------------------------ inverse
    def q_to_pixel(self, qxy, qz):
        """(qxy_signed, qz) -> (col, row); NaN where not reachable (missing wedge / behind)."""
        qxy = np.asarray(qxy, float)
        qz = np.asarray(qz, float)
        k = self.k
        a = np.radians(self.alpha_i_deg)
        if self.alpha_c_deg:
            # invert the Snell correction: external sin(alpha_f) from internal q_z
            ac = np.radians(self.alpha_c_deg)
            sin_i = np.sqrt(np.clip(np.sin(a) ** 2 - np.sin(ac) ** 2, 0, None))
            sin_f_int = qz / k - sin_i
            sin_f_ext = np.sign(sin_f_int) * np.sqrt(sin_f_int ** 2 + np.sin(ac) ** 2)
            qz = k * (sin_f_ext + np.sin(a))
        # |k_i + q| = k  ->  qx fixed by (qxy, qz)
        qx = (2 * k * np.sin(a) * qz - qxy ** 2 - qz ** 2) / (2 * k * np.cos(a))
        rad = qxy ** 2 - qx ** 2
        with np.errstate(invalid="ignore"):
            qy = np.sign(qxy) * np.sqrt(rad)
        qy = np.where(rad < 0, np.nan, qy)
        # k_f in sample frame, then lab
        kfs_x = k * np.cos(a) + qx
        kfs_y = qy
        kfs_z = -k * np.sin(a) + qz
        ex, ey, ez = self._axes()
        KX = kfs_x * ex[0] + kfs_y * ey[0] + kfs_z * ez[0]
        KY = kfs_x * ex[1] + kfs_y * ey[1] + kfs_z * ez[1]
        KZ = kfs_x * ex[2] + kfs_y * ey[2] + kfs_z * ez[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            col = self.x0 + self.sdd_mm * KY / KX / self.pixel_mm
            row = self.y0 - self.sdd_mm * KZ / KX / self.pixel_mm
        bad = ~(KX > 0)
        col = np.where(bad, np.nan, col)
        row = np.where(bad, np.nan, row)
        return col, row

    def q_maps(self, shape):
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
        return self.pixel_to_q(cols, rows)
