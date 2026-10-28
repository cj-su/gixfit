"""Assignment of observed peaks to predicted reflections and least-squares refinement of the unit cell.

Model parameters: any subset of (a, b, c, alpha, beta, gamma), optional fibre-axis tilt (2 angles) and optional
detector nuisance parameters (x0, y0, sdd_mm).  Observed peaks are stored in PIXELS, so refining the geometry
re-maps them consistently.

Residuals per matched pair (polar coordinates, because a wrong cell moves peaks mainly in |q| while texture /
mosaic errors move them in chi):
    r_q   = (|q|_obs - |q|_pred) / sigma_q ,   sigma_q   = sqrt(err_meas^2 + q_floor^2)
    r_chi = (chi_obs - chi_pred) / sigma_chi, sigma_chi = sqrt(err_meas^2 + chi_floor^2)
plus optional Gaussian priors that keep the cell near the starting (CIF) values (stabilises fits with few peaks).

Assignment = Hungarian (one-to-one) on the gated cost (dq/tol_q)^2 + (dchi/tol_chi)^2 (+ a weak intensity prior),
repeated with a shrinking tolerance schedule and re-refinement until the assignment is stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import spearmanr

from .crystal import Cell, Structure, PARAM_NAMES
from .geometry import Geometry
from .pattern import (hkl_sphere, fiber_qcoords, fiber_pattern, group_reflections, hkl_str, wedge_project,
                      polarization_factor)
from .peaks import PeakTable


@dataclass
class FitConfig:
    qmax: float = 2.0
    q_range: tuple = (0.3, 2.0)                 # observed peaks used
    mosaic_deg: float = 3.0
    min_rel_intensity: float = 0.002            # predicted groups weaker than this (rel. to strongest) ignored
    n_strong: int = 20                          # the n strongest accessible predicted peaks "should be seen"
    merge_tol: float = 0.008                    # A^-1, reflections closer than this = one predicted peak
    tol_schedule: tuple = ((0.05, 10.0), (0.03, 6.0), (0.015, 4.0))   # (tol_q A^-1, tol_chi deg)
    q_floor: float = 0.002
    chi_floor_deg: float = 0.7
    intensity_prior: float = 0.05               # cost added per decade below the strongest reflection
    free_cell: tuple = PARAM_NAMES
    max_strain: float = 0.06                    # hard bound, fraction of each length
    max_angle_dev: float = 5.0                  # hard bound, deg
    prior_strain: float | None = 0.03           # soft prior widths (None = no prior)
    prior_angle: float | None = 3.0
    free_geom: tuple = ()                       # e.g. ("y0",) or ("x0", "y0", "sdd_mm")
    geom_bounds: dict = field(default_factory=lambda: {"x0": 6.0, "y0": 30.0, "sdd_mm": 10.0})
    free_tilt: bool = False
    max_tilt_deg: float = 3.0
    max_iter: int = 6
    polarization: float | None = None           # horizontal polarisation fraction p_h (None = no polarisation factor)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("q_range", "free_cell", "free_geom"):
            if k in d:
                d[k] = tuple(d[k])
        if "tol_schedule" in d:
            d["tol_schedule"] = tuple(tuple(t) for t in d["tol_schedule"])
        return cls(**d)


@dataclass
class FitResult:
    name: str
    contact: tuple
    cell0: Cell
    cell: Cell
    cell_err: np.ndarray
    geom0: Geometry
    geom: Geometry
    geom_err: dict
    tilt: tuple
    pairs: list                 # list of dicts (one per matched observed peak)
    metrics: dict
    unmatched_obs: np.ndarray
    missing_strong: list        # labels of strong predicted peaks with no observed peak
    refl: object = None
    groups: list = None
    group_pos: tuple = None
    accessible: np.ndarray = None
    null: dict = None
    mosaic_deg: float = 3.0


def on_axis(groups) -> np.ndarray:
    """True for groups on the q_z axis (side 0).  Their wedge-projected spot can appear on EITHER edge of the
    missing wedge, so they are matched against |chi_obs| (see fold_chi)."""
    return np.array([g.side == 0 for g in groups], bool)


def fold_chi(chi_obs, side0):
    return np.where(side0, np.abs(chi_obs), chi_obs)


class Problem:
    def __init__(self, struct: Structure, contact, geom: Geometry, peaks: PeakTable, mask, cfg: FitConfig,
                 name=""):
        self.s0 = struct
        self.contact = tuple(int(v) for v in contact)
        self.g0 = geom
        self.peaks = peaks
        self.mask = mask
        self.cfg = cfg
        self.name = name or struct.name
        # parameter layout
        names, start, lo, hi, prior = [], [], [], [], []
        p0 = struct.cell.array()
        for i, n in enumerate(PARAM_NAMES):
            if n in cfg.free_cell:
                d = cfg.max_strain * p0[i] if i < 3 else cfg.max_angle_dev
                ps = (cfg.prior_strain * p0[i] if i < 3 else cfg.prior_angle) if (
                    cfg.prior_strain if i < 3 else cfg.prior_angle) else None
                names.append(n); start.append(p0[i]); lo.append(p0[i] - d); hi.append(p0[i] + d); prior.append(ps)
        if cfg.free_tilt:
            for n in ("tilt1", "tilt2"):
                names.append(n); start.append(0.0); lo.append(-cfg.max_tilt_deg); hi.append(cfg.max_tilt_deg)
                prior.append(None)
        for n in cfg.free_geom:
            v = getattr(geom, n)
            b = cfg.geom_bounds.get(n, 10.0)
            names.append(n); start.append(v); lo.append(v - b); hi.append(v + b); prior.append(None)
        self.names, self.start = names, np.array(start, float)
        self.lo, self.hi, self.prior = np.array(lo, float), np.array(hi, float), prior
        # fixed reflection set (from the starting cell, a bit beyond qmax so refinement cannot lose any)
        self.hkl = hkl_sphere(struct.cell, cfg.qmax * (1 + cfg.max_strain) + 0.05)
        ref = fiber_pattern(struct, self.contact, qmax=cfg.qmax * (1 + cfg.max_strain) + 0.05,
                            mosaic_deg=cfg.mosaic_deg)
        if cfg.polarization is not None:                # optional polarisation factor, renormalised as in fiber_pattern
            ref.I = ref.I * polarization_factor(ref.qxy, ref.qz, geom, cfg.polarization)
            u = ref.I[ref.side >= 0]
            r10 = np.sort(u)[-min(10, len(u))] if len(u) else 1.0
            ref.I = ref.I / r10 if r10 > 0 else ref.I
        self.refl = ref
        self.side = ref.side
        self.rhkl = ref.hkl

    # ------------------------------------------------------------------ parameter handling
    def unpack(self, p):
        cell = self.s0.cell.array().copy()
        geom_kw, tilt = {}, [0.0, 0.0]
        for n, v in zip(self.names, p):
            if n in PARAM_NAMES:
                cell[PARAM_NAMES.index(n)] = v
            elif n == "tilt1":
                tilt[0] = v
            elif n == "tilt2":
                tilt[1] = v
            else:
                geom_kw[n] = v
        return Cell.from_array(cell), self.g0.with_(**geom_kw), tuple(tilt)

    # ------------------------------------------------------------------ observed / predicted coordinates
    def observed(self, geom):
        pk = self.peaks
        qxy, qz = geom.pixel_to_q(pk.col, pk.row)
        # numerical Jacobian for error propagation
        e = 0.5
        a1, b1 = geom.pixel_to_q(pk.col + e, pk.row)
        a2, b2 = geom.pixel_to_q(pk.col, pk.row + e)
        q = np.hypot(qxy, qz)
        chi = np.degrees(np.arctan2(qxy, qz))
        dq_dc = (np.hypot(a1, b1) - q) / e
        dq_dr = (np.hypot(a2, b2) - q) / e
        dchi_dc = (np.degrees(np.arctan2(a1, b1)) - chi) / e
        dchi_dr = (np.degrees(np.arctan2(a2, b2)) - chi) / e
        sq = np.hypot(dq_dc * pk.ecol, dq_dr * pk.erow)
        schi = np.hypot(dchi_dc * pk.ecol, dchi_dr * pk.erow)
        return qxy, qz, q, chi, sq, schi

    def predicted(self, cell, tilt, geom):
        qxy, qz, q = fiber_qcoords(cell, self.rhkl, self.contact, tilt)
        qxy = qxy * np.where(self.side == 0, 1, self.side)
        qxy, qz = wedge_project(qxy, qz, geom, self.cfg.mosaic_deg)
        return qxy, qz, q

    def accessible(self, geom, qxy, qz):
        col, row = geom.q_to_pixel(qxy, qz)
        ok = np.isfinite(col) & np.isfinite(row)
        ci = np.clip(np.nan_to_num(np.round(col), nan=0).astype(int), 0, self.mask.shape[1] - 1)
        ri = np.clip(np.nan_to_num(np.round(row), nan=0).astype(int), 0, self.mask.shape[0] - 1)
        ok &= (col >= 0) & (col < self.mask.shape[1]) & (row >= 0) & (row < self.mask.shape[0])
        ok &= self.mask[ri, ci]
        q = np.hypot(qxy, qz)
        ok &= (q >= self.cfg.q_range[0]) & (q <= self.cfg.q_range[1])
        return ok

    def groups_at(self, p):
        cell, geom, tilt = self.unpack(p)
        qxy, qz, q = self.predicted(cell, tilt, geom)
        acc = self.accessible(geom, qxy, qz)
        r = self.refl
        r.qxy, r.qz, r.q = qxy, qz, q
        groups = group_reflections(r, self.cfg.merge_tol, self.cfg.min_rel_intensity, acc)
        return groups, acc

    def group_coords(self, groups, cell, tilt, geom):
        """intensity-weighted mean position of each group (vectorised: this sits inside every residual call)."""
        if not len(groups):
            return np.zeros(0), np.zeros(0)
        qxy, qz, _ = self.predicted(cell, tilt, geom)
        idx = np.concatenate([g.members for g in groups])
        gid = np.repeat(np.arange(len(groups)), [len(g.members) for g in groups])
        w = self.refl.I[idx]
        ws = np.bincount(gid, w, len(groups))
        gx = np.bincount(gid, w * qxy[idx], len(groups)) / ws
        gz = np.bincount(gid, w * qz[idx], len(groups)) / ws
        return gx, gz

    # ------------------------------------------------------------------ assignment
    def used_obs(self, geom):
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        return np.nonzero((q >= self.cfg.q_range[0]) & (q <= self.cfg.q_range[1]))[0]

    def assign(self, p, tol_q, tol_chi, groups=None):
        cell, geom, tilt = self.unpack(p)
        if groups is None:
            groups, _ = self.groups_at(p)
        if not groups:
            return [], groups
        gx, gz = self.group_coords(groups, cell, tilt, geom)
        gq, gchi = np.hypot(gx, gz), np.degrees(np.arctan2(gx, gz))
        gI = np.array([self.refl.I[g.members].sum() for g in groups])
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        use = self.used_obs(geom)
        co = fold_chi(chi[use, None], on_axis(groups)[None])
        C = ((q[use, None] - gq[None]) / tol_q) ** 2 + ((co - gchi[None]) / tol_chi) ** 2
        gate = C <= 1.0
        C = C + self.cfg.intensity_prior * (-np.log10(np.clip(gI, 1e-6, 1)))[None]
        big = 1e6
        Cg = np.where(gate, C, big)
        r, c = linear_sum_assignment(Cg)
        pairs = [(int(use[i]), int(j)) for i, j in zip(r, c) if Cg[i, j] < big]
        return pairs, groups

    # ------------------------------------------------------------------ residuals
    def residuals(self, p, pairs, groups, with_prior=True):
        cell, geom, tilt = self.unpack(p)
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        gsel = [groups[j] for _, j in pairs]
        gx, gz = self.group_coords(gsel, cell, tilt, geom)
        gq, gchi = np.hypot(gx, gz), np.degrees(np.arctan2(gx, gz))
        oi = np.array([i for i, _ in pairs])
        sig_q = np.hypot(sq[oi], self.cfg.q_floor)
        sig_c = np.hypot(schi[oi], self.cfg.chi_floor_deg)
        r = [(q[oi] - gq) / sig_q, (fold_chi(chi[oi], on_axis(gsel)) - gchi) / sig_c]
        if with_prior:
            pr = [(p[k] - self.start[k]) / s for k, s in enumerate(self.prior) if s]
            if pr:
                r.append(np.array(pr))
        return np.concatenate(r)

    def _lsq(self, p, pairs, groups):
        res = least_squares(self.residuals, p, bounds=(self.lo, self.hi), args=(pairs, groups),
                            x_scale="jac", method="trf")
        return res

    # ------------------------------------------------------------------ main loop
    def _refine(self, p, guard):
        """iterate assign -> least squares over the tolerance schedule.  guard=True accepts a step only if it
        does not lower the sqrt(area)-weighted matched score (stops least squares from trading the strongest
        peak for several weak wrong ones; can stop early, hence both variants are run)."""
        lab = lambda pp, gg: sorted((i, gg[j].label) for i, j in pp)
        for tol_q, tol_chi in self.cfg.tol_schedule:
            for _ in range(self.cfg.max_iter):
                groups, _ = self.groups_at(p)
                pairs, _ = self.assign(p, tol_q, tol_chi, groups)
                if len(pairs) < 2:
                    break
                p_new = self._lsq(p, pairs, groups).x
                groups_new, _ = self.groups_at(p_new)
                pairs_new, _ = self.assign(p_new, tol_q, tol_chi, groups_new)
                if guard and (self._wscore(p_new, pairs_new, groups_new, tol_q, tol_chi)
                              < self._wscore(p, pairs, groups, tol_q, tol_chi) * (1 - 1e-9)):
                    break
                p = p_new
                if lab(pairs_new, groups_new) == lab(pairs, groups):
                    break
        return p

    def _wscore(self, p, pairs, groups, tol_q, tol_chi):
        """sum over matched peaks of sqrt(area) * (1 - normalised distance^2): rewards indexing strong peaks
        AND indexing them precisely (a sloppy match of a spurious spot adds little)."""
        if not pairs:
            return 0.0
        cell, geom, tilt = self.unpack(p)
        qxy, qz, q, chi, _, _ = self.observed(geom)
        gsel = [groups[j] for _, j in pairs]
        gx, gz = self.group_coords(gsel, cell, tilt, geom)
        oi = np.array([i for i, _ in pairs])
        co = fold_chi(chi[oi], on_axis(gsel))
        c = ((q[oi] - np.hypot(gx, gz)) / tol_q) ** 2 + ((co - np.degrees(np.arctan2(gx, gz))) / tol_chi) ** 2
        w = np.sqrt(np.clip(self.peaks.area[oi], 0, None))
        return float(np.sum(w * np.clip(1 - c, 0, None)))

    def run(self, p=None) -> FitResult:
        """multi-start: plain refinement, guarded refinement, unrefined start; keep the one that indexes the
        most (sqrt-area weighted) observed peaks at the final tolerance (ties -> lower chi^2)."""
        p0 = self.start.copy() if p is None else p.copy()
        tol_q, tol_chi = self.cfg.tol_schedule[-1]
        best = None
        for cand in (self._refine(p0.copy(), False), self._refine(p0.copy(), True), p0):
            groups, acc = self.groups_at(cand)
            pairs, _ = self.assign(cand, tol_q, tol_chi, groups)
            chi2 = float(np.sum(self.residuals(cand, pairs, groups, False) ** 2)) if pairs else np.inf
            key = (round(self._wscore(cand, pairs, groups, tol_q, tol_chi), 6), -chi2)
            if best is None or key > best[0]:
                best = (key, cand, pairs, groups, acc)
        _, p, pairs, groups, acc = best
        return self._result(p, pairs, groups, acc)

    # ------------------------------------------------------------------ result / metrics
    def _result(self, p, pairs, groups, acc) -> FitResult:
        cell, geom, tilt = self.unpack(p)
        cfg = self.cfg
        npar = len(p)
        cell_err = np.full(6, np.nan)
        geom_err = {}
        chi2r = np.nan
        if len(pairs) >= 1:
            r_data = self.residuals(p, pairs, groups, with_prior=False)
            ndata = len(r_data)
            dof = ndata - npar + sum(1 for s in self.prior if s)
            chi2r = float(np.sum(r_data ** 2) / max(dof, 1))
            # Jacobian (with priors) -> covariance
            eps = np.maximum(1e-6, 1e-6 * np.abs(p))
            r0 = self.residuals(p, pairs, groups)
            J = np.stack([(self.residuals(p + e, pairs, groups) - r0) / e[k]
                          for k, e in enumerate(np.diag(eps))], 1)
            cov = np.linalg.pinv(J.T @ J) * max(chi2r, 1.0)
            err = np.sqrt(np.clip(np.diag(cov), 0, None))
            for k, n in enumerate(self.names):
                if n in PARAM_NAMES:
                    cell_err[PARAM_NAMES.index(n)] = err[k]
                elif n in ("x0", "y0", "sdd_mm"):
                    geom_err[n] = float(err[k])
            corr = cov / np.outer(err, err) if np.all(err > 0) else None
        else:
            corr = None
        # pair table
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        _, _, q_cif0, chi_cif0, _, _ = self.observed(self.g0)
        gsel = [groups[j] for _, j in pairs]
        gx, gz = self.group_coords(gsel, cell, tilt, geom)
        gx0, gz0 = self.group_coords(gsel, self.s0.cell, (0.0, 0.0), self.g0)
        rows = []
        for k, (i, j) in enumerate(pairs):
            g = groups[j]
            gq, gq0 = np.hypot(gx[k], gz[k]), np.hypot(gx0[k], gz0[k])
            sgn = -1.0 if (g.side == 0 and chi[i] < 0) else 1.0      # on-axis group seen on the -q_xy edge
            rows.append(dict(
                obs=i, label=g.label, hkl=[tuple(int(v) for v in self.rhkl[m]) for m in g.members],
                side=g.side, I_pred=float(self.refl.I[g.members].sum()), area_obs=float(self.peaks.area[i]),
                snr=float(self.peaks.snr[i]), col=float(self.peaks.col[i]), row=float(self.peaks.row[i]),
                q_obs=float(q[i]), q_err=float(sq[i]), chi_obs=float(chi[i]), chi_err=float(schi[i]),
                q_obs_start_geom=float(q_cif0[i]),
                q_cif=float(gq0), chi_cif=float(sgn * np.degrees(np.arctan2(gx0[k], gz0[k]))),
                q_fit=float(gq), chi_fit=float(sgn * np.degrees(np.arctan2(gx[k], gz[k]))),
                dq_cif_pct=float(100 * (q_cif0[i] - gq0) / gq0),
                dq_fit_pct=float(100 * (q[i] - gq) / gq),
                d_obs=float(2 * np.pi / q[i]), d_cif=float(2 * np.pi / gq0), d_fit=float(2 * np.pi / gq)))
        use = self.used_obs(geom)
        matched = {i for i, _ in pairs}
        unmatched = np.array([i for i in use if i not in matched], int)
        w = np.sqrt(np.clip(self.peaks.area, 0, None))
        frac_w = float(w[list(matched)].sum() / w[use].sum()) if len(use) else 0.0
        # strong predicted peaks with no observed peak
        gI = np.array([self.refl.I[g.members].sum() for g in groups]) if groups else np.zeros(0)
        Imax = 1.0                     # intensities are already normalised (10th strongest reflection = 1)
        matched_g = {j for _, j in pairs}
        strong = [int(j) for j in np.argsort(-gI)[:cfg.n_strong]]
        miss = [j for j in strong if j not in matched_g]
        miss_w = float(len(miss) / len(strong)) if strong else 0.0
        rho = np.nan
        if len(rows) >= 4:
            rho = float(spearmanr([r["I_pred"] for r in rows], [r["area_obs"] for r in rows]).correlation)
        p0 = self.s0.cell.array()
        pf = cell.array()
        strain = np.where(np.arange(6) < 3, 100 * (pf - p0) / p0, pf - p0)
        dqf = np.array([r["dq_fit_pct"] for r in rows]) if rows else np.zeros(0)
        dqc = np.array([r["dq_cif_pct"] for r in rows]) if rows else np.zeros(0)
        metrics = dict(
            n_obs=int(len(use)), n_match=int(len(pairs)), n_params=int(npar),
            frac_match=float(len(pairs) / max(len(use), 1)), frac_match_w=frac_w,
            n_strong_pred=len(strong), n_strong_missing=len(miss), missing_strong_w=miss_w,
            fom=frac_w * (1 - miss_w),
            chi2_red=chi2r, rms_dq_cif_pct=float(np.sqrt(np.mean(dqc ** 2))) if len(dqc) else np.nan,
            rms_dq_fit_pct=float(np.sqrt(np.mean(dqf ** 2))) if len(dqf) else np.nan,
            spearman_I=rho, strain=strain.tolist(), corr=corr.tolist() if corr is not None else None,
            param_names=list(self.names))
        gx_all, gz_all = self.group_coords(groups, cell, tilt, geom) if groups else (np.zeros(0), np.zeros(0))
        return FitResult(self.name, self.contact, self.s0.cell, cell, cell_err, self.g0, geom, geom_err, tilt,
                         rows, metrics, unmatched, [groups[j].label for j in miss], self.refl, groups,
                         (gx_all, gz_all, gI / Imax if len(gI) else gI), acc, mosaic_deg=cfg.mosaic_deg)


# ---------------------------------------------------------------------- null (chance-level) tests
def observable(geom, mask, qxy, qz):
    """(qxy, qz) lands on the detector, on an unmasked pixel (outside the missing wedge by construction of the map)."""
    col, row = geom.q_to_pixel(qxy, qz)
    ok = np.isfinite(col) & np.isfinite(row)
    ci = np.clip(np.nan_to_num(np.round(col), nan=0).astype(int), 0, mask.shape[1] - 1)
    ri = np.clip(np.nan_to_num(np.round(row), nan=0).astype(int), 0, mask.shape[0] - 1)
    return ok & (col >= 0) & (col < mask.shape[1]) & (row >= 0) & (row < mask.shape[0]) & mask[ri, ci]


def observable_permutation(q, chi, geom, mask, rng, max_pass=400):
    """random reassignment of the polar angles that keeps every spot observable at its own |q|: a uniform random
    permutation, then randomised swaps of each unobservable spot with a partner for which both positions become
    observable (up to max_pass passes); spots that cannot be placed keep their own angle.  Not an exactly uniform
    draw from all observable permutations."""
    Q, C = np.meshgrid(q, np.radians(chi), indexing="ij")
    A = observable(geom, mask, (Q * np.sin(C)).ravel(), (Q * np.cos(C)).ravel()).reshape(Q.shape)
    n = len(q)
    p = rng.permutation(n)
    for _ in range(max_pass):
        bad = np.nonzero(~A[np.arange(n), p])[0]
        if not len(bad):
            break
        for i in rng.permutation(bad):
            if A[i, p[i]]:
                continue
            js = np.nonzero(A[i, p] & A[np.arange(n), p[i]])[0]
            if len(js):
                j = rng.choice(js)
                p[i], p[j] = p[j], p[i]
    bad = ~A[np.arange(n), p]
    p[bad] = np.nonzero(bad)[0]
    return p


def surrogate_peaks(prob: Problem, rng, mode="chi"):
    """observed peaks with scrambled information, mapped back to pixels with the starting geometry.
    mode 'chi'     : keep every |q|, reassign chi among peaks so that every spot stays observable
                     (observable_permutation)                     -> tests the 2D/orientation information.
    mode 'chi_free': unrestricted permutation of chi (versions before the 2026 revision; spots can land where
                     nothing is observable, which inflates the score).
    mode 'q'       : keep chi, multiply each |q| by an independent random factor in [0.97, 1.03] -> tests |q|
                     information; jittered spots that become unobservable are kept.
    mode 'q_obs'   : as 'q', but a jitter that would make a spot unobservable is redrawn (up to 50 times; then the
                     spot keeps its |q|).
    """
    g = prob.g0
    pk = prob.peaks
    qxy, qz, q, chi, sq, schi = prob.observed(g)
    if mode == "chi":
        chi2 = chi[observable_permutation(q, chi, g, prob.mask, rng)]
        q2 = q
    elif mode == "chi_free":
        chi2 = rng.permutation(chi)
        q2 = q
    elif mode == "q_obs":
        chi2 = chi
        f = rng.uniform(0.97, 1.03, len(q))
        c = np.radians(chi)
        for _ in range(50):
            bad = ~observable(g, prob.mask, q * f * np.sin(c), q * f * np.cos(c))
            if not bad.any():
                break
            f[bad] = rng.uniform(0.97, 1.03, bad.sum())
        f[~observable(g, prob.mask, q * f * np.sin(c), q * f * np.cos(c))] = 1.0
        q2 = q * f
    else:
        chi2 = chi
        q2 = q * rng.uniform(0.97, 1.03, len(q))
    nx, nz = q2 * np.sin(np.radians(chi2)), q2 * np.cos(np.radians(chi2))
    col, row = g.q_to_pixel(nx, nz)
    ok = np.isfinite(col) & np.isfinite(row)
    d = pk.to_dict()
    d = {k: v[ok] for k, v in d.items()}
    d["col"], d["row"] = col[ok], row[ok]
    return PeakTable(**d)


def null_test(prob: Problem, real: FitResult, n=20, modes=("chi", "q"), seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for mode in modes:
        foms, fr = [], []
        for _ in range(n):
            sp = surrogate_peaks(prob, rng, mode)
            p2 = Problem(prob.s0, prob.contact, prob.g0, sp, prob.mask, prob.cfg, prob.name)
            r = p2.run()
            foms.append(r.metrics["fom"]); fr.append(r.metrics["frac_match_w"])
        foms = np.array(foms)
        sd = foms.std(ddof=1) if n > 1 else np.nan
        out[mode] = dict(fom_mean=float(foms.mean()), fom_sd=float(sd),
                         z=float((real.metrics["fom"] - foms.mean()) / sd) if sd > 0 else np.nan,
                         p=float((np.sum(foms >= real.metrics["fom"]) + 1) / (n + 1)),
                         frac_w_mean=float(np.mean(fr)))
    return out
