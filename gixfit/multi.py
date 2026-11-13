"""Multi-component fits: one pattern explained by SEVERAL fibre textures (contact planes) and/or SEVERAL phases.

Model
-----
* phases      : candidate structures; each phase present in the model has ONE cell, refined jointly and shared by
                all of its textures.
* components  : (phase, contact plane) pairs.  Every component predicts its own fibre pattern.
* assignment  : one Hungarian problem over the union of all components' predicted peaks, so every observed
                peak is indexed by at most one (phase, orientation) group.
* refinement  : residuals of all matched peaks together (same polar residuals / priors / multi-start guard as the
                single-texture fit in fit.py).

Greedy decomposition
--------------------
k = 1 .. max_components:
  1. cheap pre-score of every candidate (phase, plane) on the peaks NOT yet indexed (current cell, no refinement);
  2. the best ``prescreen_top`` candidates are each added to the current model and the JOINT model is refined;
  3. the candidate giving the highest joint figure of merit is kept.
The figure of merit is the single-texture one generalised: (sqrt-area-weighted fraction of observed peaks indexed)
x (1 - fraction of every component's n_strong strongest predicted peaks that are absent).  Adding a texture that
predicts many absent peaks therefore costs fom.

Chance level: surrogate data (chi-shuffled, |q|-jittered; fit.surrogate_peaks) go through the SAME greedy +
joint pipeline, so the flexibility of choosing k textures out of ~50 planes is included in the null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import spearmanr

from .crystal import Cell, Structure, PARAM_NAMES
from .geometry import Geometry
from .pattern import group_reflections, low_index_planes
from .peaks import PeakTable
from .fit import FitConfig, Problem, surrogate_peaks, on_axis, fold_chi, match_spots


@dataclass
class MultiConfig:
    max_components: int = 3
    planes_maxidx: int = 2           # contact planes scanned: all (h k l) with |indices| <= this
    prescreen_top: int = 10          # candidates per greedy step (k >= 2) that get a full joint refinement
    prescreen_top_first: int = 0     # same for k = 1 (0 = every candidate: the pre-score at the unrefined CIF cell
                                     # is too noisy to choose the first texture when the cell is a few % off)
    min_new_match: int = 3           # a new component must index at least this many still-unindexed peaks
    n_null: int = 10
    null_modes: tuple = ("chi", "q")
    workers: int = 0                 # parallel processes for the null tests (0 = all cores - 1, 1 = serial)
    seed: int = 0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "null_modes" in d:
            d["null_modes"] = tuple(d["null_modes"])
        return cls(**d)


@dataclass
class MultiResult:
    phase_names: list
    components: list            # [(phase index, contact)]
    cells0: list                # CIF cells per phase
    cells: list                 # refined cells per phase (None if the phase is not used)
    cell_errs: list
    geom0: Geometry
    geom: Geometry
    geom_err: dict
    pairs: list                 # rows (dict) incl. "component"
    metrics: dict
    comp_metrics: list          # one dict per component
    unmatched_obs: np.ndarray
    groups: list = None         # per component
    group_pos: list = None      # per component (gx, gz, gI)
    probs: list = None          # helper Problems (reflection lists)
    mosaic_deg: float = 3.0
    history: list = None        # fom after k = 1..K components (greedy)
    null: dict = None

    def label(self):
        return " + ".join(f"{self.phase_names[p]}{tuple(c)}" for p, c in self.components)


class MultiProblem:
    def __init__(self, phases, components, geom: Geometry, peaks: PeakTable, mask, cfg: FitConfig,
                 start_cells=None, start_geom=None):
        self.phases = phases
        self.components = [(int(p), tuple(int(v) for v in c)) for p, c in components]
        self.g0 = geom                                   # reference (start) geometry for priors / dq_cif
        self.gs = start_geom or geom
        self.peaks, self.mask, self.cfg = peaks, mask, cfg
        self.probs = [Problem(phases[p], c, geom, peaks, mask, cfg) for p, c in self.components]
        self.used_phases = sorted({p for p, _ in self.components})
        start_cells = start_cells or [ph.cell for ph in phases]
        names, start, lo, hi, prior = [], [], [], [], []
        for ph in self.used_phases:
            c0 = phases[ph].cell.array()
            cs = start_cells[ph].array()
            for i, n in enumerate(PARAM_NAMES):
                if n not in cfg.free_cell:
                    continue
                d = cfg.max_strain * c0[i] if i < 3 else cfg.max_angle_dev
                pw = cfg.prior_strain if i < 3 else cfg.prior_angle
                names.append((ph, n)); start.append(np.clip(cs[i], c0[i] - d, c0[i] + d))
                lo.append(c0[i] - d); hi.append(c0[i] + d)
                prior.append(((pw * c0[i] if i < 3 else pw), c0[i]) if pw else None)
        for n in cfg.free_geom:
            v0 = getattr(geom, n)
            b = cfg.geom_bounds.get(n, 10.0)
            names.append((None, n)); start.append(getattr(self.gs, n)); lo.append(v0 - b); hi.append(v0 + b)
            prior.append(None)
        self.names = names
        self.start = np.array(start, float)
        self.lo, self.hi, self.prior = np.array(lo, float), np.array(hi, float), prior
        self.start_cells = start_cells

    # ------------------------------------------------------------------ parameters
    def unpack(self, p):
        cells = [c.array().copy() for c in self.start_cells]
        gkw = {}
        for (ph, n), v in zip(self.names, p):
            if ph is None:
                gkw[n] = v
            else:
                cells[ph][PARAM_NAMES.index(n)] = v
        return [Cell.from_array(c) for c in cells], self.gs.with_(**gkw)

    # ------------------------------------------------------------------ groups / coordinates
    def groups_at(self, p):
        cells, geom = self.unpack(p)
        out, accs = [], []
        for (ph, _), P in zip(self.components, self.probs):
            qxy, qz, q = P.predicted(cells[ph], (0.0, 0.0), geom)
            acc = P.accessible(geom, qxy, qz)
            r = P.refl
            r.qxy, r.qz, r.q = qxy, qz, q
            out.append(group_reflections(r, self.cfg.merge_tol, self.cfg.min_rel_intensity, acc))
            accs.append(acc)
        return out, accs

    def _flat(self, p, groups):
        """coordinates / intensities of all components' groups, concatenated."""
        cells, geom = self.unpack(p)
        gx, gz, gI, comp = [], [], [], []
        for c, ((ph, _), P) in enumerate(zip(self.components, self.probs)):
            x, z = P.group_coords(groups[c], cells[ph], (0.0, 0.0), geom)
            gx.append(x); gz.append(z)
            gI.append(np.array([P.refl.I[g.members].sum() for g in groups[c]]))
            comp.append(np.full(len(groups[c]), c))
        cat = lambda a: np.concatenate(a) if a else np.zeros(0)
        return cat(gx), cat(gz), cat(gI), cat(comp).astype(int)

    def observed(self, geom):
        return self.probs[0].observed(geom)

    def used_obs(self, geom):
        return self.probs[0].used_obs(geom)

    def _split(self, groups):
        off = np.cumsum([0] + [len(g) for g in groups])
        return off

    # ------------------------------------------------------------------ assignment
    def assign(self, p, tol_q, tol_chi, groups):
        cells, geom = self.unpack(p)
        gx, gz, gI, _ = self._flat(p, groups)
        if not len(gx):
            return []
        gq, gchi = np.hypot(gx, gz), np.degrees(np.arctan2(gx, gz))
        qxy, qz, q, chi, _, _ = self.observed(geom)
        use = self.used_obs(geom)
        side0 = np.concatenate([on_axis(g) for g in groups])   # on-axis groups: one spot per wedge edge
        m = match_spots(q[use], chi[use], gq, gchi, gI, side0, tol_q, tol_chi, self.cfg.intensity_prior)
        return [(int(use[i]), j) for i, j in m]

    def _pair_coords(self, p, pairs, groups):
        """predicted (q, chi) of the matched groups, one entry per pair.  For on-axis groups chi takes the sign of
        the observed spot (the wedge-projected spot can sit on either edge)."""
        cells, geom = self.unpack(p)
        off = self._split(groups)
        gq = np.zeros(len(pairs)); gc = np.zeros(len(pairs)); s0 = np.zeros(len(pairs), bool)
        flat = np.array([j for _, j in pairs], int)
        comp = np.searchsorted(off, flat, side="right") - 1
        for c, ((ph, _), P) in enumerate(zip(self.components, self.probs)):
            k = np.nonzero(comp == c)[0]
            if not len(k):
                continue
            gsel = [groups[c][flat[i] - off[c]] for i in k]
            x, z = P.group_coords(gsel, cells[ph], (0.0, 0.0), geom)
            gq[k] = np.hypot(x, z); gc[k] = np.degrees(np.arctan2(x, z)); s0[k] = on_axis(gsel)
        if s0.any():
            chi = self.observed(geom)[3]
            oi = np.array([i for i, _ in pairs], int)
            gc = np.where(s0 & (chi[oi] < 0), -gc, gc)
        return gq, gc, comp

    def residuals(self, p, pairs, groups, with_prior=True):
        cells, geom = self.unpack(p)
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        gq, gc, _ = self._pair_coords(p, pairs, groups)
        oi = np.array([i for i, _ in pairs], int)
        r = [(q[oi] - gq) / np.hypot(sq[oi], self.cfg.q_floor),
             (chi[oi] - gc) / np.hypot(schi[oi], self.cfg.chi_floor_deg)]
        if with_prior:
            pr = [(p[k] - pv[1]) / pv[0] for k, pv in enumerate(self.prior) if pv]
            if pr:
                r.append(np.array(pr))
        return np.concatenate(r)

    def _wscore(self, p, pairs, groups, tol_q, tol_chi):
        if not pairs:
            return 0.0
        cells, geom = self.unpack(p)
        qxy, qz, q, chi, _, _ = self.observed(geom)
        gq, gc, _ = self._pair_coords(p, pairs, groups)
        oi = np.array([i for i, _ in pairs], int)
        c = ((q[oi] - gq) / tol_q) ** 2 + ((chi[oi] - gc) / tol_chi) ** 2
        w = np.sqrt(np.clip(self.peaks.area[oi], 0, None))
        return float(np.sum(w * np.clip(1 - c, 0, None)))

    # ------------------------------------------------------------------ refinement
    def _refine(self, p, guard):
        for tol_q, tol_chi in self.cfg.tol_schedule:
            for _ in range(self.cfg.max_iter):
                groups, _ = self.groups_at(p)
                pairs = self.assign(p, tol_q, tol_chi, groups)
                if len(pairs) < 2 or len(p) == 0:
                    break
                p_new = least_squares(self.residuals, p, bounds=(self.lo, self.hi), args=(pairs, groups),
                                      x_scale="jac", method="trf").x
                groups_new, _ = self.groups_at(p_new)
                pairs_new = self.assign(p_new, tol_q, tol_chi, groups_new)
                if guard and (self._wscore(p_new, pairs_new, groups_new, tol_q, tol_chi)
                              < self._wscore(p, pairs, groups, tol_q, tol_chi) * (1 - 1e-9)):
                    break
                same = sorted(pairs_new) == sorted(pairs)
                p = p_new
                if same:
                    break
        return p

    def run(self) -> MultiResult:
        p0 = self.start.copy()
        tol_q, tol_chi = self.cfg.tol_schedule[-1]
        best = None
        cands = (p0,) if len(p0) == 0 else (self._refine(p0.copy(), False), self._refine(p0.copy(), True), p0)
        for cand in cands:
            groups, accs = self.groups_at(cand)
            pairs = self.assign(cand, tol_q, tol_chi, groups)
            chi2 = float(np.sum(self.residuals(cand, pairs, groups, False) ** 2)) if pairs else np.inf
            key = (round(self._wscore(cand, pairs, groups, tol_q, tol_chi), 6), -chi2)
            if best is None or key > best[0]:
                best = (key, cand, pairs, groups)
        _, p, pairs, groups = best
        return self._result(p, pairs, groups)

    # ------------------------------------------------------------------ result
    def _result(self, p, pairs, groups) -> MultiResult:
        cfg = self.cfg
        cells, geom = self.unpack(p)
        npar = len(p)
        errs = [np.full(6, np.nan) for _ in self.phases]
        geom_err, chi2r = {}, np.nan
        if pairs and npar:
            rd = self.residuals(p, pairs, groups, False)
            dof = len(rd) - npar + sum(1 for s in self.prior if s)
            chi2r = float(np.sum(rd ** 2) / max(dof, 1))
            eps = np.maximum(1e-6, 1e-6 * np.abs(p))
            r0 = self.residuals(p, pairs, groups)
            J = np.stack([(self.residuals(p + e, pairs, groups) - r0) / eps[k]
                          for k, e in enumerate(np.diag(eps))], 1)
            err = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J) * max(chi2r, 1.0)), 0, None))
            for k, (ph, n) in enumerate(self.names):
                if ph is None:
                    geom_err[n] = float(err[k])
                else:
                    errs[ph][PARAM_NAMES.index(n)] = err[k]
        elif pairs:
            rd = self.residuals(p, pairs, groups, False)
            chi2r = float(np.mean(rd ** 2))
        qxy, qz, q, chi, sq, schi = self.observed(geom)
        _, _, q_s, _, _, _ = self.observed(self.g0)
        gq, gc, comp = self._pair_coords(p, pairs, groups)
        p_cif = self._cif_vector()
        gq0, gc0, _ = self._pair_coords_cells(p_cif, pairs, groups)
        off = self._split(groups)
        rows = []
        for k, (i, j) in enumerate(pairs):
            c = int(comp[k]); g = groups[c][j - off[c]]; P = self.probs[c]
            rows.append(dict(
                component=c, phase=self.phases[self.components[c][0]].name, contact=str(self.components[c][1]),
                obs=i, group=int(j - off[c]), label=g.label, I_pred=float(P.refl.I[g.members].sum()),
                area_obs=float(self.peaks.area[i]),
                snr=float(self.peaks.snr[i]), col=float(self.peaks.col[i]), row=float(self.peaks.row[i]),
                q_obs=float(q[i]), q_err=float(sq[i]), chi_obs=float(chi[i]), chi_err=float(schi[i]),
                q_cif=float(gq0[k]), chi_cif=float(gc0[k]), q_fit=float(gq[k]), chi_fit=float(gc[k]),
                dq_cif_pct=float(100 * (q_s[i] - gq0[k]) / gq0[k]), dq_fit_pct=float(100 * (q[i] - gq[k]) / gq[k]),
                d_obs=float(2 * np.pi / q[i]), d_cif=float(2 * np.pi / gq0[k]), d_fit=float(2 * np.pi / gq[k])))
        use = self.used_obs(geom)
        w = np.sqrt(np.clip(self.peaks.area, 0, None))
        wsum = w[use].sum() if len(use) else 1.0
        matched = {i for i, _ in pairs}
        unmatched = np.array([i for i in use if i not in matched], int)
        comp_metrics, n_strong_tot, n_miss_tot = [], 0, 0
        group_pos = []
        for c, ((ph, contact), P) in enumerate(zip(self.components, self.probs)):
            gI = np.array([P.refl.I[g.members].sum() for g in groups[c]]) if groups[c] else np.zeros(0)
            mg = {j - off[c] for (_, j), cc in zip(pairs, comp) if cc == c}
            strong = [int(j) for j in np.argsort(-gI)[:cfg.n_strong]]
            miss = [j for j in strong if j not in mg]
            n_strong_tot += len(strong); n_miss_tot += len(miss)
            cr = [r for r in rows if r["component"] == c]
            rho = (float(spearmanr([r["I_pred"] for r in cr], [r["area_obs"] for r in cr]).correlation)
                   if len(cr) >= 4 else np.nan)
            comp_metrics.append(dict(
                component=c, phase=self.phases[ph].name, contact=str(contact), n_match=len(cr),
                frac_w=float(sum(w[r["obs"]] for r in cr) / wsum),
                n_strong_pred=len(strong), n_strong_missing=len(miss), spearman_I=rho,
                missing_labels=[groups[c][j].label for j in miss]))
            gx, gz = P.group_coords(groups[c], cells[ph], (0.0, 0.0), geom)
            group_pos.append((gx, gz, gI))
        frac_w = float(w[list(matched)].sum() / wsum) if len(use) else 0.0
        miss_frac = n_miss_tot / n_strong_tot if n_strong_tot else 0.0
        rho_pool = _pooled_rank_corr(rows)
        dqf = np.array([r["dq_fit_pct"] for r in rows]); dqc = np.array([r["dq_cif_pct"] for r in rows])
        metrics = dict(
            n_components=len(self.components), n_obs=int(len(use)), n_match=len(pairs), n_params=int(npar),
            frac_match=float(len(pairs) / max(len(use), 1)), frac_match_w=frac_w,
            n_strong_pred=n_strong_tot, n_strong_missing=n_miss_tot, missing_strong_frac=miss_frac,
            fom=frac_w * (1 - miss_frac), chi2_red=chi2r,
            rms_dq_cif_pct=float(np.sqrt(np.mean(dqc ** 2))) if len(dqc) else np.nan,
            rms_dq_fit_pct=float(np.sqrt(np.mean(dqf ** 2))) if len(dqf) else np.nan,
            spearman_I_pooled=rho_pool)
        return MultiResult([ph.name for ph in self.phases], list(self.components),
                           [ph.cell for ph in self.phases],
                           [cells[i] if i in self.used_phases else None for i in range(len(self.phases))],
                           errs, self.g0, geom, geom_err, rows, metrics, comp_metrics, unmatched,
                           groups, group_pos, self.probs, cfg.mosaic_deg)

    def _cif_vector(self):
        v = []
        for ph, n in self.names:
            v.append(getattr(self.g0, n) if ph is None else self.phases[ph].cell.array()[PARAM_NAMES.index(n)])
        return np.array(v, float)

    def _pair_coords_cells(self, p, pairs, groups):
        """like _pair_coords, but with the CIF cells and the START geometry (for dq_cif)."""
        save = self.gs, self.start_cells
        self.gs, self.start_cells = self.g0, [ph.cell for ph in self.phases]
        try:
            return self._pair_coords(p, pairs, groups)
        finally:
            self.gs, self.start_cells = save


def _pooled_rank_corr(rows):
    """intensity agreement pooled over components: ranks are taken WITHIN each component (each texture has its
    own unknown volume fraction), scaled to 0..1, then correlated across all matched peaks."""
    a, b = [], []
    for c in {r["component"] for r in rows}:
        cr = [r for r in rows if r["component"] == c]
        if len(cr) < 3:
            continue
        for key, out in (("I_pred", a), ("area_obs", b)):
            v = np.array([r[key] for r in cr])
            out.extend((np.argsort(np.argsort(v)) / (len(v) - 1)).tolist())
    if len(a) < 4:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------- greedy decomposition
def _prescore(st, contact, geom, peaks, mask, cfg, tol):
    P = Problem(st, contact, geom, peaks, mask, cfg)
    groups, _ = P.groups_at(P.start)
    if not groups:
        return 0.0
    pairs, _ = P.assign(P.start, tol[0], tol[1], groups)
    return P._wscore(P.start, pairs, groups, tol[0], tol[1])


def unique_planes(st: Structure, planes, cfg: FitConfig):
    """drop contact planes that are symmetry-equivalent for this structure (identical predicted fibre pattern,
    e.g. (100)/(010)/(001) in a cubic cell): fitting them twice only costs time."""
    from .pattern import fiber_pattern
    seen, out = set(), []
    for pl in planes:
        r = fiber_pattern(st, pl, qmax=cfg.q_range[1], mosaic_deg=cfg.mosaic_deg)
        sel = r.I > 0.01
        sig = hash(tuple(sorted(zip(np.round(np.abs(r.qxy[sel]), 3), np.round(r.qz[sel], 3),
                                    np.round(np.log10(r.I[sel]), 1)))))
        if sig not in seen:
            seen.add(sig)
            out.append(tuple(pl))
    return out


def decompose(phases, geom, peaks, mask, cfg: FitConfig, mcfg: MultiConfig, planes=None, log=None):
    """greedy k = 1..max_components; returns the list of MultiResults (one per k).
    planes: per phase, list of contact planes to consider (default: all with |index| <= planes_maxidx)."""
    planes = planes or [low_index_planes(mcfg.planes_maxidx) for _ in phases]
    planes = [unique_planes(st, pl, cfg) if len(pl) > 1 else list(pl) for st, pl in zip(phases, planes)]
    comps, cells, g_cur = [], [ph.cell for ph in phases], geom
    history, current = [], None
    probe = Problem(phases[0], (0, 0, 1), geom, peaks, mask, cfg)
    used = probe.used_obs(geom)
    for k in range(mcfg.max_components):
        matched = {r["obs"] for r in current.pairs} if current else set()
        remaining = np.array([i for i in used if i not in matched], int)
        sub = peaks.subset(remaining)
        scored = []
        for ph_i, st in enumerate(phases):
            present = any(p == ph_i for p, _ in comps)
            tol = cfg.tol_schedule[1 if present else 0]
            st_cur = st.with_cell(cells[ph_i])
            for pl in planes[ph_i]:
                if (ph_i, tuple(pl)) in comps:
                    continue
                scored.append((_prescore(st_cur, pl, g_cur, sub, mask, cfg, tol), ph_i, tuple(pl)))
        scored.sort(key=lambda t: -t[0])
        ntop = mcfg.prescreen_top if comps else (mcfg.prescreen_top_first or len(scored))
        best = None
        for s, ph_i, pl in scored[:ntop]:
            mp = MultiProblem(phases, comps + [(ph_i, pl)], geom, peaks, mask, cfg, cells, g_cur)
            r = mp.run()
            new = {row["obs"] for row in r.pairs if row["component"] == len(comps)} - matched
            ok = len(new) >= mcfg.min_new_match
            key = (ok, r.metrics["fom"])
            if best is None or key > best[0]:
                best = (key, r, ph_i, pl)
        if best is None or not best[0][0]:
            if log:
                log(f"    k={k + 1}: no candidate indexes >= {mcfg.min_new_match} new peaks -> stop")
            while len(history) < mcfg.max_components:      # keep fom_k defined for every k
                history.append(current)
            break
        _, current, ph_i, pl = best
        comps.append((ph_i, pl))
        cells = [current.cells[i] if current.cells[i] is not None else cells[i] for i in range(len(phases))]
        g_cur = current.geom
        history.append(current)
        if log:
            m = current.metrics
            log(f"    k={k + 1}: + {phases[ph_i].name}{pl}  matched {m['n_match']}/{m['n_obs']}  "
                f"fom {m['fom']:.3f}  missing strong {m['n_strong_missing']}/{m['n_strong_pred']}  "
                f"rho {m['spearman_I_pooled']:.2f}")
    final = history[-1] if history else None
    if final is not None:
        final.history = [h.metrics["fom"] if h else 0.0 for h in history]
    return history


# ---------------------------------------------------------------------- null tests (same pipeline)
def _null_worker(args):
    phases, geom, peaks, mask, cfg, mcfg, planes, mode, seed = args
    rng = np.random.default_rng(seed)
    probe = Problem(phases[0], (0, 0, 1), geom, peaks, mask, cfg)
    sp = surrogate_peaks(probe, rng, mode)
    hist = decompose(phases, geom, sp, mask, cfg, mcfg, planes)
    return [h.metrics["fom"] if h else 0.0 for h in hist]


def null_test_multi(phases, geom, peaks, mask, cfg, mcfg: MultiConfig, history, planes=None, log=None):
    """fom_k of the real data vs the same greedy pipeline on surrogates.  Returns per mode and per k:
    null mean/sd, z, p, and the same for the INCREMENT fom_k - fom_(k-1) (is the k-th texture real?)."""
    from concurrent.futures import ProcessPoolExecutor
    real = np.array([h.metrics["fom"] if h else 0.0 for h in history])
    K = len(real)
    # each worker imports numpy/scipy/pymatgen (~130 MB): cap the default so a busy PC does not run out of memory
    workers = mcfg.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    out = {}
    for mi, mode in enumerate(mcfg.null_modes):
        jobs = [(phases, geom, peaks, mask, cfg, mcfg, planes, mode, mcfg.seed + 1000 * mi + i)
                for i in range(mcfg.n_null)]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                res = list(ex.map(_null_worker, jobs))
        else:
            res = [_null_worker(j) for j in jobs]
        N = np.array([(r + [r[-1]] * (K - len(r)))[:K] for r in res], float)
        inc_real = np.diff(np.concatenate([[0.0], real]))
        inc_null = np.diff(np.concatenate([np.zeros((len(N), 1)), N], 1), axis=1)
        per_k = []
        for k in range(K):
            sd = N[:, k].std(ddof=1) if len(N) > 1 else np.nan
            isd = inc_null[:, k].std(ddof=1) if len(N) > 1 else np.nan
            per_k.append(dict(
                k=k + 1, fom=float(real[k]), null_mean=float(N[:, k].mean()), null_sd=float(sd),
                z=float((real[k] - N[:, k].mean()) / sd) if sd > 0 else np.nan,
                p=float((np.sum(N[:, k] >= real[k]) + 1) / (len(N) + 1)),
                inc=float(inc_real[k]), inc_null_mean=float(inc_null[:, k].mean()), inc_null_sd=float(isd),
                inc_z=float((inc_real[k] - inc_null[:, k].mean()) / isd) if isd > 0 else np.nan,
                inc_p=float((np.sum(inc_null[:, k] >= inc_real[k]) + 1) / (len(N) + 1))))
        out[mode] = per_k
        if log:
            for d in per_k:
                log(f"    null[{mode}] k={d['k']}: fom {d['fom']:.3f} vs {d['null_mean']:.3f}+-{d['null_sd']:.3f} "
                    f"(z={d['z']:.1f}, exceed={d['p']:.3f});  increment {d['inc']:+.3f} vs {d['inc_null_mean']:+.3f}"
                    f"+-{d['inc_null_sd']:.3f} (z={d['inc_z']:.1f})")
    return out
