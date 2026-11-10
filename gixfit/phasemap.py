"""Which crystal does each spot belong to?  Per-spot phase labels for a pattern that contains two (or more)
crystals, e.g. a bilayer "A on B" or a film with two polymorphs.

Input is a MultiResult from multi.decompose / MultiProblem.run: every phase in it has a REFINED cell and one or
more contact planes (fibre textures).  For every detected spot in the fitted q range:

* candidates : every phase that predicts a spot within the final matching tolerance (|q|, chi) of it, using the
               refined cells and geometry of the fit (all textures of a phase are tried);
* assigned   : the phase the joint one-to-one assignment of the fit gave the spot (or none);
* label      : "<A>"              only phase A predicts a spot there;
               "<A> + <B>"        both predict one: the spot is SHARED / ambiguous; the fit gave it to ``assigned``,
                                  but the data alone cannot say which crystal it comes from;
               "unassigned"       no phase predicts a spot there (another phase, substrate, artefact ...).

``share_A`` gives, for an ambiguous spot, the fraction of the predicted intensity contributed by each candidate
phase (relative |F|^2 x Lorentz of the nearest predicted spot), a hint for which crystal dominates it.

Usage (Python)::

    from gixfit.multi import MultiConfig, decompose
    from gixfit.phasemap import phase_map, best_two_phase, write_csv, phase_map_figure
    hist = decompose([A, B], geom, peaks, mask, fit_cfg, MultiConfig(max_components=3))
    res = best_two_phase(hist)
    rows = phase_map(res, peaks)
    write_csv(rows, "phase_map.csv"); phase_map_figure(D, mask, res, peaks, rows, "phase_map.png")

Command line: ``python -m gixfit phases config.toml`` (two or more [[candidates]]; see README).
"""
from __future__ import annotations

import csv
import numpy as np

from .fit import FitConfig, on_axis, fold_chi
from .multi import MultiConfig, MultiProblem, decompose, unique_planes
from .pattern import low_index_planes

UNASSIGNED = "unassigned"
PHASE_COLORS = ["#ff3b30", "#00e5ff", "#7CFC00", "#ff9500", "#bf5af2"]   # A red, B cyan, C green, ...
SHARED_COLOR = "#ffd400"
NONE_COLOR = "#b0b0b0"


def best_two_phase(history):
    """the fitted model from a decompose() history that uses the most phases (ties: highest fom)."""
    hist = [h for h in history if h is not None]
    if not hist:
        raise ValueError("empty fit history")
    nph = lambda h: len({p for p, _ in h.components})
    return max(hist, key=lambda h: (nph(h), h.metrics["fom"]))


def fit_phases(phases, geom, peaks, mask, cfg: FitConfig, mcfg: MultiConfig, planes=None, log=None):
    """multi-phase fit in which EVERY given phase gets at least one texture.

    The greedy decomposition alone can leave a crystal out: a large organic cell predicts so many reflections that
    the first phase fitted absorbs the other crystal's spots, which then no longer count as "new".  Each phase the
    greedy model lacks is added with the contact plane whose own strong spots are best observed
    (frac_w x (1 - missing strong) of that component), and the joint model is refined again so that every spot is
    re-assigned.  Returns (result, history, forced) with ``forced`` = names of the phases that had to be added: their
    presence is assumed by you, not established by the data (check their spot counts and missing strong spots)."""
    planes = planes or [low_index_planes(mcfg.planes_maxidx) for _ in phases]
    planes = [unique_planes(st, pl, cfg) if len(pl) > 1 else list(pl) for st, pl in zip(phases, planes)]
    hist = decompose(phases, geom, peaks, mask, cfg, mcfg, planes, log=log)
    res = best_two_phase(hist)
    forced = []
    for ph_i, st in enumerate(phases):
        if any(p == ph_i for p, _ in res.components):
            continue
        cells = [res.cells[i] if res.cells[i] is not None else phases[i].cell for i in range(len(phases))]
        best = None
        for pl in planes[ph_i]:
            r = MultiProblem(phases, list(res.components) + [(ph_i, tuple(pl))], geom, peaks, mask, cfg,
                             cells, res.geom).run()
            cm = r.comp_metrics[-1]
            own = cm["frac_w"] * (1 - cm["n_strong_missing"] / max(cm["n_strong_pred"], 1))
            key = (own, r.metrics["fom"])
            if best is None or key > best[0]:
                best = (key, r, pl)
        if best is None:
            continue
        res = best[1]
        forced.append(st.name)
        if log:
            cm = res.comp_metrics[-1]
            log(f"    forced + {st.name}{tuple(best[2])}: {cm['n_match']} spots, missing strong "
                f"{cm['n_strong_missing']}/{cm['n_strong_pred']}  (joint fom {res.metrics['fom']:.3f})")
    return res, hist, forced


def _tolerance(res):
    cfg = res.probs[0].cfg
    return cfg.tol_schedule[-1], cfg.q_range


def phase_map(res, peaks, tol=None, min_I=0.02):
    """one row per detected spot in the fitted q range (see module docstring).
    min_I: a phase counts as a candidate for a spot only if its nearest predicted spot has at least this relative
    intensity (1 = the 10th strongest reflection of that phase); stops the many weak reflections of a large cell
    from 'explaining' every spot."""
    (tol_q, tol_chi), q_range = _tolerance(res)
    if tol is not None:
        tol_q, tol_chi = tol
    geom = res.geom
    qxy, qz = geom.pixel_to_q(peaks.col, peaks.row)
    q = np.hypot(qxy, qz)
    chi = np.degrees(np.arctan2(qxy, qz))
    use = np.nonzero((q >= q_range[0]) & (q <= q_range[1]))[0]
    assigned = {r["obs"]: r for r in res.pairs}
    phase_of_comp = [res.phase_names[p] for p, _ in res.components]
    names = list(dict.fromkeys(phase_of_comp))                 # phases present in the model, in order
    rows = []
    for i in use:
        best = {}                                              # phase -> (cost, I, label, contact)
        for c, (gx, gz, gI) in enumerate(res.group_pos):
            if not len(gx):
                continue
            gq, gc = np.hypot(gx, gz), np.degrees(np.arctan2(gx, gz))
            side0 = on_axis(res.groups[c])
            co = fold_chi(np.full(len(gx), chi[i]), side0)
            cost = ((q[i] - gq) / tol_q) ** 2 + ((co - gc) / tol_chi) ** 2
            cost = np.where(np.asarray(gI) >= min_I, cost, np.inf)
            j = int(np.argmin(cost))
            if cost[j] <= 1.0:
                ph = phase_of_comp[c]
                if ph not in best or cost[j] < best[ph][0]:
                    best[ph] = (float(cost[j]), float(gI[j]), res.groups[c][j].label, res.components[c][1])
        cands = [n for n in names if n in best]
        a = assigned.get(i)
        if not cands:
            label = UNASSIGNED
        elif len(cands) == 1:
            label = cands[0]
        else:
            label = " + ".join(cands)
        Itot = sum(best[n][1] for n in cands) or 1.0
        rows.append(dict(
            spot=int(i), col=float(peaks.col[i]), row=float(peaks.row[i]), qxy=float(qxy[i]), qz=float(qz[i]),
            q=float(q[i]), d_A=float(2 * np.pi / q[i]), chi_deg=float(chi[i]), area=float(peaks.area[i]),
            snr=float(peaks.snr[i]), label=label, n_candidates=len(cands),
            assigned_phase=a["phase"] if a else "", assigned_hkl=a["label"] if a else "",
            assigned_contact=a["contact"] if a else "", dq_fit_pct=a["dq_fit_pct"] if a else np.nan,
            **{f"hkl_{n}": best[n][2] if n in best else "" for n in names},
            **{f"share_{n}": (best[n][1] / Itot) if n in best else 0.0 for n in names}))
    return rows


def summary(rows, res=None):
    """counts per label, and the sqrt(area)-weighted share of the detected intensity in each label."""
    w = {r["spot"]: np.sqrt(max(r["area"], 0.0)) for r in rows}
    tot = sum(w.values()) or 1.0
    out = {}
    for r in rows:
        d = out.setdefault(r["label"], dict(n=0, w=0.0))
        d["n"] += 1
        d["w"] += w[r["spot"]] / tot
    return out


def write_csv(rows, path):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def phase_map_figure(D, mask, res, peaks, rows, path, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), label_top=10):
    """q-map with every detected spot coloured by its label: one colour per crystal, yellow = shared by two
    crystals, grey = no crystal.  Thin circles: predicted spots of each crystal at its refined cell."""
    import matplotlib.pyplot as plt
    from . import plots
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    plots._qmap(ax, D, mask, res.geom, qxy_lim, qz_lim)
    phase_of_comp = [res.phase_names[p] for p, _ in res.components]
    names = list(dict.fromkeys(phase_of_comp))
    color = {n: PHASE_COLORS[k % len(PHASE_COLORS)] for k, n in enumerate(names)}
    for c, (gx, gz, gI) in enumerate(res.group_pos):
        if len(gx):
            ax.scatter(gx, gz, s=12 + 90 * np.sqrt(np.clip(gI, 0, 4) / 4), facecolors="none",
                       edgecolors=color[phase_of_comp[c]], lw=0.7, alpha=0.55)
    handles = []
    for lab, col, mk in [(n, color[n], "o") for n in names] + [("shared", SHARED_COLOR, "D"),
                                                                 (UNASSIGNED, NONE_COLOR, "x")]:
        sel = [r for r in rows if (r["label"] == lab if lab != "shared" else r["n_candidates"] > 1)]
        if lab != UNASSIGNED and lab != "shared":
            contacts = sorted({str(ct) for p, ct in res.components if res.phase_names[p] == lab})
            text = f"{lab} {', '.join(contacts)}: {len(sel)} spots"
        elif lab == "shared":
            text = f"shared by two crystals: {len(sel)} spots"
        else:
            text = f"no crystal: {len(sel)} spots"
        if sel:
            x = [r["qxy"] for r in sel]; z = [r["qz"] for r in sel]
            if mk == "x":
                ax.plot(x, z, "x", color=col, ms=6, mew=1.3)
            else:
                ax.scatter(x, z, s=60, marker=mk, facecolors="none", edgecolors=col, lw=1.8)
        h, = ax.plot([], [], mk, mfc="none", mec=col, color=col, mew=1.6, label=text)
        handles.append(h)
    for n in names:
        # per crystal (each colour gets its own labels), visible spots first, then the brightest
        vis = [r for r in rows if r["label"] == n
               and qxy_lim[0] < r["qxy"] < qxy_lim[1] and qz_lim[0] < r["qz"] < qz_lim[1]]
        for r in sorted(vis, key=lambda r: -r["area"])[:label_top]:
            ax.text(r["qxy"] + 0.015, r["qz"] + 0.012, r[f"hkl_{n}"].split("/")[0], color=color[n],
                    fontsize=7.5, clip_on=True)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title("Which crystal does each spot belong to? (" + " + ".join(names) + ")", fontsize=12)
    plots.savefig(fig, path)
