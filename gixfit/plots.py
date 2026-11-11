"""Figures (Times New Roman bold, stix mathtext - the user's house style)."""
from __future__ import annotations

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .peaks import to_qgrid

plt.rcParams.update({
    "font.family": "Times New Roman", "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold", "mathtext.fontset": "stix", "mathtext.default": "bf",
    "font.size": 13, "axes.linewidth": 1.4,
})
QXY = r"$q_{xy}$ ($\AA^{-1}$)"
QZ = r"$q_z$ ($\AA^{-1}$)"


def savefig(fig, path, dpi=200):
    for k in range(5):           # OneDrive can lock files briefly
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            break
        except PermissionError:
            time.sleep(1.0 + k)
    plt.close(fig)


def _qmap(ax, D, mask, geom, qxy_lim, qz_lim, dq=0.004):
    qxy, qz, img = to_qgrid(D, mask, geom, qxy_lim, qz_lim, dq)
    v = img[np.isfinite(img) & (img > 0)]
    lo, hi = (np.percentile(v, 30), np.percentile(v, 99.7)) if v.size else (1, 10)
    ax.imshow(np.clip(img, lo, None), origin="lower", extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
              cmap="magma", norm=LogNorm(lo, hi), aspect="equal", interpolation="nearest")
    ax.set_facecolor("#bdbdbd")
    ax.set_xlabel(QXY); ax.set_ylabel(QZ)
    ax.set_xlim(qxy_lim); ax.set_ylim(qz_lim)


def peaks_figure(D, mask, geom, peaks, path, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), title=""):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    _qmap(ax, D, mask, geom, qxy_lim, qz_lim)
    qxy, qz = geom.pixel_to_q(peaks.col, peaks.row)
    s = np.clip(np.log10(np.clip(peaks.area, 1, None)) * 12, 10, 90)
    ax.scatter(qxy, qz, s=s, facecolors="none", edgecolors="cyan", lw=1.1)
    for i, (x, z) in enumerate(zip(qxy, qz)):
        if qxy_lim[0] < x < qxy_lim[1] and qz_lim[0] < z < qz_lim[1]:
            ax.text(x + 0.012, z + 0.012, str(i), color="cyan", fontsize=7, clip_on=True)
    ax.set_title(title or f"{len(peaks)} detected peaks")
    savefig(fig, path)


def fit_figure(D, mask, res, peaks, path, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), label_top=25):
    """left: CIF prediction (start cell) vs observed; right: refined cell vs observed.
    white circles = predicted (size ~ sqrt intensity); cyan + = matched observed; yellow x = unmatched observed."""
    from .pattern import fiber_qcoords, wedge_project
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.8))
    for ax, which in zip(axs, ("cif", "fit")):
        geom = res.geom0 if which == "cif" else res.geom
        _qmap(ax, D, mask, geom, qxy_lim, qz_lim)
        cell = res.cell0 if which == "cif" else res.cell
        tilt = (0.0, 0.0) if which == "cif" else res.tilt
        refl = res.refl
        qxy, qz, _ = fiber_qcoords(cell, refl.hkl, res.contact, tilt)
        qxy = qxy * np.where(refl.side == 0, 1, refl.side)
        qxy, qz = wedge_project(qxy, qz, geom, res.mosaic_deg)
        gx = np.array([np.average(qxy[g.members], weights=refl.I[g.members] + 1e-12) for g in res.groups])
        gz = np.array([np.average(qz[g.members], weights=refl.I[g.members] + 1e-12) for g in res.groups])
        gI = res.group_pos[2]
        ax.scatter(gx, gz, s=15 + 160 * np.sqrt(gI), facecolors="none", edgecolors="white", lw=1.0, alpha=0.9)
        ax0 = np.array([g.side == 0 for g in res.groups], bool)      # on-axis: either wedge edge
        if ax0.any():
            ax.scatter(-gx[ax0], gz[ax0], s=15 + 160 * np.sqrt(gI[ax0]), facecolors="none", edgecolors="white",
                       lw=1.0, alpha=0.9, linestyles="--")
        for p in res.pairs:
            x, z = geom.pixel_to_q(p["col"], p["row"])
            ax.plot(x, z, "+", color="#00e5ff", ms=9, mew=1.6)
        if len(res.unmatched_obs):
            ux, uz = geom.pixel_to_q(peaks.col[res.unmatched_obs], peaks.row[res.unmatched_obs])
            ax.plot(ux, uz, "x", color="#ffd400", ms=6, mew=1.3)
        # labels for the strongest matched
        vis = []                                          # visible first, then the strongest
        for p in res.pairs:
            x, z = geom.pixel_to_q(p["col"], p["row"])
            if qxy_lim[0] < x < qxy_lim[1] and qz_lim[0] < z < qz_lim[1]:
                vis.append((p, float(x), float(z)))
        for p, x, z in sorted(vis, key=lambda t: -t[0]["I_pred"])[:label_top]:
            lab = p["label"].split("/")[0]
            dq = p["dq_cif_pct"] if which == "cif" else p["dq_fit_pct"]
            ax.text(x + 0.015, z + 0.012, f"{lab} {dq:+.1f}%", color="#7CFC00", fontsize=7.5, clip_on=True)
        m = res.metrics
        if which == "cif":
            ax.set_title(f"CIF cell, contact {tuple(res.contact)}: rms $\\Delta q$ = {m['rms_dq_cif_pct']:.2f}%")
        else:
            ax.set_title(f"refined: rms $\\Delta q$ = {m['rms_dq_fit_pct']:.2f}%, "
                         f"{m['n_match']}/{m['n_obs']} peaks, FOM {m['fom']:.2f}")
    savefig(fig, path)


def error_figure(res, path):
    """per-reflection Delta q/q (%) and Delta chi: CIF vs refined, sorted by q."""
    rows = sorted(res.pairs, key=lambda r: r["q_obs"])
    if not rows:
        return
    q = np.array([r["q_obs"] for r in rows])
    dc = np.array([r["dq_cif_pct"] for r in rows])
    df = np.array([r["dq_fit_pct"] for r in rows])
    ech = np.array([100 * r["q_err"] / r["q_obs"] for r in rows])
    xc = np.array([r["chi_obs"] - r["chi_cif"] for r in rows])
    xf = np.array([r["chi_obs"] - r["chi_fit"] for r in rows])
    lab = [r["label"].split("/")[0] for r in rows]
    fig, axs = plt.subplots(2, 1, figsize=(max(9, 0.32 * len(rows) + 3), 8), sharex=True)
    x = np.arange(len(rows))
    axs[0].axhline(0, color="k", lw=1)
    axs[0].errorbar(x - 0.15, dc, yerr=ech, fmt="o", color="#c0392b", ms=5, capsize=2, label="vs CIF cell")
    axs[0].errorbar(x + 0.15, df, yerr=ech, fmt="s", color="#1f6fb4", ms=5, capsize=2, label="vs refined cell")
    axs[0].set_ylabel(r"$(q_{obs}-q_{calc})/q_{calc}$ (%)")
    axs[0].legend(frameon=False)
    axs[1].axhline(0, color="k", lw=1)
    axs[1].plot(x - 0.15, xc, "o", color="#c0392b", ms=5)
    axs[1].plot(x + 0.15, xf, "s", color="#1f6fb4", ms=5)
    axs[1].set_ylabel(r"$\chi_{obs}-\chi_{calc}$ (deg)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{l}\n{qq:.3f}" for l, qq in zip(lab, q)], rotation=90, fontsize=8)
    axs[0].set_title(f"{res.name}  contact {tuple(res.contact)}")
    savefig(fig, path)


def ranking_figure(table, path):
    """table: list of dicts with name, fom, null fom mean/sd."""
    if not table:
        return
    lab = [f"{t['candidate']}\n{t['contact']} | {t['geometry']}" for t in table]
    fom = np.array([t["fom"] for t in table])
    nm = np.array([t.get("null_chi_fom_mean", np.nan) for t in table])
    ns = np.array([t.get("null_chi_fom_sd", np.nan) for t in table])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
    y = np.arange(len(table))[::-1]
    ax.barh(y, fom, color="#1f6fb4", label="model")
    ax.errorbar(nm, y, xerr=ns, fmt="|", color="#c0392b", ms=14, mew=2, label="chance ($\\chi$-shuffled)")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("figure of merit (matched fraction x (1 - missing strong))")
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, path)


COMP_COLORS = ["#ffffff", "#00e5ff", "#7CFC00", "#ff66cc", "#ffa500"]


def multi_fit_figure(D, mask, res, peaks, path, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), label_top=8):
    """refined multi-component model: circles = predicted spots of each component (colour), + = indexed spots in
    the colour of their component, yellow x = spots no component explains."""
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    _qmap(ax, D, mask, res.geom, qxy_lim, qz_lim)
    draw_multi(ax, res, peaks, qxy_lim, qz_lim, label_top)
    savefig(fig, path)


def draw_multi(ax, res, peaks, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), label_top=8, title=True):
    """overlay of a MultiResult on an axis that already shows the q-map (used by the GUI and multi_fit_figure).
    hkl labels: the `label_top` strongest indexed spots INSIDE the plotted q range, counted over all components
    together (None = every visible indexed spot; 0 = none).  Returns the number of labels drawn."""
    n = draw_overlay(ax, [res], peaks, qxy_lim, qz_lim, label_top)
    if title:
        m = res.metrics
        ax.set_title(f"{m['n_components']} textures: {m['n_match']}/{m['n_obs']} spots indexed, fom {m['fom']:.2f}, "
                     f"$\\rho_I$ {m['spearman_I_pooled']:.2f}", fontsize=12)
    return n


def draw_overlay(ax, results, peaks, qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0), label_top=8):
    """several fits on one map (each fit = a MultiResult; every crystal/texture gets its own colour).

    For separate fits shown together, a spot indexed by two or more of them is marked with a diamond (each fit
    claims it on its own; use a joint fit to make the crystals compete), and 'not indexed' means not indexed by ANY
    of them.  hkl labels: the `label_top` strongest visible indexed spots over all fits together (None = all).
    Returns the number of labels drawn."""
    handles, candidates = [], []
    claimed = {}                                     # observed spot -> set of fits that index it
    base = 0                                         # colour index offset of the current fit
    for ri, res in enumerate(results):
        n_ph = len(getattr(res, "phase_names", None) or [0])
        by_phase = n_ph > 1                          # crystals fitted together: one colour per CRYSTAL (stable across
        for c, (gx, gz, gI) in enumerate(res.group_pos):                         # images); one crystal: per texture
            key = res.components[c][0] if by_phase else c
            col = COMP_COLORS[(base + key) % len(COMP_COLORS)]
            if len(gx):
                ax.scatter(gx, gz, s=15 + 140 * np.sqrt(np.clip(gI, 0, 4) / 4), facecolors="none", edgecolors=col,
                           lw=1.0, alpha=0.9)
                ax0 = np.array([g.side == 0 for g in res.groups[c]], bool)      # on-axis: either wedge edge
                if ax0.any():
                    ax.scatter(-gx[ax0], gz[ax0], s=15 + 140 * np.sqrt(np.clip(gI[ax0], 0, 4) / 4),
                               facecolors="none", edgecolors=col, lw=1.0, alpha=0.9, linestyles="--")
            cm = res.comp_metrics[c]
            h, = ax.plot([], [], "o", mfc="none", mec=col, mew=1.5,
                         label=f"{cm['phase']} {cm['contact']}: {cm['n_match']} spots, "
                               f"missing {cm['n_strong_missing']}/{cm['n_strong_pred']}")
            handles.append(h)
            for r in (r for r in res.pairs if r["component"] == c):
                x, z = res.geom.pixel_to_q(r["col"], r["row"])
                ax.plot(x, z, "+", color=col, ms=10, mew=1.8)
                claimed.setdefault(int(r["obs"]), set()).add(ri)
                if qxy_lim[0] < x < qxy_lim[1] and qz_lim[0] < z < qz_lim[1]:   # visible first, then the strongest
                    candidates.append((-r["I_pred"], float(x), float(z), r["label"].split("/")[0], col))
        base += n_ph if by_phase else len(res.group_pos)
    candidates.sort(key=lambda t: t[0])
    shown = candidates if label_top is None else candidates[:max(0, int(label_top))]
    for _, x, z, lab, col in shown:
        ax.text(x + 0.015, z + 0.012, lab, color=col, fontsize=7.5, clip_on=True)
    if len(results) > 1:
        both = np.array(sorted(i for i, f in claimed.items() if len(f) > 1), int)
        if len(both):
            bx, bz = results[0].geom.pixel_to_q(peaks.col[both], peaks.row[both])
            h, = ax.plot(bx, bz, "D", mfc="none", mec="#ffd400", ms=9, mew=1.5,
                         label=f"indexed by more than one fit ({len(both)})")
            handles.append(h)
    unm = set(int(i) for i in results[0].unmatched_obs) if results else set()
    for res in results[1:]:
        unm &= set(int(i) for i in res.unmatched_obs)  # not indexed by ANY fit
    unm = np.array(sorted(unm), int)
    if len(unm):
        ux, uz = results[0].geom.pixel_to_q(peaks.col[unm], peaks.row[unm])
        h, = ax.plot(ux, uz, "x", color="#ffd400", ms=6, mew=1.3, label=f"not indexed ({len(unm)})")
        handles.append(h)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.85)
    return len(shown)


def fom_vs_k_figure(run, path):
    """fom after k textures (real data) vs the same greedy pipeline on surrogates (mean +- 1 sd)."""
    hist = run["hist"]
    K = len(hist)
    k = np.arange(1, K + 1)
    real = [h.metrics["fom"] if h else 0.0 for h in hist]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    styles = {"chi": ("#c0392b", r"$\chi$-shuffled"), "q": ("#7f8c8d", r"|q| jittered $\pm$3%")}
    for mode, per_k in run["null"].items():
        col, lab = styles.get(mode, ("#555555", mode))
        mu = np.array([d["null_mean"] for d in per_k]); sd = np.array([d["null_sd"] for d in per_k])
        ax.fill_between(k, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.plot(k, mu, "--", color=col, lw=1.6, label=f"chance ({lab})")
    ax.plot(k, real, "o-", color="#1f6fb4", lw=2.2, ms=8, label="data")
    ax.set_xticks(k)
    ax.set_xlabel("number of textures k")
    ax.set_ylabel("figure of merit")
    ax.set_ylim(0, max(0.3, max(real) * 1.25))
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(f"{run['model']}  [{run['geometry']}]", fontsize=11)
    savefig(fig, path)
