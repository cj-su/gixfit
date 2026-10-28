"""Command line:  python -m gixfit peaks  config.toml     (peak finding only, to tune thresholds)
                python -m gixfit run    config.toml     (peaks -> CIF error table -> refine -> rank candidates)
                python -m gixfit multi  config.toml     (several textures / phases per pattern + chance level)
                python -m gixfit phases config.toml     (two or more crystals: which crystal each spot belongs to)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib

import numpy as np

from .crystal import load_structure, PARAM_NAMES, write_cif
from .geometry import Geometry
from .peaks import load_image, build_mask, find_peaks
from .pattern import low_index_planes
from .fit import FitConfig, Problem, null_test
from . import plots


def _slug(s):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_")


def load_config(path):
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_base"] = os.path.dirname(os.path.abspath(path))
    return cfg


def geometries(cfg, image_path):
    g = dict(cfg.get("geometry", {}))
    if "alpha_i_deg" not in g:          # take it from the file name (..._th0.100_...)
        m = re.search(r"_th(\d+\.\d+)", os.path.basename(image_path))
        g["alpha_i_deg"] = float(m.group(1)) if m else 0.1
    base = Geometry(**g)
    vars_ = cfg.get("geometry_variants") or [{"name": "nominal"}]
    out = []
    for v in vars_:
        v = dict(v)
        name = v.pop("name", "variant")
        out.append((name, base.with_(**v)))
    return out


def get_peaks(cfg):
    im = cfg["image"]
    path = os.path.join(cfg["_base"], im["path"])
    D = load_image(path, despike=im.get("despike", True), dead_pixels=[tuple(p) for p in im.get("dead_pixels", [])])
    mask = build_mask(D, edge=im.get("edge", 15), nonpositive=im.get("mask_nonpositive", True),
                      rects=[tuple(r) for r in im.get("mask_rects", [])], grow=im.get("mask_grow", 2))
    if "qz_min" in im:          # sample horizon: below it = substrate edge / transmission, not the film
        _, qz = geometries(cfg, path)[0][1].q_maps(D.shape)
        mask &= qz >= im["qz_min"]
    pk = find_peaks(D, mask, **cfg.get("peaks", {}))
    return path, D, mask, pk


def write_peaks_csv(pk, geom, path):
    qxy, qz = geom.pixel_to_q(pk.col, pk.row)
    q = np.hypot(qxy, qz)
    chi = np.degrees(np.arctan2(qxy, qz))
    with open(path, "w") as fh:
        fh.write("id,col,row,ecol,erow,qxy,qz,q,d_A,chi_deg,sig_major_px,sig_minor_px,area,snr\n")
        for i in range(len(pk)):
            fh.write(f"{i},{pk.col[i]:.3f},{pk.row[i]:.3f},{pk.ecol[i]:.3f},{pk.erow[i]:.3f},{qxy[i]:.5f},"
                     f"{qz[i]:.5f},{q[i]:.5f},{2*np.pi/q[i]:.4f},{chi[i]:.2f},{pk.sig_major[i]:.2f},"
                     f"{pk.sig_minor[i]:.2f},{pk.area[i]:.1f},{pk.snr[i]:.1f}\n")


def cmd_peaks(cfg, out):
    path, D, mask, pk = get_peaks(cfg)
    name, geom = geometries(cfg, path)[0]
    write_peaks_csv(pk, geom, os.path.join(out, "peaks.csv"))
    pl = cfg.get("plot", {})
    plots.peaks_figure(D, mask, geom, pk, os.path.join(out, "peaks_map.png"),
                       tuple(pl.get("qxy_lim", (-0.6, 2.0))), tuple(pl.get("qz_lim", (0.0, 2.0))),
                       title=f"{os.path.basename(path)}\n{len(pk)} peaks ({name})")
    print(f"{len(pk)} peaks -> {out}")
    return path, D, mask, pk


def indistinguishable(structs, qmax=2.0, tol=0.999):
    """flag candidate pairs whose powder patterns are identical (e.g. R and S enantiomers - Friedel's law)."""
    from .pattern import hkl_sphere
    grid = np.linspace(0.1, qmax, 4000)
    pats = []
    for s in structs:
        hkl = hkl_sphere(s.cell, qmax)
        q = np.linalg.norm(s.cell.g(hkl), axis=1)
        F2 = s.f_squared(hkl)
        y = np.zeros_like(grid)
        for qq, ff in zip(q, F2):
            y += ff * np.exp(-0.5 * ((grid - qq) / 0.003) ** 2)
        pats.append(y / (np.linalg.norm(y) + 1e-30))
    flags = []
    for i in range(len(structs)):
        for j in range(i + 1, len(structs)):
            c = float(pats[i] @ pats[j])
            if c > tol:
                flags.append((structs[i].name, structs[j].name, c))
    return flags


def cmd_run(cfg, out):
    t0 = time.time()
    path, D, mask, pk = cmd_peaks(cfg, out)
    fc = FitConfig.from_dict(cfg.get("fit", {}))
    nullc = cfg.get("null", {"n": 20, "modes": ["chi", "q"]})
    pl = cfg.get("plot", {})
    qlims = (tuple(pl.get("qxy_lim", (-0.6, 2.0))), tuple(pl.get("qz_lim", (0.0, 2.0))))
    cands = cfg["candidates"]
    structs = [load_structure(c, cfg["_base"]) for c in cands]
    for st, c in zip(structs, cands):
        write_cif(st, os.path.join(out, f"start_{_slug(st.name)}.cif"), st.name)
    flags = indistinguishable(structs, fc.qmax)
    results, table = [], []
    for gname, geom in geometries(cfg, path):
        for st, c in zip(structs, cands):
            planes = c.get("contact_planes", "auto")
            if planes == "auto":
                planes = low_index_planes(c.get("auto_maxidx", 2))
            scan = []
            for hkl in planes:
                r = Problem(st, hkl, geom, pk, mask, fc, st.name).run()
                scan.append(r)
                print(f"  [{gname}] {st.name} {tuple(hkl)}: {r.metrics['n_match']}/{r.metrics['n_obs']} "
                      f"fom={r.metrics['fom']:.3f} rms_cif={r.metrics['rms_dq_cif_pct']:.2f}% "
                      f"rms_fit={r.metrics['rms_dq_fit_pct']:.2f}%", flush=True)
            scan.sort(key=lambda r: -r.metrics["fom"])
            for r in scan[: c.get("auto_top", 3)]:
                prob = Problem(st, r.contact, geom, pk, mask, fc, st.name)
                r.null = null_test(prob, r, n=nullc.get("n", 20), modes=tuple(nullc.get("modes", ("chi", "q"))))
                r.geom_name = gname
                results.append(r)
    # ------------------------------------------------------------------ outputs
    import pickle
    with open(os.path.join(out, "results.pkl"), "wb") as fh:
        pickle.dump(dict(results=results, flags=flags, peaks=pk, image=path), fh)
    report(cfg, out, path, D, mask, pk, structs, results, flags, qlims, t0)


def report(cfg, out, path, D, mask, pk, structs, results, flags, qlims, t0=None):
    t0 = t0 or time.time()
    rows = []
    for k, r in enumerate(results):
        m = r.metrics
        row = dict(rank=0, candidate=r.name, contact=str(tuple(r.contact)), geometry=r.geom_name,
                   n_obs=m["n_obs"], n_match=m["n_match"], frac_match_w=m["frac_match_w"],
                   strong_missing=f"{m['n_strong_missing']}/{m['n_strong_pred']}", fom=m["fom"],
                   rms_dq_cif_pct=m["rms_dq_cif_pct"], rms_dq_fit_pct=m["rms_dq_fit_pct"],
                   chi2_red=m["chi2_red"], spearman_I=m["spearman_I"])
        for mode, v in (r.null or {}).items():
            row[f"null_{mode}_fom_mean"] = v["fom_mean"]; row[f"null_{mode}_fom_sd"] = v["fom_sd"]
            row[f"null_{mode}_z"] = v["z"]; row[f"null_{mode}_p"] = v["p"]
        for i, n in enumerate(PARAM_NAMES):
            row[f"{n}_cif"] = r.cell0.array()[i]
            row[f"{n}_fit"] = r.cell.array()[i]
            row[f"{n}_err"] = r.cell_err[i]
            row[f"{n}_change"] = m["strain"][i]
        row["V_cif"] = r.cell0.volume(); row["V_fit"] = r.cell.volume()
        for gk, gv in r.geom_err.items():
            row[f"{gk}_fit"] = getattr(r.geom, gk); row[f"{gk}_err"] = gv
        row["_idx"] = k
        rows.append(row)
    rows.sort(key=lambda t: -(t.get("null_chi_z") if np.isfinite(t.get("null_chi_z", np.nan)) else -1e9))
    for i, t in enumerate(rows):
        t["rank"] = i + 1
    plots.ranking_figure(rows, os.path.join(out, "ranking.png"))
    import pandas as pd
    xl = os.path.join(out, "gixfit_summary.xlsx")
    with pd.ExcelWriter(xl) as w:
        pd.DataFrame([{k: v for k, v in t.items() if k != "_idx"} for t in rows]).to_excel(w, sheet_name="ranking", index=False)
        pd.read_csv(os.path.join(out, "peaks.csv")).to_excel(w, sheet_name="observed_peaks", index=False)
        for t in rows:
            r = results[t["_idx"]]
            tag = f"{t['rank']:02d}_{_slug(r.name)[:10]}_{''.join(map(str, r.contact)).replace('-', 'm')}"
            df = pd.DataFrame([{k: (str(v) if k == "hkl" else v) for k, v in p.items()} for p in r.pairs])
            if len(df):
                df.sort_values("q_obs").to_excel(w, sheet_name=tag[:31], index=False)
        notes = [
            "fom = (sqrt(area)-weighted fraction of observed peaks matched) x (1 - fraction of the n_strong strongest predicted peaks not seen)",
            "null_chi: same fit on surrogate data with chi permuted among peaks (|q| kept); null_q: |q| jittered +-3% (chi kept)",
            "z = (fom - null mean)/null sd ; p = fraction of surrogates with fom >= real",
            "dq_cif_pct = (q_obs - q_CIF)/q_CIF with the START geometry ; dq_fit_pct uses refined cell (+ refined geometry if free)",
            "cell changes: lengths in %, angles in deg",
            f"indistinguishable candidate pairs (powder pattern correlation > 0.999): {flags or 'none'}",
        ]
        pd.DataFrame({"notes": notes}).to_excel(w, sheet_name="notes", index=False)
    for t in rows[: cfg.get("figures_top", 6)]:
        r = results[t["_idx"]]
        tag = f"{t['rank']:02d}_{_slug(r.name)}_{''.join(map(str, r.contact))}_{_slug(r.geom_name)}"
        plots.fit_figure(D, mask, r, pk, os.path.join(out, f"fit_{tag}.png"), *qlims)
        plots.error_figure(r, os.path.join(out, f"errors_{tag}.png"))
        st = [s for s in structs if s.name == r.name][0]
        write_cif(st.with_cell(r.cell), os.path.join(out, f"refined_{tag}.cif"), f"{r.name} refined")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(f"image: {path}\npeaks: {len(pk)}\n")
        if flags:
            fh.write("WARNING diffraction-indistinguishable candidates (e.g. enantiomers): " + str(flags) + "\n")
        for t in rows:
            r = results[t["_idx"]]
            fh.write(f"\n#{t['rank']} {r.name} contact {tuple(r.contact)} [{r.geom_name}]\n")
            fh.write(f"   matched {t['n_match']}/{t['n_obs']}  fom {t['fom']:.3f}  strong missing {t['strong_missing']}\n")
            for mode, v in (r.null or {}).items():
                fh.write(f"   null[{mode}] fom {v['fom_mean']:.3f} +- {v['fom_sd']:.3f}  z={v['z']:.1f}  exceed={v['p']:.3f}\n")
            fh.write(f"   rms dq: CIF {t['rms_dq_cif_pct']:.2f}%  refined {t['rms_dq_fit_pct']:.2f}%  chi2_red {t['chi2_red']:.2f}"
                     f"  Spearman(I) {t['spearman_I']:.2f}\n")
            fh.write(f"   CIF     {r.cell0}\n   refined {r.cell}\n")
            fh.write("   +-      " + " ".join(f"{n}={e:.4f}" for n, e in zip(PARAM_NAMES, r.cell_err)) + "\n")
            if r.geom_err:
                fh.write("   geometry " + " ".join(f"{k}={getattr(r.geom, k):.2f}+-{v:.2f}" for k, v in r.geom_err.items()) + "\n")
    print(open(os.path.join(out, "summary.txt")).read())
    print(f"done in {time.time() - t0:.0f} s -> {out}")


def _sig_k(null, K, z_min=3.0):
    """largest k such that every increment fom_j - fom_(j-1), j <= k, beats EVERY null mode by z >= z_min."""
    k_sig = 0
    for k in range(K):
        if null and all(np.isfinite(v[k]["inc_z"]) and v[k]["inc_z"] >= z_min for v in null.values()):
            k_sig = k + 1
        else:
            break
    return k_sig


def cmd_multi(cfg, out):
    """several textures / phases per pattern (gixfit.multi): greedy k = 1..max_components + chance level."""
    from .multi import MultiConfig, decompose, null_test_multi
    import pickle
    t0 = time.time()
    out = os.path.join(out, "multi")
    os.makedirs(out, exist_ok=True)
    path, D, mask, pk = get_peaks(cfg)
    write_peaks_csv(pk, geometries(cfg, path)[0][1], os.path.join(out, "peaks.csv"))
    fc = FitConfig.from_dict(cfg.get("fit", {}))
    md = dict(cfg.get("multi", {}))
    models = md.pop("models", None)
    mc = MultiConfig.from_dict(md)
    pl = cfg.get("plot", {})
    qlims = (tuple(pl.get("qxy_lim", (-0.6, 2.0))), tuple(pl.get("qz_lim", (0.0, 2.0))))
    cands = cfg["candidates"]
    structs = [load_structure(c, cfg["_base"]) for c in cands]
    byname = {s.name: (s, c) for s, c in zip(structs, cands)}
    models = models or [[s.name] for s in structs]
    flags = indistinguishable(structs, fc.qmax)
    runs = []
    for gname, geom in geometries(cfg, path):
        for model in models:
            phases = [byname[n][0] for n in model]
            planes = []
            for n in model:
                cp = byname[n][1].get("contact_planes", "auto")
                planes.append(low_index_planes(mc.planes_maxidx) if cp == "auto" else [tuple(p) for p in cp])
            print(f"[{gname}] model {' + '.join(model)}  ({sum(map(len, planes))} candidate textures)", flush=True)
            hist = decompose(phases, geom, pk, mask, fc, mc, planes, log=lambda s: print(s, flush=True))
            null = (null_test_multi(phases, geom, pk, mask, fc, mc, hist, planes, log=lambda s: print(s, flush=True))
                    if mc.n_null else {})
            runs.append(dict(geometry=gname, model=" + ".join(model), hist=hist, null=null,
                             k_sig=_sig_k(null, len(hist))))
            print(f"    -> textures passing both diagnostics: k = {runs[-1]['k_sig']}  ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(out, "multi_results.pkl"), "wb") as fh:
        pickle.dump(dict(runs=runs, flags=flags, peaks=pk, image=path), fh)
    report_multi(cfg, out, path, D, mask, pk, structs, runs, flags, qlims, mc, t0)


def report_multi(cfg, out, path, D, mask, pk, structs, runs, flags, qlims, mc, t0=None):
    import pandas as pd
    t0 = t0 or time.time()
    rows_k, rows_c, rows_cell = [], [], []
    for ri, run in enumerate(runs):
        for k, h in enumerate(run["hist"]):
            if h is None:
                continue
            m = h.metrics
            row = dict(run=ri + 1, geometry=run["geometry"], model=run["model"], k=k + 1, components=h.label(),
                       n_obs=m["n_obs"], n_match=m["n_match"], frac_match_w=m["frac_match_w"],
                       strong_missing=f"{m['n_strong_missing']}/{m['n_strong_pred']}", fom=m["fom"],
                       spearman_I_pooled=m["spearman_I_pooled"], chi2_red=m["chi2_red"],
                       rms_dq_cif_pct=m["rms_dq_cif_pct"], rms_dq_fit_pct=m["rms_dq_fit_pct"],
                       significant=(k + 1) <= run["k_sig"])
            for mode, per_k in run["null"].items():
                d = per_k[k]
                row.update({f"null_{mode}_mean": d["null_mean"], f"null_{mode}_sd": d["null_sd"],
                            f"null_{mode}_z": d["z"], f"null_{mode}_p": d["p"],
                            f"inc_{mode}": d["inc"], f"inc_null_{mode}": d["inc_null_mean"],
                            f"inc_{mode}_z": d["inc_z"], f"inc_{mode}_p": d["inc_p"]})
            rows_k.append(row)
            for cm in h.comp_metrics:
                rows_c.append(dict(run=ri + 1, geometry=run["geometry"], model=run["model"], k=k + 1,
                                   **{kk: (str(v) if kk == "missing_labels" else v) for kk, v in cm.items()}))
            for pi, (c0, c1, ce) in enumerate(zip(h.cells0, h.cells, h.cell_errs)):
                if c1 is None:
                    continue
                d = dict(run=ri + 1, geometry=run["geometry"], model=run["model"], k=k + 1, phase=h.phase_names[pi])
                for i, n in enumerate(PARAM_NAMES):
                    d[f"{n}_cif"] = c0.array()[i]; d[f"{n}_fit"] = c1.array()[i]; d[f"{n}_err"] = ce[i]
                    d[f"{n}_change"] = (100 * (c1.array()[i] / c0.array()[i] - 1)) if i < 3 else c1.array()[i] - c0.array()[i]
                d["V_cif"] = c0.volume(); d["V_fit"] = c1.volume()
                rows_cell.append(d)
    xl = os.path.join(out, "gixfit_multi.xlsx")
    with pd.ExcelWriter(xl) as w:
        pd.DataFrame(rows_k).to_excel(w, sheet_name="models_by_k", index=False)
        pd.DataFrame(rows_c).to_excel(w, sheet_name="components", index=False)
        pd.DataFrame(rows_cell).to_excel(w, sheet_name="cells", index=False)
        for ri, run in enumerate(runs):
            h = run["hist"][-1]
            if h is not None and h.pairs:
                pd.DataFrame(h.pairs).sort_values(["component", "q_obs"]).to_excel(
                    w, sheet_name=f"pairs_run{ri + 1}_k{len(run['hist'])}"[:31], index=False)
        notes = [
            "Greedy: k = 1..max_components textures (phase, contact plane); every step refines ALL components jointly "
            "(one cell per phase, shared by its textures); each observed spot is indexed by at most one component.",
            "fom = (sqrt(area)-weighted fraction of spots indexed) x (1 - fraction of each component's n_strong strongest "
            "predicted spots that are absent, pooled over components).",
            "null_<mode>: the SAME greedy + joint pipeline on surrogate spots (chi: chi permuted, |q| kept; q: |q| x U(0.97,1.03)).",
            "inc = fom_k - fom_(k-1) (what the k-th texture adds); inc_z compares it with the same increment on surrogates.",
            f"k counted = every increment up to k has inc_z >= 3 in every surrogate mode (descriptive).  n_null = {mc.n_null} per mode "
            f"(smallest possible p = {1 / (mc.n_null + 1):.3f}).",
            "spearman_I_pooled: predicted vs observed intensity, ranks taken within each component (unknown volume fractions).",
            f"indistinguishable candidate pairs (powder correlation > 0.999): {flags or 'none'}",
        ]
        pd.DataFrame({"notes": notes}).to_excel(w, sheet_name="notes", index=False)
    for ri, run in enumerate(runs):
        tag = f"run{ri + 1}_{_slug(run['model'])}_{_slug(run['geometry'])}"
        h = run["hist"][-1]
        if h is not None:
            plots.multi_fit_figure(D, mask, h, pk, os.path.join(out, f"multi_fit_{tag}.png"), *qlims)
        if run["null"]:
            plots.fom_vs_k_figure(run, os.path.join(out, f"fom_vs_k_{tag}.png"))
        for k, hk in enumerate(run["hist"]):
            if hk is None:
                continue
            for pi, c1 in enumerate(hk.cells):
                if c1 is not None and (k + 1) == len(run["hist"]):
                    st = structs[[s.name for s in structs].index(hk.phase_names[pi])]
                    write_cif(st.with_cell(c1), os.path.join(out, f"refined_{tag}_{_slug(st.name)}.cif"),
                              f"{st.name} refined (multi)")
    with open(os.path.join(out, "summary_multi.txt"), "w") as fh:
        fh.write(f"image: {path}\nspots: {len(pk)}\n")
        if flags:
            fh.write("WARNING diffraction-indistinguishable candidates (e.g. enantiomers): " + str(flags) + "\n")
        for ri, run in enumerate(runs):
            fh.write(f"\n=== run {ri + 1}: {run['model']}  [{run['geometry']}]   textures passing both diagnostics k = {run['k_sig']}\n")
            for k, h in enumerate(run["hist"]):
                if h is None:
                    continue
                m = h.metrics
                fh.write(f"  k={k + 1}: {h.label()}\n")
                fh.write(f"       indexed {m['n_match']}/{m['n_obs']} (sqrt-area {m['frac_match_w']:.2f})  "
                         f"strong missing {m['n_strong_missing']}/{m['n_strong_pred']}  fom {m['fom']:.3f}  "
                         f"rho(I) {m['spearman_I_pooled']:.2f}  rms dq CIF {m['rms_dq_cif_pct']:.2f}% -> {m['rms_dq_fit_pct']:.2f}%\n")
                for mode, per_k in run["null"].items():
                    d = per_k[k]
                    fh.write(f"       null[{mode}] {d['null_mean']:.3f}+-{d['null_sd']:.3f} z={d['z']:.1f} exceed={d['p']:.3f}"
                             f" | increment {d['inc']:+.3f} vs {d['inc_null_mean']:+.3f}+-{d['inc_null_sd']:.3f}"
                             f" z={d['inc_z']:.1f} exceed={d['inc_p']:.3f}\n")
                for cm in h.comp_metrics:
                    fh.write(f"       - {cm['phase']}{cm['contact']}: {cm['n_match']} spots (sqrt-area {cm['frac_w']:.2f}), "
                             f"strong missing {cm['n_strong_missing']}/{cm['n_strong_pred']}, rho(I) {cm['spearman_I']:.2f}\n")
                for pi, c1 in enumerate(h.cells):
                    if c1 is not None:
                        fh.write(f"       {h.phase_names[pi]} refined {c1}\n")
    print(open(os.path.join(out, "summary_multi.txt")).read())
    print(f"done in {time.time() - t0:.0f} s -> {out}")


def cmd_phases(cfg, out):
    """every [[candidates]] entry is a crystal assumed present (e.g. A and B of a bilayer); fit them jointly and label
    each detected spot with the crystal(s) that explain it (gixfit.phasemap)."""
    from .multi import MultiConfig
    from .phasemap import fit_phases, phase_map, summary, write_csv, phase_map_figure
    t0 = time.time()
    out = os.path.join(out, "phases")
    os.makedirs(out, exist_ok=True)
    path, D, mask, pk = get_peaks(cfg)
    gname, geom = geometries(cfg, path)[0]
    fc = FitConfig.from_dict(cfg.get("fit", {}))
    md = dict(cfg.get("multi", {}))
    md.pop("models", None)
    md.setdefault("max_components", len(cfg["candidates"]) + 1)
    md["n_null"] = 0
    mc = MultiConfig.from_dict(md)
    pl = cfg.get("plot", {})
    qlims = (tuple(pl.get("qxy_lim", (-0.6, 2.0))), tuple(pl.get("qz_lim", (0.0, 2.0))))
    structs = [load_structure(c, cfg["_base"]) for c in cfg["candidates"]]
    planes = [low_index_planes(mc.planes_maxidx) if c.get("contact_planes", "auto") == "auto"
              else [tuple(p) for p in c["contact_planes"]] for c in cfg["candidates"]]
    print(f"[{gname}] crystals: {' + '.join(s.name for s in structs)}", flush=True)
    res, hist, forced = fit_phases(structs, geom, pk, mask, fc, mc, planes, log=lambda s: print(s, flush=True))
    rows = phase_map(res, pk, min_I=cfg.get("phases", {}).get("min_I", 0.02))
    write_csv(rows, os.path.join(out, "phase_map.csv"))
    phase_map_figure(D, mask, res, pk, rows, os.path.join(out, "phase_map.png"), *qlims)
    cnt = summary(rows)
    lines = [f"image: {path}", f"geometry: {gname}", f"model: {res.label()}",
             f"phases added by force (assumed present, not established by the data): {forced or 'none'}", ""]
    for cm in res.comp_metrics:
        lines.append(f"{cm['phase']} {cm['contact']}: {cm['n_match']} spots indexed, strong predicted spots missing "
                     f"{cm['n_strong_missing']}/{cm['n_strong_pred']}, intensity rank corr {cm['spearman_I']:.2f}")
    for ph, cell, err in zip(res.phase_names, res.cells, res.cell_errs):
        if cell is not None:
            lines.append(f"  {ph} refined {cell}")
    lines += ["", "spot labels (count, sqrt(area)-weighted share of the detected intensity):"]
    for lab, d in sorted(cnt.items(), key=lambda t: -t[1]["n"]):
        lines.append(f"  {lab:30s} {d['n']:4d}   {100 * d['w']:5.1f} %")
    lines += ["", "label 'A + B' = both crystals predict a spot there (shared / ambiguous); 'unassigned' = neither.",
              f"done in {time.time() - t0:.0f} s"]
    txt = "\n".join(lines)
    with open(os.path.join(out, "phase_map_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


def main(argv=None):
    if (argv if argv is not None else sys.argv[1:])[:1] == ["app"]:
        from .app import main as app_main
        return app_main()
    ap = argparse.ArgumentParser(prog="gixfit", description="'python -m gixfit app' opens the desktop app")
    ap.add_argument("command", choices=["peaks", "run", "multi", "phases"])
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = a.out or os.path.join(cfg["_base"], cfg.get("out_dir", "gixfit_out"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config_used.json"), "w") as fh:
        json.dump({k: v for k, v in cfg.items()}, fh, indent=1, default=str)
    if a.command == "peaks":
        cmd_peaks(cfg, out)
    elif a.command == "multi":
        cmd_multi(cfg, out)
    elif a.command == "phases":
        cmd_phases(cfg, out)
    else:
        cmd_run(cfg, out)


if __name__ == "__main__":
    main()
