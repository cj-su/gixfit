"""High-level API behind the gixfit app (also usable from scripts / notebooks).

    s = Settings(x0=577, y0=643)                       # detector geometry + options
    ses = Session("image.tiff", s)                     # 1. load, mask, find spots  -> ses.peak_rows(), ses.hints()
    ev = check_cif(ses, "film.cif")                    # 2. how well does ONE CIF match (vs chance)?
    evs, flags = compare_cifs(ses, ["a.cif", "b.cif"]) # 3. which of several CIFs matches best?
    evs, flags, failed = identify(ses, [r"...\\cif files"])   # 1b. search a CIF library for the structure
    export(ses, evs, flags, "report_folder")

Every candidate goes through the same procedure (gixfit.multi): scan all low-index contact planes, add up to
``max_textures`` fibre textures greedily, refine one cell per CIF, then (supplementary) compare the figure of merit
with surrogate (scrambled) spot sets pushed through the identical procedure.  The outcome only says which surrogate
diagnostics were passed; passing them does not identify the structure (a different ordered lattice can pass too).

Honest limits (shown in the app):
* "Identify" means: which structure IN YOUR CIF LIBRARY explains the pattern.  It cannot solve an unknown structure.
* R and S enantiomers give identical diffraction (Friedel's law) - flagged, never ranked against each other.
* CIFs without atomic coordinates (cell only) are tested on positions + systematic absences; intensities unused.
"""
from __future__ import annotations

import glob
import os
import re
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np

from .crystal import read_cif, write_cif, Structure, PARAM_NAMES
from .geometry import Geometry
from .peaks import load_image, build_mask, find_peaks
from .pattern import low_index_planes
from .fit import FitConfig
from .multi import MultiConfig, decompose, null_test_multi


# ---------------------------------------------------------------------- settings
@dataclass
class Settings:
    # detector / beam (beam centre = DIRECT beam, 0-indexed column, row)
    wavelength: float = 0.9184
    sdd_mm: float = 259.0
    pixel_mm: float = 0.172
    x0: float = 577.0
    y0: float = 643.0
    alpha_i_deg: float | None = None        # None = read "_th0.100_" from the file name
    alpha_c_deg: float | None = None
    # image / mask
    qz_min: float = 0.03                    # mask below the sample horizon
    edge: int = 15
    dead_pixels: list = field(default_factory=list)
    mask_rects: list = field(default_factory=list)
    # spot finding
    threshold: float = 7.0
    min_amp_sigma: float = 4.0
    max_peaks: int = 300
    # fitting
    q_min: float = 0.35
    q_max: float = 1.9
    mosaic_deg: float = 3.0
    max_strain: float = 0.06
    max_angle_dev: float = 5.0
    n_strong: int = 20
    # model search / chance level
    max_textures: int = 1
    planes_maxidx: int = 2
    n_null: int = 10
    null_modes: tuple = ("chi", "q")
    workers: int = 0
    identify_null_top: int = 3              # identify: chance test only for the best N library entries

    def geometry(self, image_path=None) -> Geometry:
        ai = self.alpha_i_deg
        if ai is None:
            m = re.search(r"_th(\d+\.\d+)", os.path.basename(image_path or ""))
            ai = float(m.group(1)) if m else 0.1
        return Geometry(self.wavelength, self.sdd_mm, self.pixel_mm, self.x0, self.y0, ai, self.alpha_c_deg)

    def fit_config(self) -> FitConfig:
        return FitConfig(qmax=self.q_max + 0.1, q_range=(self.q_min, self.q_max), mosaic_deg=self.mosaic_deg,
                         max_strain=self.max_strain, max_angle_dev=self.max_angle_dev, n_strong=self.n_strong)

    def multi_config(self, **kw) -> MultiConfig:
        mc = MultiConfig(max_components=self.max_textures, planes_maxidx=self.planes_maxidx, n_null=self.n_null,
                         null_modes=tuple(self.null_modes), workers=self.workers)
        return replace(mc, **kw)

    # ---- persistence (simple TOML, readable by `python -m gixfit` configs as well)
    def to_toml(self, path, image_path=""):
        d = asdict(self)
        lines = ["# gixfit app settings", "[app]", f'image = "{image_path.replace(chr(92), "/")}"', ""]
        lines.append("[settings]")
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, (list, tuple)):
                lines.append(f"{k} = {_toml_list(v)}")
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    @classmethod
    def from_toml(cls, path, variant=None):
        """reads either an app settings file or a `python -m gixfit` example config ([geometry], [peaks], [fit])."""
        import tomllib
        with open(path, "rb") as fh:
            c = tomllib.load(fh)
        s = cls()
        if "settings" in c:
            for k, v in c["settings"].items():
                if hasattr(s, k):
                    setattr(s, k, tuple(v) if k == "null_modes" else v)
            img = c.get("app", {}).get("image", "")
            return s, img, []
        g = dict(c.get("geometry", {}))
        variants = c.get("geometry_variants", [])
        if variant is not None and variants:
            v = dict(variants[variant]); v.pop("name", None); g.update(v)
        for k in ("wavelength", "sdd_mm", "pixel_mm", "x0", "y0", "alpha_i_deg", "alpha_c_deg"):
            if k in g:
                setattr(s, k, g[k])
        im = c.get("image", {})
        s.qz_min = im.get("qz_min", s.qz_min); s.edge = im.get("edge", s.edge)
        s.dead_pixels = im.get("dead_pixels", []); s.mask_rects = im.get("mask_rects", [])
        pk = c.get("peaks", {})
        s.threshold = pk.get("threshold", s.threshold); s.min_amp_sigma = pk.get("min_amp_sigma", s.min_amp_sigma)
        s.max_peaks = pk.get("max_peaks", s.max_peaks)
        ft = c.get("fit", {})
        if "q_range" in ft:
            s.q_min, s.q_max = ft["q_range"]
        for k in ("mosaic_deg", "max_strain", "max_angle_dev", "n_strong"):
            if k in ft:
                setattr(s, k, ft[k])
        img = im.get("path", "")
        if img and not os.path.isabs(img):
            img = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), img))
        return s, img, [v.get("name", f"variant {i}") for i, v in enumerate(variants)]


def _toml_list(v):
    out = []
    for x in v:
        if isinstance(x, (list, tuple)):
            out.append(_toml_list(x))
        elif isinstance(x, str):
            out.append(f'"{x}"')
        else:
            out.append(str(x))
    return "[" + ", ".join(out) + "]"


# ---------------------------------------------------------------------- 1. spots
class Session:
    """one image + settings: raw data, mask and detected spots."""

    def __init__(self, image_path, settings: Settings, log=None):
        self.path, self.s = image_path, settings
        log = log or (lambda *_: None)
        t0 = time.time()
        self.D = load_image(image_path, dead_pixels=[tuple(p) for p in settings.dead_pixels])
        self.geom = settings.geometry(image_path)
        mask = build_mask(self.D, edge=settings.edge, rects=[tuple(r) for r in settings.mask_rects])
        if settings.qz_min is not None:
            _, qz = self.geom.q_maps(self.D.shape)
            mask &= qz >= settings.qz_min
        self.mask = mask
        self.peaks = find_peaks(self.D, mask, threshold=settings.threshold, min_amp_sigma=settings.min_amp_sigma,
                                max_peaks=settings.max_peaks)
        log(f"{len(self.peaks)} spots found in {time.time() - t0:.1f} s "
            f"({self.n_used()} within q = {settings.q_min}-{settings.q_max} A^-1 are used for matching)")

    def q(self):
        qxy, qz = self.geom.pixel_to_q(self.peaks.col, self.peaks.row)
        return qxy, qz, np.hypot(qxy, qz), np.degrees(np.arctan2(qxy, qz))

    def n_used(self):
        q = self.q()[2]
        return int(np.sum((q >= self.s.q_min) & (q <= self.s.q_max)))

    def peak_rows(self):
        qxy, qz, q, chi = self.q()
        pk = self.peaks
        order = np.argsort(-pk.area)
        return [dict(id=int(i), q=float(q[i]), d=float(2 * np.pi / q[i]), chi=float(chi[i]), qxy=float(qxy[i]),
                     qz=float(qz[i]), intensity=float(pk.area[i]), snr=float(pk.snr[i]),
                     used=bool(self.s.q_min <= q[i] <= self.s.q_max)) for i in order]

    def hints(self):
        """structure hints that need no CIF: out-of-plane (specular) series, in-plane spots, repeated |q|."""
        qxy, qz, q, chi = self.q()
        k = self.geom.k
        area = self.peaks.area
        lines = []
        near_axis = np.abs(qxy) < q ** 2 / (2 * k) + 0.03
        oop = np.argsort(q[near_axis])
        qo, ao = q[near_axis][oop], area[near_axis][oop]
        if len(qo):
            lines.append("Out-of-plane (specular) spots: " + ", ".join(f"q {a:.3f} (d {2 * np.pi / a:.2f} A)" for a in qo[:6]))
            q0 = qo[0]
            orders = [(n, a) for a in qo[1:] for n in range(2, 7) if abs(a / q0 - n) < 0.015 * n]
            if orders:
                lines.append(f"  -> a layered series: spacing d = {2 * np.pi / q0:.2f} A normal to the substrate "
                             f"(orders 1, {', '.join(str(n) for n, _ in orders)} seen)")
            else:
                lines.append(f"  -> strongest stacking normal to the substrate: d = {2 * np.pi / qo[np.argmax(ao)]:.2f} A")
        else:
            lines.append("No out-of-plane spot found (no preferred orientation along q_z, or masked).")
        inpl = np.abs(chi) > 75
        if inpl.any():
            o = np.argsort(-area[inpl])[:6]
            lines.append("In-plane spots (chi > 75 deg): " + ", ".join(f"d {2 * np.pi / a:.2f} A" for a in q[inpl][o]))
        # spots sharing one |q| at different chi = same d-spacing seen at several orientations
        used = (q >= self.s.q_min) & (q <= self.s.q_max)
        qs = np.sort(q[used])
        if len(qs):
            groups, cur = [], [qs[0]]
            for a in qs[1:]:
                if a - cur[-1] < 0.006:
                    cur.append(a)
                else:
                    groups.append(cur); cur = [a]
            groups.append(cur)
            rings = sorted([g for g in groups if len(g) >= 3], key=len, reverse=True)[:5]
            if rings:
                lines.append("d-spacings seen at >= 3 orientations: " +
                             ", ".join(f"{2 * np.pi / np.mean(g):.2f} A (x{len(g)})" for g in rings))
        sharp = np.median(self.peaks.sig_major)
        lines.append(f"Spot width (median sigma) {sharp:.1f} px -> "
                     + ("sharp, crystalline" if sharp < 2.5 else "broad (small or disordered crystallites)"))
        lines.append("A full unit cell needs a candidate: use 'Identify' with a CIF library, or 'Check CIF'.")
        return lines


# ---------------------------------------------------------------------- 2./3. evaluation
@dataclass
class Evaluation:
    name: str
    path: str
    struct: Structure
    hist: list
    null: dict
    k_sig: int
    best: object                 # MultiResult reported
    verdict: str
    grade: str                   # "both" | "chi" | "q" | "neither" | "untested" (surrogate diagnostics passed)
    fom: float
    z_chi: float
    z_q: float
    cell_only: bool
    seconds: float

    def summary_line(self):
        m = self.best.metrics if self.best else {}
        tex = self.best.label() if self.best else "-"
        return (f"{self.name}: {self.verdict}  | fom {self.fom:.3f}, z(chi) {self.z_chi:.1f}, z(q) {self.z_q:.1f}, "
                f"{m.get('n_match', 0)}/{m.get('n_obs', 0)} spots | {tex}")


def _short(path):
    return os.path.splitext(os.path.basename(str(path)))[0]


def load_cif(path) -> Structure:
    st = read_cif(path)
    st.name = _short(path)
    return st


def free_cell_for(struct: Structure) -> tuple:
    """refine a, b, c always; keep angles that are fixed by symmetry (90 or 120 deg in the CIF), e.g. alpha = gamma
    = 90 for monoclinic.  (Equal lengths in tetragonal/cubic cells are still refined independently.)"""
    c = struct.cell.array()
    return tuple(PARAM_NAMES[:3]) + tuple(n for i, n in zip(range(3, 6), PARAM_NAMES[3:])
                                          if not any(abs(c[i] - v) < 0.01 for v in (90.0, 120.0)))


def _grade(k_sig, fom, tested, z_chi=np.nan, z_q=np.nan):
    """descriptive outcome of the two supplementary surrogate diagnostics (not a structure identification)."""
    if not tested:
        return "untested", "surrogate diagnostics not run"
    if k_sig >= 1:
        return "both", "both surrogate diagnostics passed (z >= 3)"
    if np.isfinite(z_chi) and z_chi >= 3:
        return "chi", "orientation (chi) diagnostic passed only"
    if np.isfinite(z_q) and z_q >= 3:
        return "q", "radial (|q|) diagnostic passed only"
    return "neither", "neither surrogate diagnostic passed"


def evaluate(session: Session, struct: Structure, settings: Settings = None, contact=None, with_null=True,
             log=None, max_textures=None) -> Evaluation:
    s = settings or session.s
    log = log or (lambda *_: None)
    t0 = time.time()
    fc = replace(s.fit_config(), free_cell=free_cell_for(struct))
    mc = s.multi_config(**({"max_components": max_textures} if max_textures else {}))
    planes = [[tuple(contact)]] if contact else [low_index_planes(s.planes_maxidx)]
    hist = decompose([struct], session.geom, session.peaks, session.mask, fc, mc, planes, log=log)
    null, k_sig = {}, 0
    if with_null and s.n_null > 0 and any(h is not None for h in hist):
        log(f"  chance test: {s.n_null} surrogates x {len(mc.null_modes)} modes through the same procedure ...")
        null = null_test_multi([struct], session.geom, session.peaks, session.mask, fc, mc, hist, planes, log=log)
        for k in range(len(hist)):
            if all(np.isfinite(v[k]["inc_z"]) and v[k]["inc_z"] >= 3.0 for v in null.values()):
                k_sig = k + 1
            else:
                break
    kk = max(k_sig, 1) - 1
    best = hist[kk] if hist and hist[kk] is not None else next((h for h in hist if h is not None), None)
    fom = best.metrics["fom"] if best else 0.0
    zc = null.get("chi", [{}] * (kk + 1))[kk].get("z", np.nan) if null else np.nan
    zq = null.get("q", [{}] * (kk + 1))[kk].get("z", np.nan) if null else np.nan
    grade, verdict = _grade(k_sig, fom, bool(null), zc, zq)
    ev = Evaluation(struct.name, struct.name, struct, hist, null, k_sig, best, verdict, grade, fom, zc, zq,
                    not struct.has_atoms, time.time() - t0)
    log(f"  => {ev.summary_line()}  ({ev.seconds:.0f} s)")
    return ev


def _rank(evs):
    order = {"both": 0, "chi": 1, "q": 2, "untested": 3, "neither": 4}
    return sorted(evs, key=lambda e: (order[e.grade], -e.fom))


def same_pattern_pairs(structs, qmax=2.0, tol=0.999):
    """pairs whose powder patterns coincide (e.g. R and S enantiomers): diffraction cannot tell them apart."""
    from .cli import indistinguishable
    return indistinguishable(structs, qmax, tol)


def check_cif(session: Session, cif_path, settings=None, contact=None, log=None) -> Evaluation:
    log = log or (lambda *_: None)
    st = load_cif(cif_path) if isinstance(cif_path, str) else cif_path
    log(f"Checking {st.name}  ({st.cell})" + ("  [cell only: no atoms in CIF -> positions only]" if not st.has_atoms else ""))
    return evaluate(session, st, settings, contact, True, log)


def compare_cifs(session: Session, cif_paths, settings=None, log=None):
    log = log or (lambda *_: None)
    structs = [load_cif(p) if isinstance(p, str) else p for p in cif_paths]
    flags = same_pattern_pairs(structs, (settings or session.s).q_max + 0.1)
    for a, b, c in flags:
        log(f"NOTE: {a} and {b} give the same diffraction pattern (corr {c:.4f}) - they cannot be told apart.")
    evs = []
    for i, st in enumerate(structs):
        log(f"[{i + 1}/{len(structs)}] {st.name}  ({st.cell})")
        evs.append(evaluate(session, st, settings, None, True, log))
    return _rank(evs), flags


def load_library(dirs, exclude=r"slab|vacuum", log=None):
    log = log or (lambda *_: None)
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "**", "*.cif"), recursive=True)
    rx = re.compile(exclude, re.I) if exclude else None
    structs, failed, seen = [], [], set()
    for f in sorted(set(files)):
        if rx and rx.search(os.path.basename(f)):
            continue
        try:
            st = load_cif(f)
        except Exception as e:                       # noqa: BLE001 - report, keep going
            failed.append((f, repr(e)[:120]))
            continue
        key = tuple(np.round(st.cell.array(), 3)) + (len(st.species),)
        if key in seen:                              # same structure saved twice
            continue
        seen.add(key)
        st.path = f
        structs.append(st)
    log(f"Library: {len(structs)} structures loaded" + (f", {len(failed)} unreadable" if failed else ""))
    for f, err in failed:
        log(f"   unreadable: {os.path.basename(f)}  {err}")
    return structs, failed


def identify(session: Session, library_dirs, settings=None, exclude=r"slab|vacuum", log=None):
    """which library structure explains the pattern: fast screen of every CIF (1 texture, no chance test), then
    the full procedure + chance test for the best ``identify_null_top``."""
    s = settings or session.s
    log = log or (lambda *_: None)
    structs, failed = load_library(library_dirs, exclude, log)
    flags = same_pattern_pairs(structs, s.q_max + 0.1)
    screen = []
    for i, st in enumerate(structs):
        log(f"screen [{i + 1}/{len(structs)}] {st.name}")
        screen.append(evaluate(session, st, s, None, False, None, max_textures=1))
        log(f"    fom {screen[-1].fom:.3f}  ({screen[-1].best.label() if screen[-1].best else '-'})")
    screen.sort(key=lambda e: -e.fom)
    top = screen[:max(1, s.identify_null_top)]
    log(f"Chance test for the top {len(top)}: " + ", ".join(e.name for e in top))
    full = [evaluate(session, e.struct, s, None, True, log) for e in top]
    rest = screen[len(top):]
    return _rank(full) + rest, flags, failed


# ---------------------------------------------------------------------- export
def export(session: Session, evs, flags, out_dir, log=None):
    import pandas as pd
    from . import plots
    log = log or (lambda *_: None)
    os.makedirs(out_dir, exist_ok=True)
    rows, pair_tables = [], {}
    for i, e in enumerate(evs):
        m = e.best.metrics if e.best else {}
        cell = e.best.cells[0] if e.best and e.best.cells[0] is not None else e.struct.cell
        err = e.best.cell_errs[0] if e.best else np.full(6, np.nan)
        row = dict(rank=i + 1, structure=e.name, verdict=e.verdict, fom=e.fom, z_chi=e.z_chi, z_q=e.z_q,
                   textures=e.best.label() if e.best else "", textures_passing_both_diagnostics=e.k_sig,
                   spots_indexed=m.get("n_match"), spots_used=m.get("n_obs"),
                   strong_missing=f"{m.get('n_strong_missing')}/{m.get('n_strong_pred')}",
                   intensity_rho=m.get("spearman_I_pooled"), rms_dq_cif_pct=m.get("rms_dq_cif_pct"),
                   rms_dq_refined_pct=m.get("rms_dq_fit_pct"), cell_only=e.cell_only, file=getattr(e.struct, "path", ""))
        for j, n in enumerate(PARAM_NAMES):
            row[f"{n}_cif"] = e.struct.cell.array()[j]; row[f"{n}_refined"] = cell.array()[j]; row[f"{n}_err"] = err[j]
        rows.append(row)
        if e.best and e.best.pairs:
            pair_tables[f"{i + 1:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', e.name)}"[:31]] = pd.DataFrame(e.best.pairs)
    with pd.ExcelWriter(os.path.join(out_dir, "gixfit_results.xlsx")) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="ranking", index=False)
        pd.DataFrame(session.peak_rows()).to_excel(w, sheet_name="spots", index=False)
        for k, df in pair_tables.items():
            df.sort_values("q_obs").to_excel(w, sheet_name=k, index=False)
        pd.DataFrame({"notes": [
            "outcome: which supplementary surrogate diagnostics were passed (standardized surrogate score z >= 3 "
            "against observable chi reassignments and against |q| jittered by +-3%); a descriptive outcome, not an "
            "identification of the structure.",
            "fom = (indexed fraction of spot intensity) x (1 - fraction of the strongest predicted spots that are absent)",
            "dq_cif_pct = (q_obs - q_CIF)/q_CIF per spot (the CIF error); dq_fit_pct after refining the cell",
            f"indistinguishable pairs (same diffraction, e.g. R/S enantiomers): {flags or 'none'}"]}).to_excel(
            w, sheet_name="notes", index=False)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"image: {session.path}\nspots: {len(session.peaks)}\n\nHints (no CIF needed):\n")
        fh.write("\n".join("  " + h for h in session.hints()) + "\n\n")
        if flags:
            fh.write("Indistinguishable by diffraction: " + "; ".join(f"{a} = {b}" for a, b, _ in flags) + "\n\n")
        for i, e in enumerate(evs):
            fh.write(f"#{i + 1} {e.summary_line()}\n")
            if e.best:
                c = e.best.cells[0]
                fh.write(f"     CIF     {e.struct.cell}\n     refined {c}\n")
    for i, e in enumerate(evs):
        if not e.best:
            continue
        tag = f"{i + 1:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', e.name)}"
        plots.multi_fit_figure(session.D, session.mask, e.best, session.peaks, os.path.join(out_dir, f"{tag}_map.png"))
        if e.null:
            plots.fom_vs_k_figure(dict(hist=e.hist, null=e.null, model=e.name, geometry=""),
                                  os.path.join(out_dir, f"{tag}_vs_chance.png"))
        if e.best.cells[0] is not None and e.struct.has_atoms:
            write_cif(e.struct.with_cell(e.best.cells[0]), os.path.join(out_dir, f"{tag}_refined.cif"), e.name)
    log(f"Report written to {out_dir}")
