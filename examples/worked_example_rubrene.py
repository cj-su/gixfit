"""End-to-end worked example (paper Section 3.7): a rubrene film on ITO, from the raw detector image to a refined cell,
using only the public gixfit API, as a user would.

    python examples/worked_example_rubrene.py [image.tiff] [rubrene.cif] [out_dir]

Inputs (defaults: the files used in the paper; raw image to be deposited, see the paper's availability statement):
    image : YLL_R5_pos1_x0.000_th0.120_10.00s_2020855_waxs_stitched.tiff  (CMS / NSLS-II, alpha_i = 0.12 deg)
    CIF   : rubrene_relaxed_PBE+MBD.cif  (bulk rubrene relaxed by DFT; NOT the experimental structure)
Independent reference: the room-temperature (293 K) crystal structure of rubrene, Jurchescu et al., Acta Cryst. B62
(2006) 330: a = 26.86, b = 7.193, c = 14.433 A (Cmca).  It is used only to judge the result, never in the fit.

User decisions (the only manual steps): (1) the detector geometry, from the ITO (222) ring of the same image
(examples/ito_ring_calibration.py: x0 464.60, y0 646.67 px, L 258.37 mm); (2) the candidate CIF; (3) the analysed
|q| range (0.3-1.85 A^-1, below the ITO ring).  Everything else uses the default settings written to settings.toml.
Writes to out_dir: settings.toml, summary.json, worked_example.pkl, and the standard gixfit report (export()).
"""
import os, sys, time, json, pickle
from dataclasses import replace
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gixfit.engine import Settings, Session, load_cif, evaluate, export, free_cell_for       # noqa: E402
from gixfit.fit import Problem                                                               # noqa: E402
from gixfit.geometry import Geometry                                                         # noqa: E402
from gixfit.multi import unique_planes                                                       # noqa: E402
from gixfit.pattern import low_index_planes                                                  # noqa: E402

RAND = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Rand Lab", "Chiao-Jung (Jason) Su")
IMAGE = os.path.join(RAND, "GIWAXS Data", "202506 Heterojunction", "stitched",
                     "YLL_R5_pos1_x0.000_th0.120_10.00s_2020855_waxs_stitched.tiff")
CIF = os.path.join(RAND, "cif files", "CMU Rubrene", "rubrene_relaxed_PBE+MBD.cif")
REFERENCE = np.array([26.86, 7.193, 14.433])                       # 293 K, Jurchescu et al. 2006
CALIBRATIONS = {"ITO ring of this image": (464.60, 646.67, 258.37),
                "ITO ring of a bare substrate (R11, same run)": (464.38, 646.83, 257.31)}


def main(image=IMAGE, cif=CIF, out=os.path.join(ROOT, "examples", "worked_rubrene")):
    os.makedirs(out, exist_ok=True)
    T = {}
    s, _, _ = Settings.from_toml(os.path.join(ROOT, "examples", "rac_binap_r20.toml"), variant=1)
    s.x0, s.y0, s.sdd_mm = CALIBRATIONS["ITO ring of this image"]
    s.max_textures, s.planes_maxidx = 1, 2
    s.n_null, s.null_modes, s.workers = 100, ("chi", "q"), int(os.environ.get("GIXFIT_WORKERS", "6"))
    s.to_toml(os.path.join(out, "settings.toml"), image_path=image)

    t = time.time(); ses = Session(image, s, log=print); T["spot finding"] = time.time() - t
    hints = ses.hints()
    st = load_cif(cif)

    # orientation search: figure of merit of every symmetry-distinct contact plane (indices <= 2)
    t = time.time()
    fc = replace(s.fit_config(), free_cell=free_cell_for(st))
    planes = unique_planes(st, low_index_planes(2), fc)
    scan = []
    for pl in planes:
        r = Problem(st, pl, ses.geom, ses.peaks, ses.mask, fc).run()
        scan.append((tuple(pl), r.metrics["fom"]))
    scan.sort(key=lambda x: -x[1])
    T["orientation search (all planes)"] = time.time() - t

    t = time.time(); ev = evaluate(ses, st, s, with_null=False, log=print); T["assignment + refinement"] = time.time() - t
    t = time.time(); evd = evaluate(ses, st, s, with_null=True, log=print); T["surrogate diagnostics (2 x 100)"] = time.time() - t

    # calibration / refraction sensitivity of the refined cell at the selected contact plane
    t = time.time()
    contact = ev.best.components[0][1]
    sweep = []
    for cname, (x0, y0, L) in CALIBRATIONS.items():
        for free in ((), ("y0",)):
            for ac in (None, 0.09, 0.10, 0.11):
                g = Geometry(ses.geom.wavelength, L, ses.geom.pixel_mm, x0, y0, ses.geom.alpha_i_deg, ac)
                r = Problem(st, contact, g, ses.peaks, ses.mask, replace(fc, free_geom=free)).run()
                sweep.append(dict(calibration=cname, free_geom=list(free), alpha_c=ac, cell=r.cell.array()[:3].tolist(),
                                  fom=r.metrics["fom"]))
    T["calibration / refraction sweep (16 fits)"] = time.time() - t
    t = time.time(); export(ses, [evd], [], os.path.join(out, "report"), log=print); T["report"] = time.time() - t

    c0, c1, ce = st.cell.array()[:3], ev.best.cells[0].array()[:3], np.asarray(ev.best.cell_errs[0])[:3]
    S = np.array([w["cell"] for w in sweep])
    summary = dict(
        image=os.path.basename(image), cif=os.path.basename(cif), n_spots=len(ses.peaks), n_used=ses.n_used(),
        hints=hints, contact_plane=list(contact), best_planes=[[list(p), f] for p, f in scan[:5]],
        fom=ev.fom, n_indexed=ev.best.metrics["n_match"], n_obs=ev.best.metrics["n_obs"],
        n_strong_unobserved=ev.best.metrics["n_strong_missing"], intensity_rank_corr=ev.best.metrics["spearman_I_pooled"],
        cell_cif=c0.tolist(), cell_refined=c1.tolist(), cell_err_jacobian=ce.tolist(),
        cell_range_over_scenarios=[S.min(0).tolist(), S.max(0).tolist()], reference_293K=REFERENCE.tolist(),
        dev_cif_from_reference_pct=(100 * (c0 / REFERENCE - 1)).tolist(),
        dev_refined_from_reference_pct=(100 * (c1 / REFERENCE - 1)).tolist(),
        dev_range_from_reference_pct=[(100 * (S.min(0) / REFERENCE - 1)).tolist(), (100 * (S.max(0) / REFERENCE - 1)).tolist()],
        diagnostics=dict(outcome=evd.verdict, z_chi=evd.z_chi, z_q=evd.z_q, n_surrogates=s.n_null),
        runtime_s=T, cpu_workers=s.workers,
        user_decisions=["detector geometry from the ITO (222) ring of the same image",
                        "candidate CIF (DFT-relaxed bulk rubrene)", "analysed |q| range 0.3-1.85 A^-1"])
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    pickle.dump(dict(summary=summary, scan=scan, sweep=sweep, peaks=ses.peaks, geom=ses.geom, mask=ses.mask,
                     best=ev.best, image=image), open(os.path.join(out, "worked_example.pkl"), "wb"))
    print(json.dumps({k: summary[k] for k in ("contact_plane", "fom", "cell_refined", "dev_refined_from_reference_pct",
                                              "dev_range_from_reference_pct", "diagnostics", "runtime_s")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
