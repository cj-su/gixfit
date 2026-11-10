# gixfit

Candidate unit-cell fitting and orientation search for fibre-textured GIWAXS films.
Version 0.3.1 requires Python 3.11 or later.

gixfit detects spots in a detector image, maps them into reciprocal space, compares candidate CIFs, searches contact planes and refines unit cells. It provides a Tkinter desktop interface and Python/command-line workflows. It does not solve an unknown structure or refine atomic coordinates.

## Install and start

From this folder, using the Python environment in which you intend to run gixfit:

```console
python -m pip install .
python -m gixfit app
```

Dependencies are declared in pyproject.toml: NumPy, SciPy, matplotlib, tifffile, pymatgen, pandas and openpyxl. Tkinter must be available in your Python installation. The Windows batch launchers are conveniences; launching through the installed Python environment avoids interpreter ambiguity.

## Try the self-contained example

The tutorial requires no laboratory files. It uses an invented carbon structure and a detector image generated with gixfit's own forward model. It demonstrates operation, not independent scientific validation.

```console
python -m gixfit.demo tutorial --run
python -m gixfit.replay tutorial/gixfit_synthetic_tutorial/report/run_manifest.json tutorial/replayed
```

The fixed tutorial produced 38 detected spots and a (100) contact plane, with FOM 0.800 in the tested environment. Inspect the replay's printed FOM difference; dependency differences may affect numerical results.

In the desktop app, choose **Try demo...**, select an output folder, press **Find spots**, then **Check one CIF > Check match**. Inspect the overlay, unmatched spots and refined cell before exporting.

## Analyse an experimental image

1. Select the image and load or enter its measured detector geometry. The default geometry is an example, not your calibration.
2. Press **Find spots** and inspect the detected centres and masks.
3. Select a CIF and contact plane (or auto), compare several CIFs, or use **Rank candidate CIFs** on a supplied library.
4. Inspect fit quality, missing strong reflections, contact plane and cell changes. A high-ranking candidate is not proof of identity.
5. Use **Export report...** to save tables, figures, refined CIFs and a portable replay bundle.
6. Use **Export figure (preview, intensity bar)...** to save the map shown (with the selected fit) as PNG, TIFF, PDF or SVG. The dialog previews the saved figure and lets you set the colour-bar limits, the number of hkl labels (none, 8, 20 or all: the strongest indexed spots inside the shown q range, counted over all crystals together; it does not choose particular hkl, and "all" can overlap) (percentile sliders or typed values), log or linear scale, colour map, overlay/labels/legend, title, q range, size and dpi. From Python: `gixfit.figexport.save_map(path, session, evaluation, FigureOptions(...))`.

Changed geometry, detection settings, image paths or image contents require spots to be recomputed. Candidate results are ordered by descending FOM; optional surrogate diagnostics are annotations. Diagnostics are off by default. The existing multi-texture component-count selection still depends on surrogate increments when diagnostics are enabled, so changing diagnostics can change that selected model.

## Replay and provenance

Desktop/API exports copy the input image and record its SHA-256 checksum, effective settings, in-memory candidate models, fixed/automatic plane choice, fitting defaults, seed and dependency versions in run_manifest.json. Move the complete report folder to retain its relative input path. Replay checks the image and fitting defaults, warns about environment changes, and reruns the recorded candidates. It does not repeat a search of a changing CIF library. Legacy CLI workflows retain their separate configuration/output conventions.

## Interpretation and limits

- FOM combines observed-spot coverage and missing strong predictions. Compare it only across fits made with comparable settings; tolerance changes can substantially alter it.
- Surrogate scores describe agreement relative to selected scrambled data. A different ordered structure can pass; failure does not establish absence.
- Reported cell errors are local curvature estimates conditional on assignments, geometry and priors. They are not validated confidence intervals and do not include calibration/model uncertainty.
- R/S enantiomers can be indistinguishable under this diffraction model. Cell-only CIFs lack atomic intensity information.
- The fibre model does not describe all single-crystal grains, rods, overlaps or polymorphs. An unassigned spot does not identify the missing physical mechanism.
- The intensity model includes a fibre factor and optional polarisation (off by default); it omits absorption and DWBA. Optional Snell refraction is not a full transmitted/reflected-beam treatment.

## Validation and paper

Run from a source checkout:

```console
python -X utf8 tests/test_workflow.py
python -X utf8 tests/test_core.py
python -X utf8 tests/test_synthetic.py
python -X utf8 tests/test_multi.py --no-null
python -X utf8 tests/test_phasemap.py
```

These cover application state, ranking, moved-report replay, geometry, structure factors and model-generated recovery. They do not establish independent-user usability, experimental absolute accuracy or clean dependency installation. The full manuscript benchmark was generated with 0.3.0 and has not been regenerated for 0.3.1.

[Experimental worked example](docs/REPRODUCE.md) currently requires author-supplied inputs whose public deposition is unfinished. [Independent-user report](docs/independent_user_report.md) records installation, assistance, task time and interpretation. [Publication readiness](docs/PUBLICATION_READINESS.md) distinguishes completed improvements from evidence still needed.

[Legacy reference](docs/legacy_v030_readme.md) preserves detailed CLI/configuration descriptions and historical results. Its significance language and historical numerical results should not be read as current validation claims.
