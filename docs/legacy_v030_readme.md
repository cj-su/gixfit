# gixfit — automated candidate-structure fitting, lattice refinement and orientation search for GIWAXS

Version 0.3.0.  Python ≥ 3.11; needs numpy, scipy, matplotlib, tifffile, pymatgen (CIF reading), pandas + openpyxl (xlsx).

**Install:** `pip install -e .` in this folder (gives the commands `gixfit` and `gixfit-app`), or run from the folder
with `python -m gixfit ...`.  Check the installation with `python tests/test_core.py` (about 10 s, prints `ALL OK`).

**Reproduce the worked example of the paper (Section 3.1, Figure 1):** see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
Independent users: please report your experience with [`docs/independent_user_report.md`](docs/independent_user_report.md).

## The app (start here)
Double-click **`gixfit_app.bat`** (or run `python -m gixfit app`).  If no window appears, run `gixfit_app_debug.bat`
to see the error.  Optional install: `pip install -e .` gives the commands `gixfit` and `gixfit-app`.

| step | what you do | what you get |
|---|---|---|
| **1 Find spots** | choose the .tif image, enter the detector geometry (or *Load settings…* a `.toml` such as `examples/npb_R1x1_th010.toml`; pick the beam-centre variant from the drop-down), press **Find spots** | 2D map in true q with the spots; table of every spot (q, d, χ, intensity); **structure hints that need no CIF** (out-of-plane spacing, in-plane d-spacings, d-spacings seen at several orientations, crystallinity) |
| **2 Identify** | list your CIF library folders (default: `gixfit/library` + `Rand Lab/.../cif files`), press **Identify structure** | every CIF in the library is screened and ranked; the best N also get the surrogate diagnostics |
| **3 Check one CIF** | choose a CIF (contact plane `auto` or e.g. `1 0 1`), press **Check match** | **per-spot error vs the CIF** (Δq % before and after refinement, χ), refined a, b, c, α, β, γ ± error, overlay of predicted vs observed spots, surrogate diagnostics |
| **4 Compare CIFs** | add ≥ 2 CIFs, press **Compare** | ranking of the CIFs; pairs that diffract identically (R/S enantiomers) are flagged |

Click a row in *Results* to see its overlay, its spot errors and the details.  **Export report…** writes an xlsx, a summary,
overlay figures, data-vs-chance plots and refined CIFs.

*Options:* **max textures** (1–3 orientations per structure), **contact planes up to index** (1–3), **surrogate
diagnostics** (Quick 6 / Standard 12 / Thorough 24 surrogates per mode, or Off; the paper uses 100).

**Surrogate diagnostics (supplementary, descriptive).** Each fit is repeated on surrogate spot sets with the identical
procedure: (χ) polar angles reassigned among the spots so that every spot stays observable; (|q|) every |q| jittered
by ±3 %.  The standardized surrogate score z = (fom − mean)/sd and the exceedance fraction are reported, and the outcome
says which diagnostics reach z ≥ 3: *both passed*, *orientation (χ) passed only*, *radial (|q|) passed only*, or
*neither passed*.  These labels describe the diagnostics, **not** the structure: on held-out simulations a different,
well-ordered structure sometimes passed both, and "neither passed" is not evidence that a structure is absent.

**What it cannot do:** solve an unknown structure that is in no CIF (Identify only searches your library); tell R from S
enantiomers (identical diffraction); use intensities when a CIF has no atomic coordinates (cell-only CSD entries such as
the C60 files are read from cell + symmetry: positions and systematic absences only).

**Run time** (20-core PC): spots ≈ 3 s; one CIF with the Quick chance test ≈ 0.5–4 min (more for low-symmetry cells and
for 2–3 textures); Identify ≈ 1 min per library CIF for the screen plus the chance test for the top N.

## The two problems it is built for
1. **The film cell is not the CIF cell.** Thin-film lattices are strained, solvated or just different from the
   bulk structure.  gixfit measures the error of every indexed peak against the CIF and then *refines*
   a, b, c, α, β, γ against the 2D peak positions.
2. **Polymorphs with similar cells** (e.g. rac vs enantiopure).  Positions alone are often not enough.  gixfit scores
   every candidate structure × contact plane with the same fit, adds an intensity check (|F|² from the
   atoms vs. observed spot intensity), and reports how each score compares with surrogate (scrambled) data.
   *Limit:* R and S enantiomers of the same compound give identical diffraction (Friedel's law, no anomalous
   signal from C/H/N/O).  gixfit flags such pairs as indistinguishable.  It *can* separate a racemic compound
   from an enantiopure crystal, because those have different cells and structure factors.

## Pipeline
| step | module | what |
|---|---|---|
| geometry | `geometry.py` | exact pixel ↔ (q_xy, q_z) for grazing incidence (direct-beam centre, α_i, optional refraction).  No per-axis approximation. |
| peaks | `peaks.py` | despike, mask, difference-of-Gaussians detection, 2D Gaussian fit per spot → sub-pixel centre ± error, integrated intensity |
| model | `crystal.py`, `pattern.py` | CIF / FHI-aims reader, |F(hkl)|², fibre texture for a contact plane (HKL): q_z = G·n, q_xy = √(G² − q_z²); overlapping reflections grouped |
| fit | `fit.py` | Hungarian assignment in (|q|, χ) with a shrinking tolerance; least-squares cell refinement (+ optional fibre tilt, beam centre / SDD); covariance errors; chance-level tests |
| multi | `multi.py` | several textures and/or phases in one pattern: greedy decomposition + joint refinement (one cell per phase), chance level through the same pipeline |
| app | `engine.py`, `app.py` | high-level API (Session, check_cif, compare_cifs, identify, export) and the Tkinter desktop app |
| report | `cli.py`, `plots.py` | peaks.csv, per-fit 2D overlays (CIF vs refined), per-peak error plots, ranking, xlsx, refined CIFs |

## Use
```
python -m gixfit peaks examples/npb_R1x1_th010.toml     # tune peak detection, look at runs/.../peaks_map.png
python -m gixfit run   examples/npb_R1x1_th010.toml     # single texture: every candidate x contact plane
python -m gixfit multi examples/npb_R1x1_th010.toml     # several textures / phases per pattern -> <out_dir>/multi/
python -m gixfit phases examples/rac_on_rubrene_r6.toml # crystals A and B: which crystal each spot belongs to
python tests/test_core.py ; python tests/test_synthetic.py ; python tests/test_multi.py ; python tests/test_phasemap.py
python tests/test_app_gui.py       # opens the app window and presses every button (~6 min)
```
Config (TOML): `[image]`, `[geometry]` + `[[geometry_variants]]`, `[peaks]`, `[fit]`, `[null]`, `[multi]`, `[[candidates]]`.
A candidate is `cif = ...` or `aims = ...` (with `drop_elements`), optional `cell = [a,b,c,al,be,ga]` (overrides the cell
but keeps the fractional coordinates), `contact_planes = [[1,0,1], ...]` or `"auto"` (all planes with indices ≤ `auto_maxidx`).

## Reading the output (`summary.txt`, `gixfit_summary.xlsx`)
* `dq_cif_pct`: per-peak (q_obs − q_CIF)/q_CIF, i.e. **the experiment-vs-CIF error**.  `dq_fit_pct`: the same after refinement.
* `fom` = (√area-weighted fraction of observed spots that are indexed) × (1 − fraction of the `n_strong` strongest
  predicted spots that are absent).
* Reflections inside the missing wedge (e.g. the specular (HKL) itself) are placed at the wedge edge at the same |q|,
  where the mosaic arc meets the Ewald sphere, if they lie within 2× `mosaic_deg` of it.  On-axis reflections can appear
  on EITHER edge, so they are matched against |χ_obs| (one predicted spot, either side; dashed mirror circle in figures).
* `null_chi`: the same full fit on surrogate data with χ reassigned among spots so that every spot stays observable
  (|q| kept); `null_q`: |q| jittered ±3 %.  Every fit refines 6 parameters, so it will always index *something*; the
  standardized surrogate scores show how far the fom lies above what the same procedure gets from surrogates.  They are
  descriptive (the samplers are approximate) and do not establish that the structure is present.  The unrestricted χ
  permutation of versions ≤ 0.2 is still available as mode `chi_free`.
* `spearman_I`: rank correlation of predicted vs observed intensity on matched spots.  This is what separates polymorphs
  whose cells are similar.
* Cell errors come from the Jacobian, scaled by √χ²_red.  The soft prior (`prior_strain`, `prior_angle`) keeps
  under-determined fits near the CIF; set them to `false`/remove to fit freely.

## Several textures / phases in one pattern (`python -m gixfit multi`)
Films often contain several orientations (contact planes) or phases.  `multi.py` explains one pattern with k components
(phase, contact plane):
1. **Greedy step k:** every candidate (phase, plane) gets a cheap pre-score on the spots that are *not yet indexed*.
   The best `prescreen_top` are each added to the current model, the *joint* model is refined, and the one with the
   highest joint fom is kept.  A new component must index ≥ `min_new_match` previously unindexed spots.
2. **Joint refinement:** one Hungarian assignment over the union of all components' predicted spots (each observed spot
   goes to at most one (phase, orientation) group).  Least squares over all matched spots, with **one cell per phase**
   shared by all its textures.
3. **fom** (generalised) = √area-weighted indexed fraction × (1 − fraction of *each* component's `n_strong` strongest
   predicted spots that are absent, pooled).  A texture that predicts many absent spots therefore lowers the fom.
4. **Chance level through the same pipeline:** χ-shuffled and |q|-jittered surrogate spot sets go through the full greedy
   + joint fit (parallel processes, `workers`), so the freedom of choosing k textures out of ~50 planes is in the null.
   For every k the report gives fom vs null (z, p) and the **increment** fom_k − fom_(k−1) vs the null increment.
   "significant textures k" = the largest k for which every increment up to k has z ≥ 3 in *every* null mode.
5. Intensity check: `rho(I)` per component, and pooled with ranks taken within each component (each texture has its own
   unknown volume fraction).

`[multi]` keys: `max_components`, `planes_maxidx`, `prescreen_top`, `min_new_match`, `n_null`, `null_modes`, `workers`,
`models` (default: each candidate alone; `[["A", "B"]]` = phases A and B coexisting).
Outputs in `<out_dir>/multi/`: `summary_multi.txt`, `gixfit_multi.xlsx` (models_by_k, components, cells, pairs),
`multi_fit_*.png` (component-coloured overlay), `fom_vs_k_*.png` (data vs chance bands), refined CIFs.

## Which crystal does each spot belong to? (`python -m gixfit phases`)
For a pattern that contains two (or more) known crystals, e.g. a bilayer "A on B" or two coexisting polymorphs: list
each crystal as a `[[candidates]]` entry.  gixfit fits them jointly (each crystal its own refined cell and contact
plane(s), one shared assignment) and labels **every** detected spot (`gixfit/phasemap.py`):

| label | meaning |
|---|---|
| `A` | only crystal A predicts a spot there (at its refined cell) |
| `A + B` | both crystals predict a spot there: shared / ambiguous; `assigned_phase` says which one the fit used and `share_A`, `share_B` the predicted intensity split |
| `unassigned` | neither crystal predicts a spot there (another phase, substrate, artefact) |

Every listed crystal gets at least one texture: if the greedy fit leaves one out (a large organic cell can absorb the
other crystal's spots), it is added with its best contact plane and the spots are re-assigned; `phase_map_summary.txt`
lists such crystals as *forced*, i.e. assumed by you, not established by the data.  A crystal claims a spot only if it
predicts a reasonably strong reflection there (`[phases] min_I`, default 0.02 of its 10th strongest reflection).
Outputs in `<out_dir>/phases/`: `phase_map.png` (spots coloured by crystal; yellow = shared, grey = none),
`phase_map.csv` (one row per spot: q, χ, d, label, hkl in each crystal, intensity shares), `phase_map_summary.txt`.
Test: `tests/test_phasemap.py` (NPB (101) + α-MoO3 (010) + junk spots: both contact planes found, 48/50 spots labelled
correctly; the 2 errors are MoO3 spots next to an NPB reflection, where MoO3 had only 6 spots to refine its cell).

## Validation
* `tests/test_core.py`: d-spacings of the published NPB cell match the handoff list to 0.002 Å; pixel→q→pixel round trip
  is exact; |F|² agrees with pymatgen's XRD calculator for α-MoO3 (r = 1.0000).
* `tests/test_synthetic.py`: 45 spots from a strained NPB cell (+1.5 % a, −1 % b, +0.5 % c, +0.8/−1.2/+0.5°) + 15 junk spots,
  0.3 px noise.  The fit started from the published cell picks the right contact plane (fom 0.66 vs < 0.08 for wrong planes)
  and recovers every parameter within ≤ 0.003 Å / 0.04°.  (With the current scoring: fom 0.74 vs ≤ 0.08 for wrong planes;
  intensity rank correlation 1.00.)
* `tests/test_multi.py`: one strained NPB cell in two textures, (1 0 1) (35 spots) + (0 0 1) (25 spots, 0.6× intensity),
  + 12 junk spots.  Scanning all 49 planes with |index| ≤ 2 from the published cell, the greedy fit finds (1 0 1) then
  (0 0 1), puts 60/60 true spots in the right component and indexes 0/12 junk spots.  It recovers the shared cell within
  0.004 Å / 0.03°.  fom 0.53 → 0.82 → stops at k = 3.  χ-shuffled null (same pipeline, n = 8): k = 2 increment z = 10.

## Result so far: pure NPB R1 x1, α_i = 0.10° (2026-10-19)
`python -m gixfit multi examples/npb_R1x1_th010.toml` (99 spots in 0.35–1.9 Å⁻¹; 49 contact planes; 12 surrogates per null mode).
No configuration has even ONE texture above chance, so 2–3 textures of the NPB cell do not explain the pattern either:

| cell, beam centre | k = 1 | k = 2 | k = 3 (indexed) | fom (k=1/2/3) | χ-null fom, k = 3 | ρ(I), k = 3 | significant k |
|---|---|---|---|---|---|---|---|
| published, raw j0 = 644 | (1 1 0) | + (0 0 1) | + (0 2 1) (55/99) | 0.123 / 0.152 / 0.181 | 0.145 ± 0.017 (z 2.1) | −0.20 | 0 |
| DFT, raw | (1 0 0) | + (1 −2 1) | + (2 0 −1) (52/99) | 0.114 / 0.137 / 0.132 | 0.126 ± 0.018 (z 0.3) | −0.30 | 0 |
| published, XRD-cal j0 = 660.77 | (1 0 1) | + (2 −2 −1) | + (2 1 −1) (50/98) | 0.130 / 0.134 / 0.112 | 0.133 ± 0.025 (z −0.9) | −0.05 | 0 |
| DFT, XRD-cal | (2 1 1) | + (2 −1 −1) | + (1 2 −2) (50/98) | 0.103 / 0.116 / 0.126 | 0.125 ± 0.018 (z 0.0) | −0.02 | 0 |

* Increments fom_k − fom_(k−1) are within the surrogate increments in every run (|z| ≤ 2.0, never ≥ 3).  The |q|-jitter null is
  at least as high as the real fom in most steps (z −2.7 … +0.8).
* Each added texture indexes 4–21 more spots, but 11–19 of each texture's 20 strongest predicted spots are not observed.
  The pooled intensity correlation stays ≤ 0 (−0.30 … −0.02).  Indexing half the spots with 3 textures is what chance gives.
* Compared with the single-texture `run`: there, the best plane per fixed contact plane beat the χ-shuffled null (z ≈ 3–6).
  In `multi`, the null also chooses its best plane out of 49, and the advantage disappears (k = 1, z ≤ 1.7).
* Conclusion for this image: the published/DFT NPB cell in up to three low-index fibre textures is NOT supported.  The film
  probably needs a different cell (or phase), non-fibre (single-crystal-like) grains, or both (the spots are very sharp, σ ≈ 1 px).

## Known limits / to do
* `multi` is greedy (not a global search over all k-subsets of planes); it also does not refine a fibre tilt per component.
* Intensities: |F|² × fibre Lorentz only (no polarisation, absorption or DWBA).  Use them for ranking, not quantitative refinement.
* The beam centre (j0) is strongly correlated with the cell.  Fix it from a calibrant if possible.  `free_geom = ["y0"]`
  is available, but read its correlation matrix before trusting it.
* Refraction: `alpha_c_deg` applies a simple Snell correction; refined cells can change by more than their Jacobian
  errors between calibrations and refraction settings (paper Section 3.3 and the worked example), so report such ranges.
* The NPB results below were obtained with versions ≤ 0.2 (unrestricted χ surrogates, 12 per mode).
