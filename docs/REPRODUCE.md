# Reproducing the worked example (paper Section 3.1, Figure 1)

A rubrene film on ITO (22 nm on a 5 nm TPTPA layer), measured at the CMS beamline (NSLS-II), α_i = 0.12°, is taken
from the raw detector image to a refined cell with the public gixfit API.  Expected run time: about 20 s for the
workflow, plus about 5 min on 3 cores for the optional surrogate diagnostics (2 × 100 surrogates).

## 1. Install

```
python -m pip install -e .            # in the gixfit folder; Python >= 3.11
python tests/test_core.py             # prints ALL OK
```

## 2. Get the inputs

| file | where |
|---|---|
| raw image `YLL_R5_pos1_x0.000_th0.120_10.00s_2020855_waxs_stitched.tiff` | [AUTHOR INPUT NEEDED: public data repository and DOI] |
| candidate structure `rubrene_relaxed_PBE+MBD.cif` (DFT-relaxed bulk rubrene) | [AUTHOR INPUT NEEDED: deposited with the image, or its source] |
| reference used only to judge the result | published 293 K bulk structure, Jurchescu et al., Acta Cryst. B62 (2006) 330: a = 26.86, b = 7.193, c = 14.433 Å |

## 3. Run

```
python examples/worked_example_rubrene.py <image.tiff> <rubrene.cif> <output_folder>
```

The three decisions a user makes are fixed in the script: the detector geometry (x0 = 464.60 px, y0 = 646.67 px,
L = 258.37 mm, from the ITO (222) ring of the same image; `examples/ito_ring_calibration.py` derives it), the candidate
CIF, and the analysed range 0.3 ≤ |q| ≤ 1.85 Å⁻¹.  All other settings are the defaults and are written to
`<output_folder>/settings.toml`.  Set `GIXFIT_WORKERS` to the number of cores to use for the diagnostics.

## 4. Compare with the paper

`<output_folder>/summary.json` must give:

| quantity | paper value |
|---|---|
| spots found / used | 98 / 25 |
| selected contact plane | (1 0 0), FOM 0.238 (next plane 0.072) |
| indexed spots, strong predicted spots unobserved | 12 of 25, 10 of 20 |
| refined a, b, c (Å) | 26.935, 7.201, 14.377 |
| deviation from the 293 K reference | +0.28 %, +0.11 %, −0.39 % |
| range over the 16 calibration / refraction scenarios | a −1.7 … +0.8 %, b −0.4 … +0.7 %, c −0.4 … +3.4 % |
| surrogate diagnostics (2 × 100) | orientation diagnostic passed only (z_χ ≈ 7, z_q ≈ 0.3) |

The fit is deterministic.  The surrogate scores depend on the random seeds (fixed by the settings, so a rerun gives
the same values); two independent seed sets of 100 surrogates gave z_χ = 6.9 and 8.2 for this image.

## 5. Report

Please fill in [`independent_user_report.md`](independent_user_report.md) and send it to the authors
(cj.su@princeton.edu).
