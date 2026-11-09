# Publication readiness for gixfit 0.3.1

The paper's defensible contribution is an integrated candidate unit-cell workflow with measured limitations. The present local revision improves software reliability and reproducibility; it does not establish new experimental accuracy or independent usability.

## Completed in this revision

- Defensive session settings, validation and stale-input rejection, including image replacement at the same path.
- Candidate ordering by FOM rather than surrogate diagnostic category.
- Portable image/model/settings reports with checksum validation and replay.
- Self-contained synthetic tutorial and clearer candidate-ranking language.
- Revised manuscript and SI distinguish benchmark version 0.3.0 from software revision 0.3.1, qualify runtime and conditional sensitivity results, and describe uncertainty as local curvature estimates.

## Evidence needed before submission

1. Freeze a release and rerun relevant paper benchmarks against that exact release. Preserve 0.3.0 results as provenance until the rerun is complete; changes to candidate ordering may affect comparisons.
2. Deposit permitted experimental images, structures, calibration inputs and scripts with stable public identifiers and referee access. Synthetic tutorial data do not replace experimental inputs.
3. Obtain documented installation-to-report exercises by independent researchers. Record assistance, failures, active time, computational time and interpretation. No such reports are claimed here.
4. Add an experimentally calibrated image with independently established structure/cell/orientation. Quantify peak localisation on challenging image shapes where feasible.
5. Validate uncertainty coverage before describing curvature outputs as confidence intervals. Until then retain the narrowed interpretation.
6. Resolve author, funding, provenance and experimental-method placeholders; verify citations and current target-journal requirements before submission.

## Scope of verification

Workflow regression tests, core geometry/structure-factor tests, synthetic recovery, multi-texture recovery without null sampling, and the phase-map regression are exercised locally. Full scientific benchmarks, independent-user trials and complete interactive GUI acceptance are separate tasks. No public release, upload or submission is performed by this revision.
