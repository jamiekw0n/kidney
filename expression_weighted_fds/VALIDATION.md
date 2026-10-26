# Local verification and remaining workstation checks

Created 2026-10-21. No remote cohort job submitted by the coding agent. No real
patient cache, raw image or encoder checkpoint was read while creating this folder.
Existing implementation folders were read to replicate helpers, not imported at
runtime. All program changes are confined to expression_weighted_fds.

Local CPU environment: Python3.10.14, PyTorch2.3.0, NumPy2.1.3, SciPy1.13.0 at
/Users/jamiekwon/miniconda3/bin/python. No installation or environment replacement.

Checks completed:

1. Full then-existing test suite: **19 passed**, zero failures/skips,25.621 seconds.
   Command: python -m unittest discover -s kidney_uni2/expression_weighted_fds -v.
   Actual CPU optimizer/subprocess training required approved execution outside
   the macOS sandbox because OpenMP could not open shared memory inside it.
2. Added matched-pair verifier test: **1 passed**, including rejection of unequal
   expression thresholds. Total verified tests:20 across those two commands.
3. doctor.py --device cpu: forward, weighted loss, backward, finite gradients and
   label-free inference passed.
4. compileall, train/CV CLI help, bash -n on both launchers passed.
5. Source import inspection found no cross-folder imports or path manipulations.

Important behavioral coverage:
- Original 1536->256 GELU/dropout.1->42 architecture,404266 trainable parameters,
  identical initial linear weights and evaluation output to an equivalent old head.
- Same initial weights/dropout random stream during warm-up for weighted/weighted_fds.
- Training-only equal-patient z scores for regression versus raw/255 for binning;
  accidental out-of-range z-score bins rejected and inverse scaling checked.
- Exact previous expression-weighted Huber formula; capped inverse-prevalence
  weights, equal-patient quantile, ties, constant targets, assay thresholds and AP.
- FDS means/variances, momentum, Gaussian kernel against SciPy, first-late-bin
  initialization, empty/singleton support, calibration gradients, marker-specific
  output conditioning and zero-variance bypass.
- Evaluation does not require/use labels or mutate FDS state. Eval-time moment
  collection/update attempts raise. Shared features are not mutated in place.
- Actual CPU training of both modes on synthetic four-patient caches; on/off
  boundaries, delayed selection, z-score checkpoint, reload and held-out prediction.
- Complete synthetic four-fold CV, original-unit predictions, aggregate/stratified
  metrics, successful completion manifest, patient leakage and overwrite rejection.
- Input cache hashes unchanged after training/inference.
- Pair checker rejects differing expression states despite otherwise matched runs.

Workstation still required: full tests/doctor in the actual environment (launcher
repeats them), CUDA training,39-patient fold/inventory validation, source/input
provenance review and real biological performance evaluation. CPU synthetic
success does not prove GPU compatibility, convergence or improved expression maps.
The original runtime proof above did not include dense encoder inference; the
optional dense extension was added afterward, as documented below. Existing sparse OOF plot renderer was
replicated with new labels; actual new-run galleries will be created only after
both workstation training runs complete.

## Same-ROI and OOF visualization extension, 2026-10-21

Added optional local dense inference and automatic completed-run resolution.
No training, FDS, weighting or regression-scale logic was changed. The launcher
selects exactly one complete run of each mode or refuses ambiguity; explicit
paths remain supported. The matched-pair check runs before any plotting job.

Full suite rerun: **24 tests passed**, zero failures/skips,20.649 seconds in the
same CPU environment. New checks verify exact saved ROI resolution and150000
centers for the earlier F03600 coordinates, reference cycles and patient leakage,
new-format head mode/provenance validation, frozen/evaluation head behavior,
completed-run selection, ignored partial runs and ambiguity rejection.
A separate actual Matplotlib smoke run generated four synthetic sample PNGs,
HTML gallery and both normalized report sets. Nonfatal local font-cache warnings
did not prevent rendering. Both launchers use writable MPLCONFIGDIR on scratch.
CLI help, compileall and bash -n on both plotting launchers passed.

Real dense encoder inference, registered image reads, CUDA execution and actual
patient figure appearance remain workstation-only checks; none were performed
locally. Previous raw/cache files and earlier implementation folders were not
modified. The entire updated folder must be copied so replicated image I/O and
raw-safety helpers accompany the new dense scripts.
