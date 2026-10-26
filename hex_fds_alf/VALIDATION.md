# Verification and execution status

Date: 2026-10-20. No real kidney cohort run or GPU run has been submitted or
executed while creating this folder. Existing raw images and feature caches were
not opened for this implementation. All runtime checks used explicitly synthetic
features, measured targets and patient IDs in temporary directories.

## Successful local environment

- Python 3.10.14, executable /Users/jamiekwon/miniconda3/bin/python
- PyTorch 2.3.0, CPU
- NumPy 2.1.3
- SciPy 1.13.0

The pre-existing environment was used; no packages were installed or replaced.
The usual local python3 lacks PyTorch; use the workstation's kidney-uni2 executable
rather than assuming any python executable has the correct dependencies.

## Completed checks

Command:

```
/Users/jamiekwon/miniconda3/bin/python -m unittest discover -s kidney_uni2/hex_fds_alf -v
```

Result: **17 tests passed, zero failures, zero skips**, 32.797 seconds in the final
full run. Test durations are hardware-specific, not cohort runtime estimates.
The actual training subprocesses required execution outside the macOS sandbox
because OpenMP could not open shared memory inside it. Numeric forward/backward
tests also ran directly. This was an execution-environment limitation, not an
optional suppression of a failed numerical or training check.

Coverage includes:

- ALF normalization against independent scipy.integrate.quad across alpha values
  .001, .003, .01, .05, .1, .5, 1, 1.5, 1.99, 1.999; absolute logZ tolerance 3e-7.
- Approach to analytic Cauchy/Gaussian normalizers and alpha=1 closed-form rho.
- Zero-residual NLL equals log(scale)+logZ, ruling out missing normalizer terms.
- Finite, nonzero gradients for head residual, adaptive shape and adaptive scale;
  large-residual finite checks and independent gradient comparisons.
- Shifted affine-softplus parameterization matches the reference util function.
- Gaussian kernel agreement with SciPy for both default (9,2) and smaller (3,1).
- Bucket boundary handling including target=1; sample mean/variance; EMA;
  first observation occurring late; monotonic updates; singleton/empty bins.
- Exact calibration formula, gradients through calibrated features, no in-place
  mutation, zero-variance bypass, marker-specific output conditioning.
- Inference independence from supplied labels and no FDS buffer mutations;
  explicit refusal to collect/update FDS in evaluation mode.
- Real CPU optimizer training of control, fds, alf and fds_alf modes on four
  synthetic patient caches, using a bounded four-epoch schedule.
- FDS turns on and off at the intended boundaries; checkpoint selection occurs
  only after its minimum epoch; ALF state saved only in appropriate modes.
- Strict checkpoint loading and target-free held-out prediction; training
  patients refused in held-out inference; cache hashes unchanged afterward.
- Complete four-fold synthetic CV with outer predictions, primary regression
  metrics, expression-stratified reports and successful completion manifest.
- Refusal of overlapping patient splits, inconsistent schedule and output overwrite.

Also passed: doctor.py normalization/backward preflight, CLI help for train and CV,
Python compileall and bash -n on run_cv.sbatch.

A separate synthetic rendering smoke test created four sample PNG figures, an HTML
gallery and both normalized metric reports using compare_predictions.py. Synthetic
images were temporary; they are not biological validation or presentation results.
Matplotlib 3.6.1 was used for that check. Nonfatal font-cache write warnings occurred
inside the local sandbox; the workstation should use a writable MPLCONFIGDIR when
plotting, e.g. its existing /scratch/.../cache/matplotlib directory.

## Required workstation evidence

1. Run doctor.py, the full test suite and the launcher in the existing environment.
   The launcher repeats those checks and records pip freeze and job/source metadata.
2. Verify the real 39-patient inventory and existing folds pass without exclusions.
3. Inspect per-marker training_bin_support.json, including DISTINCT PATIENTS in
   high-intensity bins. Repeated tiles cannot establish independent support.
4. Check finite objectives/gradients, ALF shape/scale trajectories, FDS on/off epochs,
   eligible checkpoint selection and successful final manifests for all five folds.
5. Compare matched control/FDS/ALF/combined results with unchanged evaluation strata.
   Inspect rare-expression recovery AND low-expression false positives and bias.
6. A real cohort success does not establish cell-level localization, calibrated
   confidence, or HEX-equivalent performance. Preserve an untouched evaluation
   cohort for final conclusions if development continues based on these outer results.

CUDA execution, numerical performance on A100, real-data convergence, runtime,
biological improvement, and compatibility with a different dependency environment
remain unverified until the workstation run. No resume or dense-ROI feature
extraction capability is advertised in this folder.

## Visualization extension, 2026-10-21

User-reported real cohort runs control_job_48680643 and fds_alf_job_48680648 both
completed, each with 39 patients and five folds. The coding agent has not directly
inspected those remote artifacts. New launchers use those exact run directories.

The self-contained dense extension adds read-only TIFF helpers, the exact saved
F03600 ROI resolver, and strict new-format control/FDS+ALF checkpoint loading.
Head training and its defaults are unchanged. The optional frozen encoder is
loaded only for dense inference. Three new synthetic checks verify saved ROI
resolution (including 150,000 centers for the earlier coordinates), reference
cycle/fold leakage rejection, expected experiment modes, encoder provenance,
evaluation mode and frozen head parameters. No measured targets are passed to
heads during inference.

Full test suite rerun: **20 passed, zero failures/skips**, 24.519 seconds, in the
same CPU environment documented above. Both new sbatch files passed bash -n;
compileall and CLI help passed. New actual-patient OOF and dense galleries must
be generated on the workstation after copying the updated folder. Dense encoder
execution, raw-image I/O and figure appearance for real F03600 data were not run
locally. Those are not established by the synthetic checkpoint tests.
