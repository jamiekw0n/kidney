# Validation record

Implementation checked locally on 2026-10-22. No remote jobs were submitted.

## Completed checks

- Full unittest discovery: **25 tests passed**, no failures or skips (35.322 s).
- CPU training/checkpoint/prediction tests for weighted and weighted+FDS modes, plus synthetic four-fold cross-validation.
- Exact histogram percentile agreement with NumPy linear percentiles for uint8 and uint16.
- Sparse zero-P99 fallback, constant channels, pixel clipping before block averaging, channel reordering and coordinate preservation.
- Synthetic cache regeneration verifies unchanged features/coordinates/source hashes and rejects raw-target mismatch, overwrite and double normalization.
- Existing expression weighting, FDS and patient partition tests.
- Python compilation, CLI help and Slurm shell syntax checks.

Test interpreter: /Users/jamiekwon/miniconda3/bin/python (Python 3.10.14, PyTorch 2.3.0 CPU). Optimizer subprocess tests ran outside the filesystem sandbox because macOS OpenMP shared memory is unavailable inside it.

The final cohort planning refinement reuses the tested fold reviewer to check exact outer-test patient coverage and reference sample-to-patient mapping; compilation and CLI help were rechecked afterward.

## Required workstation checks

The local test environment lacks tifffile/zarr/imagecodecs. Image-reader tests use an in-memory reader; they do **not** validate real QPTIFF decoding. CUDA, workstation memory, real image headers and cohort completion remain to be verified.

After normalization, run cohort.py collect. Inspect normalization_summary.csv for zero-P99 fallback, low divisors, zero fractions and clipping rates. Confirm all expected samples complete before training. Raw target agreement is checked automatically against the existing cache. Inspect held-out performance and normalized measured/predicted visuals before drawing biological conclusions.

## Interpretation

These tests establish numerical and pipeline behavior, not prediction improvement. Targets and MAE now use relative per-slide normalized units; lower numerical MAE than a 0–255 experiment is not evidence of better predictions. Per-slide normalization can remove technical variation and genuine biological slide-wide differences. The fixed all-pixel support and sparse-channel fallback must be reported with results.

## Source manifest regression fix — 2026-10-23

Two new source-path tests and five normalization tests passed (seven targeted tests).
Covered actual extractor-style missing-mif metadata, recorded source recovery,
identity mismatch, missing provenance and cycles; existing numerical and cache
regeneration tests still pass. compileall passed. Real workstation execution is
still required. No changes to the normalization mathematics or training code.
