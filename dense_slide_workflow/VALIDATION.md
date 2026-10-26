# Verification — 2026-10-23

Final full suite: all27 local tests passed in31.552seconds with Python3.10/PyTorch2.3 CPU.
Tests include actual weighted and weighted+FDS head training, synthetic four-fold
OOF export, isolated fold training for arrays, held-out inference/leakage rejection,
expression/FDS math, exact tissue integral fractions, deterministic reservoir
selection, inference grid/band coverage, synthetic prepared-target regeneration
and mocked feature extraction, unchanged raw placeholder hashes, whole-slide PNG
assembly with NaN/valid masks and fixed limits, and mocked Slurm CPU/GPU/dependency
arguments and the full mocked automatic dependency chain. These are synthetic, not real patient results.

The final fold-output guard was narrowed to protect the reference folds FILE while
allowing sibling output RUN/cv/part_XX; an explicit real-CPU-training regression test
exercises exactly this layout. All4 training/CV tests were rerun afterward and passed
(19.615seconds). Later band progress-log additions do not change numeric operations.
Python compileall, CLI help and bash -n also passed.

Local runtime lacks tifffile/zarr/imagecodecs, so preparation's reader is mocked.
Real QPTIFF axes/decoding, frozen UNI2-h GPU forward, scheduler policies/dependencies,
actual slide dimensions/storage/memory and complete workstation runs are unverified.
Do the per-patient preparation/extraction pilot before the remaining cohort. Check
actual supported band counts/crops-per-second before promising a Monday ETA.

Training subprocess tests ran outside the sandbox to allow macOS OpenMP shared
memory. No remote jobs were launched and no patient results were generated locally.
