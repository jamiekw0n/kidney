# Dense comparison of matched augmentation heads

New entry point: `dense_comparison.py`, with local `dense_spatial.py`. Run from the
updated augmentation_matched code folder; do not modify a frozen experiment/source.
All other imported modules are local augmentation_matched modules. Raw images,
training plans, original embeddings and old figures are read-only. No retraining.

Supply an existing completed dense_slide_workflow assembled `*_figures` directory
as --reference. Its manifest must match the original dense source run, sample,
outer fold, UNI2 checkpoint and normalization. It supplies measured normalized
8x8 means, block-averaged H&E and the support mask. ALL means all markers present
in this reference (normally all42); a subset can be requested with --markers.

Two persistent GPU workers own alternating 512-native-pixel bands. Each loads UNI2
once, extracts ORIGINAL unaugmented stride8 context embeddings once, saves float32
features.npy and xy.npy, and applies all3 augmentation outer-test heads to those
same features. Training-view embeddings are stride32 and cannot substitute for the
stride8 inference grid. Earlier whole-slide outputs saved predictions, not these
features, so the first comparison needs fresh encoder inference. There is no TTA.

Outputs in a new directory:
- comparison_plan.json: hashes of reference maps and all3 checkpoints, fold and plan.
- bands/NNNNN/features.npy, xy.npy: original-view features and native center coords.
- bands/NNNNN/{control,geometry,geometry_color}.npy: full-panel predictions, NaN outside support.
- bands/NNNNN/complete.json: checksums and row range; written last.
- worker_*.log: GPU progress.
- figures/marker_NN.png and index.html: H&E | measured | control | geometry | geometry+color.
- complete.json: final success, written after figures.

Scale fixed0..1 for every panel/marker. Numerical predictions remain unclipped;
color saturation only affects display. Dense maps are qualitative, not replacements
for the sampled patient-level OOF quantitative evaluation. No claim that selected
slides represent the cohort. Supported native bottom/right extent is floored to8.

Disk: each supported point adds6144bytes for features plus504bytes for3x42
predictions, plus unsupported prediction-grid cells. The launcher prints the feature
estimate before encoding. Cache is intentionally retained. At most2GPUs plusbounded
512pixel mask bands; plots hold one full single-marker grid at a time. Runtime is
principally one dense encoder pass, not three; worker logs show actual progress.

Use a fresh --out. --resume verifies the plan and completed-band hashes, then skips
completed bands. Partial band directories are preserved and rejected; this version
does not recover partial bands. Use a fresh output directory after a mid-band crash.
If plotting alone fails, no automatic partial-figure overwrite is attempted.
Run only when the allocation's GPUs are free. nohup survives terminal disconnects,
not termination of the compute allocation. Requires matplotlib and original env.

Validation: test_dense_comparison.py has2 synthetic tests for stride8 coordinates,
context support, five-panel gallery output and preserving >1 numerical predictions.
Compilation passed. Real patient and CUDA extraction must be verified remotely.
