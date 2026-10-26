# Matched image augmentation with frozen UNI2-h

Three new experiments, entirely separate from existing code/caches/results:

1. control: original embeddings only.
2. geometry: each sampled location uses original or a fixed geometry view,50/50.
3. geometry_color: original or geometry+color view,50/50. The geometric transform
   is IDENTICAL to geometry-only for a given location.

## Fixed scientific protocol

Same existing stride32-selected locations/cap20,000 per sample, normalized central
8x8 targets, patient folds, model1536->256 GELU/dropout0.1->42 and expression-weighted
standardized Huber. FDS remains OFF. AdamW lr.001/weightdecay.0001; batch256;
max100epochs/minimum51/patience10. Same initialization seed42+fold and location
sampler draws across variants, same updates per epoch. Early stopping may yield
different total epochs; audits report actual epochs and compare shared epoch draws.
Target z scores and expression weights fit UNIQUE ORIGINAL training rows, not views.
Validation and outer-test inference use original embeddings only; no test-time
augmentation. New control retrained, not historical weights reused as a matched arm.

There are15 head fits (3variants x5folds). Extra views are not extra patients or
locations. Same sampled index may appear multiple times under patient-balanced
replacement sampling; each occurrence independently selects identity or augmentation.
View selection uses NumPy RNG separate from Torch sampler/dropout. Original targets
and their patient sampling remain fixed. View caches are pointwise transformations;
no held-out data is used to fit augmentation/stain statistics. Although views are
cached for all patients, each fold's val/test loader never uses augmented views.

Geometry is one of7 nonidentity transforms from the8-element square D4 group:
90degree multiples plus optional reflection. No arbitrary-angle interpolation,
translation, random crop or zoom. Exact central8x8 square preserved; scalar marker
means unchanged. Same native128 H&E is used. Color on H&E only: brightness/contrast/
saturation uniform0.9..1.1, hue uniform-0.01..0.01. Recipe order: crop->D4->brightness,
contrast,saturation,hue->same square224 bilinear resize->ToTensor->ImageNet normalize.
UNI2-h weights and original preprocessing remain frozen. Parameters deterministic
from SHA256(seed,sample,nativeX,nativeY), independent of scheduling/batch order.
One geometry/color realization per location is cached, not infinitely new color
jitter each epoch. Identity remains available half the draws.

## Efficiency and storage

- Existing identity embeddings referenced read-only. NEVER removed/replaced/copied
  into new identity cache. Ground-truth targets/coordinates reused unchanged.
- Reads saved prepared/he.npy arrays with mmap, not original TIFF or QPTIFF.
- Two new1536 float32 vectors per location; expected extra bytes=N*1536*4*2.
  At780,000 locations <=9.59GB new vectors (decimal), plus small metadata/parameters.
- Geometry+color share crop read and geometric orientation. Batch16 means TOTAL
  encoder images (8locations x2views), not16locations/32images.
- Two GPU workers, one persistent UNI2-h per worker for its patient queue; no model
  reload per patient. Original view needs zero additional encoder passes.
- Head training uses original RAM arrays and lazy read-only mmap augmented arrays;
  it does not concatenate all3views or triple epoch length.
- Original/view SHA256 checks once per verification stage; per-fit stat checks guard
  against changes afterward, avoiding rehashing all augmented arrays15times.
- Two head jobs in parallel, one per GPU; no uncontrolled GPU process multiplication.
- No new target prep, raw mIF reads, whole-slide inference or network downloads.
  Whole-slide visualizations for these new checkpoints are a later step after
  comparing results; old checkpoint loaders reject the new format on purpose.

## Safety / provenance / resume

Plan requires a new output directory and checks original feature shards, sample/
patient splits, checkpoint hash, context bounds and derived-H&E size/mtime matching
original extraction provenance. Source snapshot is frozen at PLAN/source; all
subprocesses execute it. Cache/model fingerprints identify this recipe separately.

New views stored PLAN/views/SAMPLE/ with uncompressed mmap-friendly .npy files,
parameters.npz and completion/checksum manifests. Original labels remain in the
source cache. Full sample completion written only after every shard succeeds.
--resume-complete reuses verified complete samples AND committed shards of partial
samples for the same fingerprint. A crash mid-write can leave orphan features;
these are NEVER removed. A new unique attempt filename avoids overwrite; only
committed checksummed records are reused. Corrupt committed data is rejected.
Original/source files are checked before/after generation. No existing outputs
are deleted, chmodded, retiled or normalized in place.

A per-plan OS lock prevents concurrent launchers. Lock automatically releases on
process exit without deleting cache artifacts. Complete head fits may be skipped
with --resume-complete; partial head directories are rejected/preserved, require
review. This is not optimizer-state resume of interrupted head training.

## Launch on the two-GPU code-server allocation

Copy WHOLE augmentation_matched folder to:
/scratch/groups/emmalu/hande_annotation/kidney_uni2/augmentation_matched
Use existing kidney-uni2 env (torch/torchvision/timm/NumPy/Pillow/scipy).
Matplotlib optional for comparison plots. No install or new environment required.
Wait for prior two-GPU image inference to finish. nohup does not outlive the
code-server allocation. Check its remaining wall time and GPUs before launching.

```bash
BASE=/scratch/groups/emmalu/hande_annotation
PY="$BASE/envs/kidney-uni2/bin/python"
AUG_CODE="$BASE/kidney_uni2/augmentation_matched"
SOURCE_RUN="$BASE/uni2_dense_training/stride32_cap20000_v1"
AUG_RUN="$BASE/uni2_augmentation_runs/geometry_color_v1"

"$PY" "$AUG_CODE/pipeline.py" plan \
 --source-run "$SOURCE_RUN" \
 --checkpoint "$BASE/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin" \
 --out "$AUG_RUN"

# Plan prints real location count and required extra feature storage.
# Freeze further execution to this experiment snapshot:
AUG_CODE="$AUG_RUN/source"
PILOT_LOG="$AUG_RUN/pilot.log"
nohup "$PY" -u "$AUG_CODE/run_direct.py" --plan "$AUG_RUN" --stage pilot \
 > "$PILOT_LOG" 2>&1 &
tail -f "$PILOT_LOG"
```

Pilot uses GPU0 and only the first cohort sample. Watch the printed logs directory
features_gpu0.log for image throughput, and inspect PLAN/views/SAMPLE/preview.png
(original | geometry | geometry+color). It must look biologically plausible.
PILOT COMPLETE means the TWO augmented embeddings per location are cached/verified
before scaling up, not head training completed.

After pilot succeeds:

```bash
AUG_LOG="$AUG_RUN/all_$(date +%Y%m%d_%H%M%S).log"
nohup "$PY" -u "$AUG_CODE/run_direct.py" --plan "$AUG_RUN" \
 --stage all --resume-complete > "$AUG_LOG" 2>&1 &
tail -f "$AUG_LOG"
```

Reuses pilot cache, generates remaining views on both GPUs, verifies caches, runs
all15head fits two at a time, performs original-view outer scoring and aggregates.
No sbatch/submit/scheduler calls. --device-count1 allows one-GPU execution. To stop
after features use --stage features; later --stage train starts head fits only.
No default claim about runtime: pilot logs provide measured image throughput;
new encoder work is2N images, not100N online epochs. Queue is local to allocation.

## Outputs / evaluation

PLAN/runs/VARIANT/fold_XX/training.log: live epochs.
PLAN/runs/VARIANT/fold_XX/training/{best.pt,history.json,target_scaler.json,
expression_state.json,initial_state.json}: model/protocol records.
PLAN/runs/VARIANT/fold_XX/outer_predictions/SAMPLE/: unaugmented held-out predictions.
PLAN/runs/VARIANT/aggregate/primary_summary.csv and primary_patient_metrics.csv:
normalized MAE/RMSE, Pearson/Spearman, bias/calibration; equal-patient means/CIs.
expression_summary.json: high/low MAE and detection tradeoffs with training thresholds.
PLAN/comparison/all_marker_metrics.csv: all3variants xall42markers.
PLAN/comparison/paired_differences.csv: candidate-minus-baseline differences per
marker/metric, shared-patient mean and95%conditional patient-bootstrap intervals.
Negative MAE delta favors candidate; positive correlation delta favors candidate.
Bias must be interpreted relative to zero; higher bias not automatically better.
No multiple-testing-adjusted significance or guaranteed augmentation improvement.
PLAN/comparison/matched_audit.json: initialization/scaler/weights/config match,
sampler draw agreement, actual epoch counts. Failing match aborts completion.
PLAN/complete.json only when all3experiments aggregate/audit successfully.

```bash
"$PY" "$AUG_CODE/plot_comparison.py" --plan "$AUG_RUN"
```

Writes comparison/matched_metrics.png and vectorPDF. Compare original-view outer
metrics first. Select augmentation strength with inner validation, not by picking
favorite outer-test slide figures. No test-time augmentation in this experiment.

## Verification boundaries

See VALIDATION.md. Local tests use synthetic images and mock UNI2 features, plus
REAL CPU optimization/checkpoint/outer-scoring for the3heads. They verify alignment,
patient roles and arithmetic, not actual frozen UNI2-h GPU execution or improved
kidney prediction. No remote training or real augmentation pilot has run locally.

Additional safeguards/throughput controls: plan records software package versions;
view generation refuses changes to them. At each sample's first shard, three original
views are recomputed and must reproduce cached identity vectors within rtol1e-3,
atol1e-4. Inspect identity_reproduction.json. This guards preprocessing/encoder
compatibility; it is not a biological quality metric. These few identity checks
are extra validation passes, not a new identity training cache.

Each encoder worker prefetches the next CPU batch in one background thread while
GPU inference runs, using pinned transfers on CUDA. Ordered batching preserves row
alignment/deterministic view parameters. --encoder-batch-size16 is the safe default
(total images, two views/location). Larger even values e.g.32 can be benchmarked
if allocated GPU memory permits; do not infer GPU VRAM from144GB host RAM. Training
batch remains256 across all variants. A faster batch size does not add more views.
