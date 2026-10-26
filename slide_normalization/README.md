# Per-slide, per-marker whole-slide percentile normalization

This folder is an independent pipeline. No scripts from parent/earlier experiment
folders are imported at runtime. No earlier code/output is changed. It reads raw
registered QPTIFFs and old caches read-only, writes new normalized-target feature
caches, and trains fresh heads using those caches and the reviewed patient folds.
It never writes normalized TIFFs over originals or retrains UNI2-h.

## Exact target definition

For EACH slide and EACH marker independently:

1. Scan ALL native-resolution pixels in the selected QPTIFF series/level, including
   zero/background pixels. This is not P99 across patients, selected tiles, a
   downsampled thumbnail or a tissue mask. Default series0/level0, as before.
2. Compute P99 exactly using an unsigned-integer histogram (uint8 or uint16).
   The percentile uses NumPy's default linear order-statistic interpolation.
3. Divide native pixel values by that slide/channel's effective divisor and
   clip to [0,1]. No background subtraction, min subtraction or histogram matching.
4. Average those normalized/clipped pixels within each ORIGINAL selected central
   8x8 target block. Keep the same xy, tile selection and H&E embeddings.

Formula: new_target = mean_8x8(clip(native_pixel / slide_channel_P99,0,1)).
This differs from clipping an already averaged target. Therefore existing cached
means alone are insufficient and the raw QPTIFF must be reread. Features need not
be recomputed, since H&E inputs and coordinates are unchanged.

### Zero/sparse channel policy

If P99 is zero, use the unsigned image dtype maximum (255 or65535) as divisor,
flag the channel, and retain its targets. This prevents division by zero or
arbitrary amplification of an almost-all-zero channel while preserving sparse
positive pixels. It is an explicit fallback, not P99 normalization for that channel.
A zero-P99 marker is NOT automatically biologically absent or excluded from loss.
Nonzero very small percentiles are also flagged (<=1 native intensity unit).

Each channel's report includes exact P99, effective divisor, zero fraction,
clipped fraction, observed range, dtype-max fraction and fallback flag. Whole-slide
background coverage affects P99. Very sparse channels may fall back often; channels
with small nonzero P99 may have noise amplified. Review reports before interpreting
changes. This is a precisely specified experiment, not proof of benefit or an
exact reconstruction of undocumented HEX preprocessing details.

The normalizer currently requires uint8/uint16 and matching named panels, geometry,
MPP and raw 8x8 means. It rejects unknown axes/channel ordering rather than guessing
unlabeled markers. Input TIFF storage implying >1GiB decode per region is rejected
for reviewed handling; originals must never be retiled/rewritten as a workaround.
For unusually chunked files, design a bounded reader or separate derived tiled copy.

## What is copied and what is stored?

New cache root is RUN/cache/SAMPLE. Each NPZ contains:
- features: copied verbatim numeric array from old cache, equality checked.
- xy: same selected integer centers/order, equality checked.
- target: new pixel-normalized-then-averaged 8x8 relative expression values.
- raw_target: original cached mean for audit, never a training target here.

No new cells/regions are selected, no tissue is refiltered, and no full normalized
image is persisted. Histogram scanning covers the whole slide, even though training
retains the original capped sampled centers. Disk use is chiefly another copy of
the existing small feature cache, plus target/histogram/audit files; no symlink
or code dependency on another experiment folder is introduced.

Per slide: normalization.json, whole_slide_histograms.npz (panel order),
source_cache.json (input manifest/shard SHA256 and feature/xy checks),
raw_input_check.json (before/after source metadata), invocation/source snapshot,
and a complete manifest written ONLY after checks pass. Source cached raw means
must agree with freshly read raw target blocks before normalized targets are
accepted. Source files are not opened writable. Existing output paths are refused.
Derived output paths are checked against raw/cached input directories.

Cache manifests carry target_units=slide_channel_percentile_pixel_clipped_0_1 and
a shared normalization_recipe. Individual factors reside in per-slide reports,
not the shared recipe, because factors intentionally differ across slides.
Training partitions/checkpoint provenance include recipe/units, rejecting mixes
of raw, differently normalized or differently configured target definitions.

## Train on the normalized targets

Uses the latest one-hidden-layer expression-aware head:
1536 ->256 ->GELU ->dropout.1 ->markers, frozen UNI2-h embeddings.
AdamW lr.001, weight decay.0001, patient-balanced sampling, batch256, seed42+fold,
maximum100 epochs, minimum51, patience10. Regression still uses TRAINING-only
patient-balanced z scores, now computed from the ALREADY slide-normalized block
values. Inverse scaling returns relative normalized expression, never raw0..255.
Expression weighting uses training90th-percentile relative block intensity and
cap5 with the same prevalence safeguards as the previous weighted pipeline.
No ALF is added. The optional weighted_fds mode uses normalized block target/1
for its intensity bins; dividing by255 again is forbidden.

Default training mode weighted isolates normalization without adding FDS.
Optional weighted_fds uses the same10/50 zero-based FDS on/off schedule; both modes
share the same minimum selection epoch and ordinary unweighted validation Huber.
Checkpoint format kidney_slide_p99_v1 is distinct from old heads. Use local
model.load_checkpoint or predict.py. No old checkpoint is altered/reused for fitting.
Scalers, expression thresholds/weights and FDS statistics are still fitted only
on inner TRAINING patients. Outer testing never selects the checkpoint.

## What held-out per-slide normalization means

A held-out slide's measured CODEX/QPTIFF is used to DEFINE its reference relative
expression labels, under this fixed protocol. Those factor values are not passed
to the H&E head and no predictor is fitted on outer labels. This evaluates relative
expression, not the ability to infer a new slide's raw acquisition brightness.
In H&E-only deployment, that new slide's measured-channel P99 is unavailable.
Predictions are relative; converting back to raw signal would require a measured
factor and clipping is irreversible in any case. Do not claim raw-scale recovery.

Changing label normalization can change biological interpretation and may erase
real between-slide intensity differences along with technical variation. Raw MAE
and normalized MAE cannot be compared numerically as proof of improvement. Clipping
can alter ranks/correlations and the definition of high expression; preserve the
old baseline as a separate experiment. The earlier harmonized evaluator correctly
rejects unequal ground-truth arrays; do not disable that check to mix these runs.

## Workstation setup

Copy this entire folder to:
/scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization
Reuse the existing kidney-uni2 environment. No package changes or model download
needed. Normalization needs NumPy, tifffile, zarr and imagecodecs as already used;
training needs PyTorch; tests use SciPy. Training tests do not exercise real TIFF
libraries locally if they are absent: see VALIDATION.md.

### 1. Make a NEW plan

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs

/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
 /scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization/cohort.py plan \
 --cache-root /scratch/groups/emmalu/hande_annotation/uni2_cohort_cache/cohort39_v1 \
 --reference-folds /scratch/groups/emmalu/hande_annotation/uni2_cv_runs/cohort39_mlp_job_47815087/folds.json \
 --out /scratch/groups/emmalu/hande_annotation/uni2_slide_normalization/cohort39_p99_v1
```

Existing plan/output directory is refused. Use v2/a new name for a genuinely new
preprocessing experiment. Plan records input sample/patient rows and copies folds.
No arbitrary cohort truncation, exclusion or patient remapping occurs.

### 2. Submit CPU normalization, one task per slide

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
 /scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization/submit_normalization.py \
 --run /scratch/groups/emmalu/hande_annotation/uni2_slide_normalization/cohort39_p99_v1 \
 --max-concurrent 4
```

The helper sizes the Slurm array from tasks.json. Each task requests4CPUs,16GB and
8hours on emmalu; no GPU. Start with concurrency4 to limit shared-storage traffic.
This scans whole slides, so runtime is I/O/decompression dependent. Progress logs
report tile/channel counts, histogram scan and grouped target reads. Tests used
small synthetic arrays and cannot establish real whole-slide runtime.

Logs: uni2_slurm_logs/slide-p99-ARRAYID_TASKID.out/.err.
Tasks run in isolated processes; one crash does not abort other samples. A failed
sample has no complete manifest. Existing successful outputs are never overwritten;
retry only failed task IDs with sbatch --array=ID1,ID2 normalize_array.sbatch RUN,
after moving its incomplete derived directory aside or using a new plan. No deletion
of raw/old progress is required. This pipeline advertises no in-place resume.

### 3. Collect completion and inspect normalization QA

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
 /scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization/cohort.py collect \
 --run /scratch/groups/emmalu/hande_annotation/uni2_slide_normalization/cohort39_p99_v1
```

Returns nonzero if any expected slide is incomplete. Writes collection_report.json
and normalization_summary.csv at the plan root. Inspect effective factors, fallback
channels and clipping/zero fractions. Each slide also has normalization.json under
RUN/cache/SAMPLE. Collect must pass before any training launcher proceeds.

### 4. Train new normalized-target models

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization/run_cv.sbatch \
 /scratch/groups/emmalu/hande_annotation/uni2_slide_normalization/cohort39_p99_v1 weighted

# Optional second matched run after QA:
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slide_normalization/run_cv.sbatch \
 /scratch/groups/emmalu/hande_annotation/uni2_slide_normalization/cohort39_p99_v1 weighted_fds
```

Fresh outputs: uni2_slide_p99_runs/MODE_job_JOBID. One GPU/32GB/fourhours, emmalu.
Source snapshots, environment, tests, preflight, fold histories, checkpoints and
outer prediction shards follow the previous standalone experiment structure.
Every output/prediction manifest and primary summary identifies normalized units.
No lower0..1 error should be compared directly with old0..255 error. Predictions
remain unconstrained; negative/>1 predictions are reported rather than hidden.

### Checks and boundaries

Run doctor.py --device cpu, full unittest discovery, compileall and bash -n.
No cohort normalization, CUDA training or actual remote image verification has
been performed by the coding agent. No dense normalized-map renderer is included
in this first implementation; old dense scripts expect raw targets and must NOT
be used unchanged on these checkpoints. A future renderer must normalize native
measured pixels using the saved slide report and keep relative units consistent.

### Source-path fix after array 49029177 (2026-10-23)

The original extractor saves `he` and `source` (prepared-data directory), but omits
`mif` from feature manifests. The initial normalization assumed `mif` existed and
failed before normalization. `source_paths.py` now follows recorded source manifest
links, validates sample/patient/panel/shape/MPP/context/block identity, checks raw
file existence and rejects cycles or conflicting paths. No filename guessing or
raw-directory discovery. Cohort planning checks these paths before creating output;
normalization records and rechecks provenance hashes. Derived manifests include
resolved raw paths. Missing prepared provenance must be restored or reviewed, not
silently replaced. Old inputs and failed output directories remain unchanged.

Use a fresh plan root (e.g. cohort39_p99_v2), then submit --array=0 for a one-sample
workstation check before scheduling remaining tasks. Existing failed afterok
training dependencies do not automatically move to a new array.
