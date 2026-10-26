# Denser training and parallel held-out whole-slide figures

New self-contained experiment; no parent modules, symlinks, raw-image writes or
changes to previous experiments. Local source is copied into RUN/source by plan;
submissions execute that snapshot. Installed packages and existing data/checkpoints
are the only external inputs. Copy the WHOLE folder to the workstation first.

## What changes

- Training grid stride **32 native pixels**, context128, target8. Previously stride128.
  Contexts overlap96 pixels in each direction; targets remain separated by24 pixels
  on this candidate grid. Reservoir cap20,000 tissue-containing locations/patient
  (vs5,000 before), seed42+sample index. Actual rows may be below cap. Shorter candidate
  stride alone does not guarantee a16x denser retained training set because of cap.
- Eligibility: mean RGB >20 and <235, maxRGB-minRGB >15 at a pixel; tissue fraction
  >=0.5 in both context128 and central8. Native context centers68+32*k preserve
  old target lattice alignment. No border padding. Registration/all-valid support
  assumption is retained; no independently validated common-validity mask exists.
  Decisions and counts exported; selection uses H&E only, never mIF expression.
- Per-slide/channel P99 factors are REUSED from the completed normalization plan.
  No new percentile fit from selected tiles. Targets = mean8x8(clip(raw/divisor,0,1)),
  identical recipe/sparse fallback. QPTIFF channel labels are explicitly reordered.
- New coordinates require NEW frozen UNI2-h embeddings; old sparse caches do not
  contain them. FP32, exact original 128->224 PIL bilinear square resize, ToTensor,
  ImageNet mean/std, 1536 CLS features; local checkpoint SHA must match references.
- Weighted MLP default1536->256 GELU/dropout0.1->42, training-only equal-patient target
  zscore and expression weights; optional weighted_fds mode. AdamW.001/weightdecay.0001,
  batch256,100epochs max, minimum51, patience10, seed42+fold. Same five reviewed patient
  folds; train-only fitting, inner validation stopping, outer test evaluation. This
  is a new density experiment, NOT a matched loss-only ablation. More overlapping
  patches do not create more independent patients or guarantee better performance.
- Five independent fold jobs can run concurrently, bounded by GPU concurrency.
  Sample feature-extraction jobs have matching aftercorr preparation dependencies.
- Whole-slide prediction stride8 uses only the sample's OUTER-TEST head; no labels
  supplied to predictor. Tissue criteria identical to training. All in-bounds8x8
  blocks are measured; predictions only where context and tissue support pass.
  Unsupported blocks/borders are NaN with explicit validity mask, shown GRAY.
- Whole-slide figure: H&E | normalized measured mIF8x8 mean | held-out prediction.
  Fixed0..1 viridis scale across markers, no percentile display scaling. Values
  outside range saturate visually only; unconstrained numeric predictions retained.
  Whole-slide images are block-resolution views, not native-pixel or cell predictions.
  Up to7px bottom/right remainder omitted; border context exclusions displayed gray.

## Cost and deadline strategy

Start with a one-patient preparation/extraction pilot, then parallelize rest. At
39x20,000 locations, feature storage <=4.79GB (decimal, excluding targets/files).
Derived native RGB he.npy adds sum(height*width*3) bytes across slides; originals
are untouched. Full H&E is decoded ONCE per patient (budget8GiB image bytes default,
64GB CPU allocation; fail if budget exceeded). GPU band jobs memory-map this derived
file, avoiding repeated decompression of the original H&E. Check scratch disk.

Training stride32 has ~16x candidate count of128, but cap bounds encoder rows.
Whole-slide stride8 candidate count=floor(H/8)*floor(W/8). A20,000x20,000 slide has
6.25million blocks BEFORE tissue rejection. At50 crops/s,1million supported blocks
cost5.56GPU-hours;4 genuinely concurrent GPUs could ideally reduce wall time toward
1.4h plus queue/I/O/initialization. This is an illustration, NOT measured throughput.
Band logs print measured crops/s and remaining seconds; use first bands to estimate
additional slides. Slurm allocations may queue; concurrency is a cap, not a promise.
Keep Monday's priority slide K2140231 first. Do not require all39 whole-slide images.
Increasing markers to42 reuses each embedding/head forward pass; increases QPTIFF
processing/output storage/CPU rendering. Whole-slide embeddings are NOT retained,
only selected-marker band maps; new markers after inference would need rerunning
bands, so choose ALL up front if storage/I/O allow.

Band512 native pixels (64prediction rows), widthwhole slide. Many independent GPU
band jobs. GPU count default2 but configurable; increase to4 only if allocations
available. Each task requests one GPU, never multiple independent CUDA jobs sharing
one exclusive GPU. No mixed precision approximation. Per-sample feature shards can
be reused for head variants; whole-slide image bands do not redo head training.

## Commands — after normalization has completed

Use the ACTUAL completed normalization plan; v4 below is the latest suggested name,
not a claim it has completed. collect must report39/39 first. These commands launch
jobs only on the workstation, not from the coding agent.

```bash
BASE=/scratch/groups/emmalu/hande_annotation
PY="$BASE/envs/kidney-uni2/bin/python"
CODE="$BASE/kidney_uni2/dense_slide_workflow"
NORM="$BASE/uni2_slide_normalization/cohort39_p99_v4"
RUN="$BASE/uni2_dense_training/stride32_cap20000_v1"

"$PY" "$BASE/kidney_uni2/slide_normalization/cohort.py" collect --run "$NORM"

"$PY" "$CODE/workflow.py" plan \
 --normalization-plan "$NORM" --out "$RUN" --stride 32 --cap 20000 \
 --checkpoint "$BASE/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin"

# Freeze all subsequent execution to the recorded source snapshot.
CODE="$RUN/source"
"$PY" "$CODE/submit.py" pilot --run "$RUN"
```

Pilot submits CPU prepare task0 then GPU extract task0 (aftercorr). It is the first
sample in sorted cohort inventory, NOT automatically K2140231. Keep fold/sample
ordering deterministic. Logs RUN/logs/prepare-ARRAYID_0.out/.err and extract-... .

```bash
# AFTER the pilot feature array succeeds; refuses missing/incomplete sample0 cache:
"$PY" "$CODE/submit.py" remaining --run "$RUN" \
 --cpu-concurrency 4 --gpu-concurrency 2
```

This submits prepare/extract arrays1..38; each GPU task releases as its corresponding
CPU preparation completes. Record the printed EXTRACT array ID. Task0 is not rerun.

```bash
# Replace EXTRACT_ARRAY_JOB_ID with the actual printed extraction array ID.
"$PY" "$CODE/submit.py" train --run "$RUN" \
 --after EXTRACT_ARRAY_JOB_ID --gpu-concurrency 2 --mode weighted
```

Queues all5 outer folds after all remaining extraction tasks succeed; each trainer
validates ALL caches, including pilot. Prints training array ID and dependent CPU
aggregate job ID. For weighted_fds, use a SEPARATE plan/run (no output overwrite).
No train command overwrites a previous fold. afterok failures stop downstream;
existing failures need a new run path, not deletion or overwrite.

## Priority slide and all42markers

Can plan before training finishes because metadata/fold assignments already exist:

```bash
SLIDE="$RUN/slides/K2140231_all42_v1"
"$PY" "$CODE/whole_slide.py" plan --run "$RUN" \
 --sample K2140231 --markers ALL --band-pixels 512 --out "$SLIDE"

# TRAIN_ARRAY_JOB_ID is the TRAINING array, not extraction or aggregate ID.
"$PY" "$CODE/submit.py" slide --slide-plan "$SLIDE" \
 --after TRAIN_ARRAY_JOB_ID --gpu-concurrency 4
```

Every band waits for training array success; submit queues automatic CPU assembly
after every band succeeds. Once training is already completed, omit --after.
No whole-slide test is performed before its held-out checkpoint exists. Fold
membership/checkpoint signature enforced at runtime. K2140231 capitalization is
case sensitive and must exist; no guessed substitutes. User excluded F03600 and
K2120228 from additional-slide choices. For another actual sample, repeat plan and
slide submit with a NEW output root/sample. Discover actual IDs with:

```bash
"$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print("\n".join(t["sample_id"] for t in c["tasks"] if t["sample_id"] not in ["F03600","K2120228"]))' "$RUN/experiment.json"
```

## Inspect results

```bash
squeue -u "$USER"
sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
# Replace filenames with actual array IDs/task indices:
tail -n 40 "$RUN/logs/infer-ARRAY_JOB_ID_0.out"
"$PY" "$CODE/workflow.py" collect-cache --run "$RUN"
ls "$RUN/cv/aggregate/primary_summary."*
ls -d "${SLIDE}"_figures_*
```

New gallery: SLIDE_figures_ASSEMBLYJOB/index.html; one PNG/marker. Full stride8
measured.npy, prediction.npy, valid.npy and H&E overview he.npy are disk-backed.
Predictions remain NaN outside support. Final manifest written only after all bands
pass coverage/shape/normalization/checkpoint checks. Numeric grids retain full block
resolution; overview images subsample at most2500cells/side (documented step).
Per-band maps.npz includes measured, prediction, valid, H&E and geometry manifest.
Predictions use all held-out slide tissue, unlike sampled OOF quantitative metrics;
these are qualitative new-support figures, not a directly comparable metric table.

## Safety and recovery

Never run duplicated array tasks or direct commands on the same destinations. Outputs
must be new; no automatic overwriting/resume of partially completed stages. Preserve
failed runs; create a new plan for retries. No raw files changed. Source metadata and
recorded normalization factors are checked; raw-target normalization uses the same
full-slide factors computed earlier. Actual completed normalized cohort prerequisite
catches missing-mif issue via recorded source chain. File hashes validate weights
and factors; this does not establish biological registration or assay calibration.

No extra conda environment needed: use existing kidney-uni2 with NumPy/Pillow,
matplotlib, torch/torchvision/timm and compatible tifffile/zarr/imagecodecs. No automatic
package install. Local synthetic tests do not confirm actual TIFF/CUDA behavior.

## One-command launch after the successful pilot (recommended)

Instead of manually submitting remaining/train/slide above, the pipeline command
plans K2140231 then queues all remaining stages with dependencies. Do NOT run both
methods on the same experiment. It refuses missing pilot output. Planning the
priority slide occurs before any additional jobs are launched.

```bash
"$PY" "$RUN/source/submit.py" pipeline --run "$RUN" \
 --sample K2140231 --markers ALL \
 --cpu-concurrency 4 --gpu-concurrency 2 --inference-concurrency 4 --mode weighted
```

This queues remaining preparation1..38, corresponding extraction, five-fold training,
OOF aggregation, priority whole-slide band inference, and final assembly. All42markers
selected once. Stage/job IDs and submission JSONs are printed/saved. If any prerequisite
fails, dependent jobs do not run. Successful sub-stages are preserved; no automatic
resume/overwrite. Additional slides should wait for actual band throughput estimates.

## Code-server continuation without Slurm

`finish_direct.py` waits for the SUCCESS line in the exact current direct extraction
launcher log, validates all caches, runs folds across cuda:0/cuda:1, aggregates OOF,
then plans/infer/assembles K2140231 across the same two GPUs. No sbatch or submit.py
calls. The active code-server allocation must remain alive through all stages.
Use --launcher-log with the existing direct_launcher_TIMESTAMP.log; do not point it
to individual worker logs or its own output. A failed extraction launcher stops
continuation. Default wait limit36hours. Submit only ONE continuation per experiment.
Existing fold/figure outputs are refused. No automatic partial-run resume.

Copy only the new finish_direct.py into the workstation's dense_slide_workflow
folder; the already-created RUN/source snapshot remains unchanged. This launcher
calls the snapshot workflow/whole_slide scripts with subprocesses. Launch with the
same kidney-uni2 Python, nohup, OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4. Use
--sample K2140231 --markers ALL --mode weighted. Its main log reports ALL COMPLETE
with the gallery path. Detailed logs live RUN/direct_finish_TIMESTAMP/.

Two added mocked-orchestration tests passed: device assignment/dependency order and
extraction-failure blocking. Actual training/rendering still require the allocation.

## Measured-expression selection and additional direct predictions

New helpers screen_expression.py and predict_more_direct.py run from the installed
folder, calling the immutable RUN/source renderer for inference. No retraining,
Slurm or modifications to old outputs. Copy the new helpers to Sherlock.

Screen command:

```bash
"$PY" "$BASE/kidney_uni2/dense_slide_workflow/screen_expression.py" \
 --run "$DENSE_RUN" --out "$BASE/uni2_expression_screen/dense_v1" \
 --markers PODXL VIM CAV CD138 CD8 COLIV POSTN
```

Reads existing exact native whole_slide_histograms.npz for each reference normalized
slide (all pixels including background) plus retained prepared measured/raw_target
arrays (sampled eligible tissue estimate). Excludes F03600, K2120228 and already
rendered K2140231 by default. Writes expression_by_slide_marker.csv and REVIEW.txt.
Each marker ranked separately by sampled tissue normalized mean; raw means,
nonzero fractions, P99 fallback and training-threshold high fractions also reported.
P99 normalized brightness does not establish high absolute expression; raw means
can reflect technical batch variation. Fold-specific high thresholds are descriptive,
not biological positivity or a single common threshold across slides. Histograms
are exact all-pixel summaries, not tissue masks; sampled means are NOT whole-slide
exhaustive tissue expression. Do not aggregate raw marker intensities into a single
cross-marker score. Select cases for qualitative review only; do not claim an
unbiased performance estimate from high-expression selected slides.

After choosing actual sample IDs from REVIEW.txt:

```bash
read -r -a CHOSEN_SLIDES -p 'Slide IDs from ranking, separated by spaces: '
MORE_LOG="$DENSE_RUN/more_slides_$(date +%Y%m%d_%H%M%S).log"
nohup "$PY" -u "$BASE/kidney_uni2/dense_slide_workflow/predict_more_direct.py" \
 --run "$DENSE_RUN" --samples "${CHOSEN_SLIDES[@]}" --markers ALL --tag high_expression_v1 \
 > "$MORE_LOG" 2>&1 &
tail -f "$MORE_LOG"
```

Slides process sequentially, each with two band workers cuda:0/cuda:1. Wait until
prior two-GPU workloads finish. Existing slide/tag outputs refused. Every slide
uses its own outer-test checkpoint; all markers reuse the same encoder forward.
Gallery RUN/slides/SAMPLE_high_expression_v1_figures/index.html. Exact histogram
arithmetic and mocked no-retraining/two-GPU orchestration tests passed (2 tests);
actual new slides have not run locally.

## Per-marker cohort plots / optional patient heatmaps

New plot_marker_metrics.py reads primary_summary.csv and writes correlation and
normalized MAE/bias PNG+vectorPDF figures, all42markers sorted by meanPearson. Values
and95% intervals are taken from the saved equal-patient OOF summary, not recomputed
from whole-slide image maps. Optional --patient-csv primary_patient_metrics.csv
produces real marker-by-patient Pearson/Spearman heatmaps with fixed[-1,1] color
limits and undefined values gray. It checks per-patient means/defined counts match
the supplied cohort summary. Different-experiment summary CSVs must not be pooled.

```bash
"$PY" "$BASE/kidney_uni2/dense_slide_workflow/plot_marker_metrics.py" \
 --summary "$DENSE_RUN/cv/aggregate/primary_summary.csv" \
 --patient-csv "$DENSE_RUN/cv/aggregate/primary_patient_metrics.csv" \
 --out "$BASE/uni2_metric_figures/dense39_v1"
```

No GPU/model inference/raw image reads. Needs NumPy/matplotlib in the existing env.
Output must be new. All42markers39patients from primary_summary-3.csv were plotted
locally in reports/cohort39_marker_metrics_v1 and PNGs visually checked. Actual local
patient-level CSV unavailable, so no real heatmap was generated locally. Aggregate
summary cannot reconstruct individual slide cells. Confidence intervals condition
on fixed predictions and omit full shared-fold/training/model-selection uncertainty.

Heatmap layout fix: colorbar has an explicit separate axis outside both panels.
To replace only existing heatmap PNG/PDF for the same inputs, add --heatmap-only
--overwrite-heatmap to the plot command. Existing output manifest must reference
those exact source files. Other plots, gallery, metrics and manifest stay unchanged.
A synthetic42x39 layout was visually checked; checksum checks confirmed no other
output files were modified. No real patient heatmap was recomputed locally.
