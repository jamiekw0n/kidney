# Standalone expression-aware frozen UNI2-h experiment

This folder is self-contained: it imports no code from the parent implementation.
Existing scripts, raw H&E/mIF, cached features and measured targets remain unchanged.
Only NumPy and PyTorch are required, using the existing kidney-uni2 environment.
No weights download, image decoding or encoder execution is needed.

## Scientific question and fixed first experiment

Does emphasizing high-expression targets improve sparse-marker recovery without
excessive false expression? This is exploratory, not an established better model.
Reuse the reviewed five outer patient folds and their inner validation partitions.
Reuse the same 1536 -> 256 -> GELU -> dropout 0.1 -> 42 MLP, AdamW lr .001,
weight decay .0001, batch 256, patient-balanced replacement sampling, seed 42+fold,
100 maximum epochs, patience 10 and training-only patient-balanced target z scores.
Encoder remains frozen; target remains each marker's central 8x8 mean.

Default strata are HIGH versus LOW EXPRESSION, not biological positive/negative.
For each marker and training fold, find the 90th percentile of raw targets using
weights 1/(number of training patients * rows belonging to that patient).
High means strictly greater than this threshold; ties belong to low. Thus high
prevalence need not be exactly 10%, particularly with zero-valued targets.
Thresholds are NEVER estimated from inner validation or outer test patients.
An optional JSON object mapping EVERY panel marker to a numeric raw-intensity
threshold replaces these quantiles. Such thresholds must be assay-defined or
established independently of held-out outcomes. Cell positivity labels do not
automatically define an 8x8 intensity threshold.

Let p be the training high-expression prevalence under equal-patient sampling.
Set w_high=min(0.5/p,5), w_low=min(0.5/(1-p),5). Normalize expected marker weight
p*w_high+(1-p)*w_low to 1, then rescale both together if needed to maintain an
actual maximum weight of 5. Degenerate all-high/all-low markers use unit weights.
The cap can prevent exact equal group contributions and can make an individual
marker's expected weight below 1; expression_state.json records the actual values.
These fixed TRAINING prevalence weights multiply unreduced standardized Huber
(delta=1) errors before averaging over batch rows and markers. This avoids unstable
minibatch-specific denominators when there are no positives in a batch. In the
uncapped case its expectation equals 0.5*mean(high loss)+0.5*mean(low loss) under
the patient-balanced distribution. This is loss weighting, not positive-only filtering.
Low-expression examples remain, penalizing false expression. Weighting is per
marker: a row can be high CD138 and low CD8.

Early stopping stays on ordinary unweighted inner-validation Huber, averaged
within patient then across patients. Train objective and validation Huber have
DIFFERENT weighting; their magnitudes are not a train/validation gap comparison.
Do not adjust settings based on outer results. A --unweighted control retains
all other settings; run it separately for a matched baseline if desired.

## Launch on the workstation

Copy this ENTIRE folder to:
/scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware
Then:

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware/run_cv.sbatch
```

Defaults: existing cache root uni2_cohort_cache/cohort39_v1; existing reviewed
folds uni2_cv_runs/cohort39_mlp_job_47815087/folds.json; emmalu partition, one GPU,
32 GB RAM, four-hour limit. The parent CV process scores on CPU; only isolated
fold trainers acquire CUDA, avoiding the earlier exclusive-GPU conflict.
All outer patients must appear exactly once; cache metadata and partitions are
validated. Numeric memory peak is guarded at 8 GB by default (framework overhead
excluded). Output directories must be new; no overwrite/resume. Re-running sbatch
creates a new job-specific run. This does not change the old MLP run.

Matched unweighted control:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware/run_cv.sbatch --unweighted
```

Assay thresholds, if available:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware/run_cv.sbatch --thresholds /absolute/path/thresholds.json
```

Slurm launcher snapshots executable source, job settings and pip freeze. The
synthetic CPU end-to-end test runs before cohort training. CUDA availability is
checked in a short-lived separate process. Every real trainer snapshots its
source, arguments, splits, seeds, scaling, expression counts, weights and epochs.

## Outputs and interpretation

uni2_expression_runs/job_JOBID contains:
- inventory.json and folds.json: exact input samples, row counts and partitions.
- fold_XX/training.log: progress for each fold.
- fold_XX/training/expression_state.json: thresholds, weights, prevalence, positive
  row and patient counts, degenerate markers, effective expected weight.
- fold_XX/training/history.json: weighted training objective and unweighted
  inner-validation Huber. best.pt: head, scaling, expression state, provenance.
- fold_XX/training/validation_expression_metrics.json: inner-validation strata.
- fold_XX/outer_predictions/SAMPLE: xy, unchanged measured target, prediction in
  original intensity units; manifests point to the checkpoint.
- primary_summary.json/.csv: existing equal-patient overall regression metrics
  and patient-bootstrap intervals, directly comparable in units to previous runs.
- oof_expression_patients.json: per-patient, per-marker high/low MAE, precision,
  recall, false-positive rate, average precision (AP), and predicted/measured
  positive fractions; raw predictions are thresholded at the training threshold.
- oof_expression_summary.json: equal-patient means and defined-patient counts.

AP groups tied prediction scores and is null for patients with only one class.
Precision is null when no predictions are called positive; recall is null when
there are no measured positives. Other empty strata are null, never silently zero.
Fold-specific quantile thresholds differ, so these are fold-relative expression
metrics, not universal assay-positive metrics. Preserve natural held-out prevalence.
Report improvements in high-expression MAE alongside low-expression MAE and false
positives. AP is meaningful relative to measured positive prevalence. Existing
Pearson/Spearman/bias/slope remain important. This folder does not add FDS, ALF,
new feature extraction, probability calibration or a presence/intensity head.

## Checks and execution status

Run `python -m unittest discover -s /absolute/path/expression_aware -v` and
`python -m compileall -q /absolute/path/expression_aware`.
Numerical tests cover patient-weighted thresholds, ties/constant markers, capped
weights, assay thresholds and subgroup metrics. A synthetic four-patient CPU CV
integration test runs automatically when PyTorch is installed. Synthetic success
is not evidence of biological performance. Workstation cohort execution is required;
this local machine has no PyTorch or remote cohort caches. No remote job has been
submitted by the coding agent.

## Side-by-side spatial comparison with the original MLP

`compare_predictions.py` and `plot_comparison.sbatch` reproduce the earlier
uni2_oof_comparisons layout, correctly labeled for the two MLPs. Copy these new
files to this folder on the workstation. Submit with the completed run directory:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware/plot_comparison.sbatch \
 /scratch/groups/emmalu/hande_annotation/uni2_expression_runs/job_TRAINING_JOB_ID
```

No GPU, model loading, extraction or retraining is needed. Reads saved outer-test
prediction shards from original MLP job 47815087 and the specified weighted run.
Rejects different patient folds, inventories, coordinates, targets or encoder/
physical preprocessing provenance. Outputs are new, never overwritten, at
`uni2_oof_comparisons/expression_job_PLOTTING_JOB_ID/index.html`.
Copy the entire directory to view the gallery locally.

For each sample/marker, the top row shows all sampled locations and the bottom
row shows only coordinates whose measured target exceeds the chosen threshold.
Columns: measured mIF, original MLP, expression-weighted MLP. Each figure uses
one shared color scale across all six panels, includes prediction extremes,
and does not clip predictions. Values are divided by 255 (verified uint8 source);
this changes displayed units, not model accuracy. Squares are sampled centers,
not filled 8x8 or 128x128 footprints; gaps are missing coverage, not zero signal.
The default plot threshold 0 matches the earlier nonzero-target comparison,
NOT the fold's training-quantile weighting threshold or biological positivity.
Bottom panels conceal false positives outside their selection: inspect top rows.
Figures show only central-block means, not native-resolution mIF or H&E.

Default markers: VIM CD10 PCNA CAV CD34 CD31 PODXL CD8 CD138 COLIV POSTN.
Append `--markers CAV CD8 CD138 COLIV POSTN` to change markers, or
`--samples F03600` to plot only that sample (numeric reports still cover all).
Append `--positive-threshold NUMBER` to change the exploratory raw-intensity
selection. POSTN is the panel name for periostin.
`original_mlp/primary_summary.*` and `weighted_mlp/primary_summary.*` contain
matched overall regression metrics in units divided by 255. Subgroup metrics
using training-derived thresholds remain in the training run's expression reports.

## Same-ROI dense original versus weighted MLP comparison

Copy the entire updated expression_aware folder; it now includes replicated
prepare.py, image_io.py and raw_safety.py to keep this experiment self-contained.
No parent files are edited. Submit:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_aware/dense_comparison.sbatch \
 /scratch/groups/emmalu/hande_annotation/uni2_expression_runs/job_TRAINING_JOB_ID
```

`dense_from_review.py` follows the source_dense_root metadata in the existing
uni2_dense_rois/f03600_cav_review_v1 review to the original completed inference.
It uses that inference's exact native ROI, input paths, intensity divisor and
original MLP run; no hard-coded reconstruction of ROI coordinates is used.
Original and weighted folds must match. Each head must exclude F03600's patient
from both training and inner validation. Encoder checkpoint SHA and head cache
signature must match. All raw images are opened read-only, with before/after
metadata checks. Writes go only to a new derived run and source snapshot.

Previous dense runs stored predictions but not dense embeddings. Therefore this
launcher requests a GPU and reruns frozen UNI2-h once per location, feeding the
same feature vector to both heads. The sparse training cache cannot supply this
dense stride-8 grid. No head retraining occurs. At the earlier 2400x4000 ROI this
is 150,000 locations. Dependencies are those already used for dense inference
(PyTorch, timm, torchvision, Pillow, tifffile, zarr, imagecodecs, matplotlib, NumPy).

Output: uni2_dense_rois/f03600_expression_job_PLOTTING_JOB_ID/index.html.
Each marker has five panels: registered H&E; native measured mIF; measured 8x8
means; original held-out MLP; weighted held-out MLP. CAV, CD8, CD138, COLIV and
POSTN are default markers (override with --markers). Display scale is shared
across the four expression panels, based on the 1st/99th percentiles of pooled
block-measured/original/weighted arrays, matching the earlier visualization
method. This may clip bright measured pixels and predictions; numeric values
are not clipped. Do not compare brightness across independently scaled figures.
All ROI locations are included, without tissue filtering, matching prior dense
runs. Each predicted map pixel represents an 8x8 mean; context remains 128x128.
Predictions are NOT native-pixel or cell-level expression.

dense_predictions.npz stores xy, target, original_mlp and weighted_mlp for the
entire panel; measured_mif_native_roi.npz stores requested native channels.
roi_metrics_normalized.json describes this single selected ROI, not cohort
performance. Exact ROI, head checkpoints and display settings are recorded.
No dense feature arrays are retained. Add further markers on a fresh run, or
build a compatible CPU renderer for the saved new-format predictions; the old
parent replot_dense_roi.py expects linear/mlp keys and must not be used unchanged.
Local tests verify reference resolution, exact 150,000-location grid and held-out
fold guards. Real dense inference remains workstation-only verification.
