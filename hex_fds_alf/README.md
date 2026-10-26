# Frozen UNI2-h with HEX-inspired FDS and adaptive loss

This directory is a **self-contained experiment**. Copy it alone to the workstation.
It imports no scripts from the parent kidney_uni2 implementation, expression_aware,
HEX, ROSIE, or any other source checkout. It does use installed NumPy, PyTorch,
SciPy (verification), and optional Matplotlib, as any Python program uses packages.
No runtime download occurs. Head training uses only caches; optional dense review
loads the frozen encoder and reads registered images without modifying them.

This is an adaptation of HEX's *head / FDS / ALF mechanics* to existing frozen
UNI2-h caches, **not a reproduction of the full HEX model or its reported results**.
Read [HEX_CROSSCHECK.md](HEX_CROSSCHECK.md) for the source-by-source comparison and
[VALIDATION.md](VALIDATION.md) for what has actually been verified.
Original and expression-weighted implementations are unchanged.

## What changes and why

The existing dataset is registered H&E context -> frozen UNI2-h CLS embedding
(1,536 values) -> measured marker means in the central 8x8 block. The cohort cache
retains the previously selected, tissue-filtered locations and 42-marker panel.
No new tiles, tissue exclusions, masks, or marker positivity calls are introduced.

This experiment implements:

1. HEX-style head: 1536 -> 256 -> ReLU -> dropout 0.5 -> 128 -> ReLU -> dropout 0.5
   -> number of panel markers (42 for the current cohort). Outputs are unconstrained.
2. Marker-specific feature distribution smoothing (FDS) on the shared 128-D hidden
   representation, before each marker's output row, only during training.
3. Adaptive robust negative log-likelihood (ALF), with independent learned shape
   and scale for each marker, optimized jointly with the head.
4. Patient-level outer CV using the exact existing fold JSON, with inner-patient
   validation, fixed checkpoint-selection metric, and original-unit outer predictions.
5. Four matched ablation modes: control (Huber), fds (FDS + Huber), alf (ALF),
   fds_alf (FDS + ALF). All four use identical architecture and target scaling.

There is **no expression-quantile loss weighting** in any mode. The copied
expression.py module computes evaluation strata and reports only. Its returned
weight fields are not used to form the training objective, select checkpoints,
or compute FDS statistics. FDS bins cover the entire fixed intensity range;
they are not the previous experiment's upper-decile categories.

## Training data and scaling

Inputs: one complete manifest.json per sample, listing NPZ shards with features
[N,1536], target [N,M], and integer xy [N,2]. Targets are unchanged measured 8x8
means. Only complete, nondiagnostic UNI2-h caches are accepted. Panel, encoder
revision/SHA, transform, input size, MPP, target block and selection policy must
match across caches. Every sample belonging to a patient remains in one split.
The current sample-as-patient mapping is inherited; independence of that mapping
is a biological assumption, not something this code can verify.

Targets are divided by a **fixed declared intensity ceiling, 255 by default**,
which matches the verified uint8 QPTIFF cohort. Every train/validation target must
lie in [0,255]; values outside are rejected, never silently clipped. If another
verified measurement scale is used, explicitly set --intensity-max. This is not
per-slide min/max scaling, not percentile normalization, and not z scoring.
No held-out statistics are fitted. Predictions are multiplied by the same ceiling
before export. Predictions can be negative or above 255; they are not clipped.
This scaling differs from the old MLP's patient-balanced z scoring and changes
the optimization objective, so old/new head comparisons are not isolated FDS tests.

## FDS in detail

For every marker, divide normalized target intensities into 50 equal-width bins.
Bucket index is min(floor(target * 50),49), so an intensity of exactly 1 is in the
last bin. Each target selects its *own marker's* bin; a row can be high CAV and
low CD8. The shared hidden feature is calibrated separately for each marker,
then passed through that marker's output weight row. There is no cross-marker
mixing of moments and no averaging of all marker targets into one label.

During each training epoch, accumulate detached, **unsmoothed** hidden features
from training minibatches (including dropout, as in HEX). Counts, sums and squared
sums are accumulated in float64. Per-bin sample variance uses the n-1 denominator.
Running means/variances use EMA momentum 0.9; a bin's first supported observation
is initialized directly rather than shrinking toward a zero prior.

At epoch end, smooth feature mean and variance across neighboring intensity bins.
The Gaussian kernel has size 9 and sigma 2. Its coefficients match HEX's SciPy
Gaussian-filtered delta construction; convolution uses torch reflect padding.
Only bins with observed support contribute, with kernel renormalization. A bin
needs at least two observations **within an epoch** to contribute a variance
estimate. Unsupported bins and zero-variance feature dimensions bypass calibration.
These explicit sparse-bin safeguards differ from HEX's placeholder filling.

For each feature dimension h in an observed bin b:

```
h_adjusted = (h - mean_b) * sqrt(clip(smoothed_var_b / var_b, 0.1, 10))
             + smoothed_mean_b
```

No in-place changes are made to the shared hidden representation. Feature
statistics are buffers, not optimized parameters. They are checkpointed but do
not receive gradients. Gradients still flow through calibrated features into
hidden layers and the output row. The encoder remains absent/frozen.

Schedule uses **zero-based epochs**, matching the released HEX script:
- 0 through 9: collect/update moments; no smoothing.
- 10 through 49: apply calibration using completed previous-epoch moments and
  update moments from current training minibatches.
- 50 onward: no FDS calibration and no moment updates.

Validation, outer testing and deployment always bypass FDS and do not update its
buffers. `model.eval(); model(features)` needs no measured target. Supplying labels
in evaluation does not affect outputs. Target-dependent calibration must NEVER be
performed on validation or test images.

Patient-balanced replacement sampling is retained from the kidney pipeline by
default: per-row sampling weight is 1/number_of_rows_for_that_patient. FDS moments
therefore describe training *draws*, not independent patients or unique cells.
training_bin_support.json separately records unique cached rows and distinct
patients per bin. A bin with many rows from one patient is still weak biological
support. --sampling uniform provides shuffled uniform-row training as an explicit
ablation; it does not reproduce HEX's distributed multi-GPU sampler exactly.

## Adaptive loss in detail

For residual r = predicted_normalized - measured_normalized and per-marker
learned shape alpha and scale c, the objective is:

```
beta = 2 - alpha
rho(r; alpha,c) = beta/alpha * expm1((alpha/2) * log1p((r/c)^2 / beta))
NLL = rho(r; alpha,c) + log(c) + log Z(alpha)
Z(alpha) = integral over real u of exp(-rho(u; alpha,1)) du
```

All three NLL terms participate in backpropagation. Shape is an affine sigmoid
of a free parameter, bounded strictly between 0.001 and 1.999; initial alpha=1.
Scale uses the reference package's shifted affine-softplus parameterization, with floor
1e-5 and initial scale=1. With 42 markers, ALF adds 84 optimized parameters.
These are marker-level residual parameters, not location-level uncertainty scores.
Both parameter groups use Adam, with no weight decay. Checkpoints include loss,
optimizer and scheduler states; loading only the head is sufficient for inference.

**Self-contained normalization:** instead of importing robust_loss_pytorch and
its binary partition-spline resource, adaptive_loss.py evaluates the same
normalization integral with 512-point Gauss-Legendre quadrature using
u=tan(pi*t/2), t in (0,1), exploiting symmetry and log-sum-exp. Calculations and
moment accumulation use float64; head weights remain float32. Quadrature is
differentiable with respect to alpha. No stop-gradient normalizer, learned
unnormalized penalty, MSE approximation, or missing log-scale term is used.
This is a numerical implementation difference from HEX's reference dependency;
see tests for independent adaptive integration, analytic limits and gradients.
CPU and CUDA support float64; MPS is not a supported execution target.

ALF is robust residual modeling, not explicit positive weighting. It may reduce
influence of true rare positives if their residuals appear outlying. FDS/ALF cannot
recover spatial information that a pooled context embedding does not contain.
Neither method guarantees reduced overprediction or calibrated intensities.

## Defaults, model selection and controls

- Adam lr=1e-3, no weight decay; ExponentialLR gamma=0.95 per epoch.
- Batch 256, patient-balanced sampling, seed 42+outer fold index.
- Maximum 120 epochs; checkpoint eligibility starts at epoch 51 (one-based).
- Patience 10, counted only after eligibility begins.
- Selection: unweighted mean squared normalized error, averaged within each
  validation patient then across patients/markers. Loss parameters are not fitted
  on validation data, and adaptive NLL is NOT the selection metric.

Starting selection after FDS turns off ensures the chosen checkpoint comes from
training under its eventual inference pathway. Earlier validation values are
still logged. This is a deliberate adaptation: the released HEX script saves
every five epochs and does not implement this early-stopping protocol. All four
modes use the same 51-epoch minimum to avoid changing training budgets between
controls. If changing FDS off, min-epochs must be greater than fds-off. The CLI
rejects an inconsistent schedule. Synthetic tests use a tiny, explicitly shorter
schedule; they are not the real-data defaults.

The cached-head lr=1e-3 is inherited from our kidney experiments, not HEX's
end-to-end lr=1e-5. To test the released optimizer settings more closely, pass
--lr 1e-5 --lr-gamma .95, and record that as a separate prespecified experiment.
ReLU, two hidden layers, dropout, normalization, optimizer and stopping differ
from the earlier GELU/dropout0.1 MLP. Use the **matched control in this folder** to
attribute changes to FDS/ALF, rather than crediting every difference to those losses.

## Workstation commands

Copy only this folder, including documentation and tests, to:

```
/scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf
```

Use the existing kidney-uni2 environment. Verify rather than silently installing
or changing PyTorch:

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
 /scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf/doctor.py --device cpu

/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
 -m unittest discover -s /scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf -v
```

Submit the combined experiment (no launch has occurred from the coding agent):

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf/run_cv.sbatch fds_alf
```

Use `control`, `fds`, or `alf` instead of `fds_alf` for matched ablations. These
are separate jobs and new outputs, with identical patient folds. Decide the
comparison protocol before examining outer results. Because earlier outer
results have already guided development, this remains exploratory; strong final
claims need an untouched patient cohort or independent external validation.

The launcher requests one GPU, four CPUs, 32 GB and four hours on emmalu. It copies
source into the job record, saves pip freeze and Slurm settings, runs all CPU tests
(including tiny real training), checks CUDA in a separate short-lived process,
then runs CV. The CV parent only scores on CPU; each isolated trainer owns CUDA,
avoiding the earlier exclusive-GPU ownership conflict. There is no background
work continuing after a command failure. Failed jobs retain diagnostics; reruns
require new directories and retrain all folds (no resume is advertised).

Defaults read caches at uni2_cohort_cache/cohort39_v1 and reviewed folds at
uni2_cv_runs/cohort39_mlp_job_47815087/folds.json. They are data artifacts; no code
from that run is executed. A changed sample-to-patient mapping or inconsistent
fold inventory is rejected. No arbitrary patient count, subsampling or truncation
is silently substituted. For custom paths, run cross_validate.py directly:

```bash
python /absolute/path/hex_fds_alf/cross_validate.py \
 --cache-root /absolute/cache_root --reference-folds /absolute/reviewed/folds.json \
 --out /absolute/NEW_output --mode fds_alf --device cuda
```

Add --plan-only for validation/inventory without training. The output still must
be new. A conservative numeric-array memory estimate is limited to 8 GB by default
(--max-numeric-gb). It excludes framework/OS overhead; compare with Slurm allocation.
It never silently reduces the cohort. Existing cache data and raw images are never
rewritten, normalized, retiled, deleted or chmodded.

## Outputs and monitoring

Run root: uni2_hex_runs/MODE_job_JOBID. Logs:
uni2_slurm_logs/hex-fds-alf-JOBID.out and .err, plus the record directory printed
in stdout. Each trainer streams epoch summaries to fold_XX/training.log.

- inventory.json, folds.json, config.json, preflight.json: exact inputs, row counts,
  reviewed partitions, physical/encoder metadata via manifests, settings/resources.
- invocation.json and source_snapshot/: arguments, local source SHA256 hashes and
  immutable copied code; job record additionally saves environment and Slurm info.
- events.jsonl: timestamps and fold status; final manifest.json complete only on success.
- fold_XX/training/config.json, target_scaler.json: architecture and fixed scaling.
- training_bin_support.json: distinct rows and patient support per intensity bin.
- history.json: training NLL/Huber, fixed validation MSE, LR, FDS schedule/support,
  learned ALF alpha/scale/log-partition for each epoch. Training NLL can be negative;
  its units differ from MSE. Do not interpret their absolute gap as overfitting.
- best.pt: head including FDS buffers, full configuration, panel, scaler, train/val/
  test split, cache signature, ALF state, optimizer/scheduler/RNG states. No encoder.
- fold_XX/outer_predictions/SAMPLE/: saved target, xy, prediction NPZs and complete
  manifest pointing to the selected patient-held-out checkpoint.
- primary_summary.json/.csv and primary_patient_metrics.json: existing overall
  equal-patient MAE/RMSE/Pearson/Spearman/bias/predicted-on-measured slope/intercept.
  Errors are in original intensity units. Patient bootstrap intervals use 2,000
  draws conditional on these fitted predictions; not full retraining uncertainty.
- oof_expression_patients.json / oof_expression_summary.json: same training-only
  90th-percentile high/low evaluation labels used for expression-aware comparisons,
  per-stratum MAE, AP, precision, recall, false-positive rate, positive fractions.
  Optional --thresholds JSON supplies every marker's raw assay threshold instead.
  These are evaluation labels ONLY; no intensity-based loss weighting is applied.

Undefined correlations, empty stratum errors, and AP with only one measured class
are JSON null, with defined-patient counts. Per-patient means avoid large slides
controlling the cohort score. Top-decile strata are not biological cell positivity.

## Standalone inference and plotting

For held-out cache predictions, use local predict.py with a new output path:

```bash
python /absolute/path/hex_fds_alf/predict.py \
 --checkpoint /absolute/run/fold_00/training/best.pt \
 --caches /absolute/cache_root/ACTUAL_HELD_OUT_SAMPLE \
 --out /absolute/NEW_predictions --role test --device cpu
```

The model's actual inference API is `model.eval(); model(features)`. No measured
mIF is supplied to the head. predict.py copies cached targets only for subsequent
comparison; it never passes them to the prediction call. It rejects training
patients in a held-out role and mismatched encoder/physical provenance.

compare_predictions.py provides the earlier sparse six-panel layout using two
completed CV roots (no head loading and no GPU). Use matched control as baseline
for attribution, or label an old MLP comparison honestly:

```bash
python /absolute/path/hex_fds_alf/compare_predictions.py \
 --baseline-root /absolute/CONTROL_cv_run --candidate-root /absolute/FDS_ALF_cv_run \
 --out /absolute/NEW_gallery --intensity-max 255 --positive-threshold 0 \
 --baseline-label 'Matched control' --candidate-label 'FDS + ALF' \
 --markers CAV CD8 CD138 COLIV POSTN
```

One shared color scale per marker/sample covers measured/baseline/candidate arrays;
all selected locations are shown in the top row. Bottom row selects GT above the
specified exploratory threshold and hides false positives outside it, so inspect
the top row too. Squares mark sampled centers, not dense 8x8/128x128 footprints.
Display/secondary metric division by 255 changes units, not accuracy. No clipping
of numeric predictions occurs. Calibrated confidence intervals are not implemented. The local dense review
launcher described below supports this checkpoint format. Old make_head/dense
inference scripts cannot load it unchanged; do not substitute their loader.

## Visualize completed control and FDS+ALF runs (2026-10-21)

The user reported both control_job_48680643 and fds_alf_job_48680648 complete with
39 patients and five folds. These exact paths are the defaults in two new launchers.
Copy the ENTIRE updated folder, including the three replicated TIFF/preparation/
raw-safety helpers, to preserve self-contained execution. Training scripts and
previous checkpoints do not need to be changed or rerun.

### Sparse out-of-fold comparison gallery

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf/plot_comparison.sbatch
```

CPU only. Output: uni2_oof_comparisons/hex_job_PLOTTING_JOB_ID/index.html.
Columns are measured mIF, matched control, FDS+ALF. Top row shows all retained
locations; bottom row uses identical GT>0 coordinates for all columns. This
threshold reproduces the earlier exploratory plot, NOT training-quantile labels
or biological positivity. Numeric reports use all locations. Shared per-figure
color scale includes all predictions; no numerical clipping. Plot symbols are
sampled centers and do not represent complete dense coverage.

### Dense same-ROI F03600 gallery

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/hex_fds_alf/dense_comparison.sbatch
```

One GPU. Output: uni2_dense_rois/f03600_hex_job_PLOTTING_JOB_ID/index.html.
`dense_from_review.py` follows the source reference in
uni2_dense_rois/f03600_cav_review_v1/manifest.json and reuses its exact ROI and
encoder/input provenance. The previously described native ROI was
x=3800..6200, y=27500..31500 (150,000 stride-8 centers); the script reads metadata
rather than assuming those coordinates. It verifies a complete F03600 stride-8
reference. Both new heads must be outer-held-out for the patient, match encoder/
physical metadata, share identical folds, and have the expected control/fds_alf
modes. `model.load_checkpoint` loads their architecture strictly and sets eval;
FDS and dropout remain disabled. No target labels are passed to the heads.

Dense figures have five columns: registered H&E; native measured mIF; measured
8x8 means; matched control; FDS+ALF. Context is still 128x128. Predictions are
contiguous block means, not native-pixel expression or individual cell labels.
All ROI blocks are included, without tissue filtering, matching the original
review. Existing dense outputs did not retain feature embeddings; frozen UNI2-h
is rerun once and the same features feed both heads. No training occurs.

Both launchers default to CAV CD8 CD138 COLIV POSTN VIM. Append
`--markers CAV CD8` to either sbatch command to restrict plotted markers. Sparse
plots also accept `--samples F03600`. Dense launcher can accept --review-root to
point to another completed *F03600* review; current resolver intentionally checks
sample identity. Use --baseline-root/--candidate-root to override the completed
runs explicitly. Each job snapshots code and saves environment/exit status.

The dense expression panels share percentile limits (1st/99th of combined block
GT/control/FDS+ALF values), matching previous dense reviews. Display clipping
can hide extremes; numeric predictions are retained unclipped. Limits differ
between independently generated galleries, so cross-model brightness comparisons
should use the panels WITHIN the same figure. Display settings and ROI metrics
are saved; a selected-ROI metric is not the full-cohort score.

dense_predictions.npz stores xy, target, control, fds_alf for the entire panel.
he_roi.png and measured_mif_native_roi.npz preserve the requested measured
channels. No dense feature cache is retained. Raw TIFFs are opened read-only;
raw_safety.py checks output placement and before/after metadata. Outputs must be
new and independent of raw sample folders, training runs, and cache roots.
Dependencies for this optional dense step additionally include timm, torchvision,
Pillow, tifffile, zarr and imagecodecs, already used by the earlier dense pipeline.

Copy the entire completed gallery directory to view index.html locally; its PNGs
and subdirectories must accompany it. These launchers prepare workstation jobs;
the coding agent has not generated actual remote patient figures locally.
