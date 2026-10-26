# Expression-weighted MLP versus the same MLP + FDS

This is a self-contained, fresh matched experiment. Copy this entire folder alone;
no parent scripts, source checkouts, symlinks or model downloads are used.
Head training reads only caches; optional dense inference reads raw images read-only. Installed scientific packages and existing cache/fold artifacts are inputs.
Existing expression_aware, hex_fds_alf and parent implementations remain unchanged.
No ALF, two-layer ReLU head, raw/255 regression loss, or loss blending is included.

## Precisely what is being compared

| Setting | weighted | weighted_fds |
|---|---|---|
| Encoder | Existing frozen UNI2-h features | Identical |
| Head | 1536 -> 256 -> GELU -> dropout .1 -> markers | Identical |
| Current panel | 42 markers, inherited order | Identical |
| Regression target | Training-only patient-balanced z score | Identical |
| Training objective | Expression-weighted Huber, delta1 | Identical |
| Optimizer | AdamW lr.001, weight decay.0001 | Identical |
| LR schedule | Constant; no scheduler | Identical |
| Batch/sampling | 256, patient-balanced replacement | Identical |
| Patient folds/seeds | Existing folds; seed42+fold | Identical |
| Checkpoint selection | Unweighted equal-patient validation Huber | Identical |
| FDS | Disabled | Training-time marker-specific feature calibration |

The 42-output head has 404,266 trainable parameters, exactly matching the earlier
expression-aware MLP. FDS buffers do not add trainable parameters. Both modes
allocate the same modules and use the same initial seeds. A separate sampler
random generator uses the same seed in both modes, independently of dropout.
This is not bit-for-bit reproduction of the old expression_aware run, whose
sampler used the global generator and whose stopping could occur before epoch51.
**Retrain BOTH modes here** rather than treating the old checkpoint as matched.

## Two distinct target representations

`sy` = (raw - training_mean) / training_std is the REGRESSION target. Means and
second moments are computed within each training patient and then averaged
across patients; SD is floored at1e-6. No validation/test statistics enter scaling.
Predicted z scores are inverse-transformed to original measured intensity for
all exported maps and primary metrics. No log transform or prediction clipping.

`ny` = raw / intensity_max is the FDS BINNING target ONLY. Default max255 matches
the verified uint8 QPTIFF cohort. Targets outside [0,255] are rejected, not clipped.
This includes declared-range validation of inner-validation targets, but no
fit/update from them. Passing z scores into FDS bins raises an error when out of
range. Each marker selects its own intensity bin. The output head still predicts
z scores; FDS never substitutes raw/255 for the regression objective.

The target itself remains the measured marker-channel mean across the central
8x8 block. Its context remains128x128, resized224 for the existing frozen encoder.
No image extraction is repeated; features[N,1536], target[N,M] and xy[N,2] come
from the same previously tissue-filtered caches. No cells, compartment targets,
new image filtering, or calibrated uncertainty are introduced.

## Expression weighting (same formula as earlier)

Each fold fits a marker-specific training-only raw-intensity threshold using the
90th percentile under equal-patient row weights. High means strictly greater;
ties are low. These are exploratory high/low strata, not biological positivity.
Optional --thresholds JSON must supply every marker's raw threshold; it replaces
quantile estimation for both training weights and stratified evaluation.

Let p be high prevalence under equal expected patient sampling. Start with:

```
w_high = min(0.5/p, 5)
w_low  = min(0.5/(1-p), 5)
```

Normalize both by p*w_high+(1-p)*w_low. If this exceeds the actual weight cap,
rescale both together so neither exceeds5. Thus expected weight can be less than1
for a marker after capping. All-high/all-low training markers use unit weights.
These are fixed training-prevalence weights, not unstable minibatch-specific
positive/negative denominators. Different markers at a row can have different
weights. The loss is mean(weights * Huber(predicted_z, measured_z, delta=1)).
Both low and high locations remain in training. No positive-only filtering or
oversampling by marker is applied. Per-row sampler weight1/n_patient instead
balances expected patient contributions. Large residuals are handled by Huber;
there are no learned adaptive shape or scale parameters.

## FDS implementation and schedule

The independently tested local fds.py is replicated from our prior isolated FDS
experiment. No runtime import from that folder occurs. It uses shared256-D hidden
features AFTER GELU/dropout and BEFORE the final linear output. Each marker has
its own 50-bin feature mean/variance statistics. For that marker, calibrated
features feed only its output weight row; no averaging over marker targets.

Training minibatch unsmoothed features are detached for moment collection, with
float64 sums/squared sums, n-1 sample variance and EMA momentum.9. First observed
supported bins initialize directly. Gaussian kernel9/sigma2 matches the earlier
SciPy delta construction and uses reflect padding along intensity bins. Empty
bins are excluded from smoothing with kernel renormalization; unsupported bins
and near-zero-variance dimensions bypass calibration. At least two observations
within an epoch are required to update a bin. Ratios are clipped[.1,10].
These safeguards are documented adaptations of HEX-style FDS, not exact source
reproduction. Local tests cover the math and evaluation safeguards.

Epoch schedule, ZERO-BASED:
- 0..9: collect moments, no calibration.
- 10..49: calibrate using previous completed epoch statistics; collect new moments.
- >=50: calibration and collection disabled.

The weighted-only mode neither applies calibration nor collects moments. Moment
statistics are NOT multiplied by expression loss weights: they follow the same
patient-balanced training draws. Multiplying moments too would introduce an
additional intervention. Repeated draws are not independent cells/patients;
training_bin_support.json separately reports unique rows and patients per bin.

Inference is always `model.eval(); model(features)`. Dropout/FDS disabled, no
measured labels needed, and buffers never updated. The new checkpoint format is
kidney_expression_weighted_fds_v1. Use local model.load_checkpoint or predict.py;
old head loaders are not interchangeable with this state dictionary.

## Common training and stopping protocol

Maximum100 epochs, patience10, minimum51 one-based epochs in BOTH modes.
Checkpoint selection and patience begin at epoch51, after FDS is disabled. This
ensures FDS actually runs before selection and both variants have the same
selection window/budget. Earlier validation values remain logged. The CLI rejects
min_epochs <= fds_off in either mode, to preserve the matched protocol. In tiny
synthetic tests a shortened schedule is explicitly requested; it is not the
workstation default. No scheduler or optimizer changes were carried over from
our HEX-like experiment.

Validation: standard unweighted Huber in training-fitted z-score units, averaged
within each inner-validation patient then equally over patients/markers. Training
is expression weighted, so its objective magnitude is not directly comparable
to the validation value. No test predictions select checkpoints, fit scalers,
fit expression thresholds or update FDS. Seeds are fixed; exact bitwise CUDA
reproducibility is not guaranteed, particularly for indexed moment accumulation.

## Run on the workstation

Copy this WHOLE folder to:

```
/scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds
```

Reuse the existing kidney-uni2 environment. No dependency installation or encoder
weight download is needed. NumPy/PyTorch are required; SciPy verifies kernel
construction and Matplotlib is used only by plotting.

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs

sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds/run_cv.sbatch weighted

sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds/run_cv.sbatch weighted_fds
```

Each job requests one GPU, four CPUs,32GB and four hours on emmalu. Both can queue
or run independently. Defaults reuse uni2_cohort_cache/cohort39_v1 and the reviewed
uni2_cv_runs/cohort39_mlp_job_47815087/folds.json, not scripts from that old run.
Every patient must appear in exactly one outer test fold; all samples retain their
patient mapping. Cache panel, encoder SHA/revision, transforms, MPP, target block,
context and selection signatures must agree. No arbitrary subsampling/exclusion.
Parent CV scoring stays CPU-only; fold training owns CUDA in isolated subprocesses.

Each job snapshots its source, saves pip freeze and Slurm metadata, runs unit tests
(including synthetic real CPU training/CV), then performs CUDA preflight in a
separate process. No automatic resume or overwrite. Preflight estimates numeric
memory and guards it at8GB by default; overhead is additional. All caches/raw data
remain read-only. Head training never reads raw images; the optional dense review
uses read-only TIFF readers and before/after metadata checks.

For alternate data/folds, use cross_validate.py --cache-root PATH
--reference-folds PATH --out NEW_PATH --mode weighted_fds --device cuda.
--plan-only validates and writes a fold plan without training. Other CLI settings
must be applied identically to both modes for a matched comparison.

## Outputs, audit and interpretation

New root: uni2_weighted_fds_runs/MODE_job_JOBID. Logs:
uni2_slurm_logs/weighted-fds-JOBID.out/.err; fold epoch logs are
fold_XX/training.log. Job records contain tests.log, doctor.json, environment.txt,
submitted script, source snapshot and status.txt.

Per run: config.json, inventory.json, folds.json, preflight.json, invocation.json,
source hashes, events.jsonl; final manifest.json is complete only after all folds.
Per training fold: config, scaler, expression_state (thresholds/prevalence/weights),
unique training bin support, history.json (weighted train objective, unweighted
validation Huber, LR, FDS phase and sampled support), best.pt (strict head state,
FDS buffers, optimizer and RNG state, config/panel/splits/cache provenance),
validation regression/stratified metrics, and final run.json.

OOF predictions preserve unchanged xy/target with prediction in original units.
primary_summary.json/.csv report equal-patient MAE, RMSE, Pearson, Spearman, bias,
and predicted-on-measured slope/intercept. Bootstrap intervals resample patients
conditional on fitted predictions, not a full retraining bootstrap.
oof_expression_summary.json and oof_expression_patients.json contain high/low MAE,
AP, precision, recall, FPR and predicted/measured positive fractions. Undefined
metrics are null with defined-patient counts. Evaluate rare-expression recovery
alongside false positives and bias; neither weighting nor FDS guarantees improvement.

We have already used previous outer results to guide development. These follow-up
comparisons remain exploratory; avoid treating them as untouched confirmatory tests.
No model here fixes missing center-level information in the pooled embedding.

## Verify the pair and make the same OOF plots

After both runs complete:

```bash
python /absolute/path/expression_weighted_fds/verify_pair.py \
 --baseline-root /absolute/weighted_run --candidate-root /absolute/weighted_fds_run
```

This rejects different architecture/settings, folds, sample/row inventories,
training arguments (except mode/output paths), scalers or expression thresholds/
weights. Different selected epochs are expected and are not rejected.

The CPU plotting launcher runs this check before the familiar six-panel gallery:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds/plot_comparison.sbatch \
 /scratch/groups/emmalu/hande_annotation/uni2_weighted_fds_runs/weighted_job_BASELINE_ID \
 /scratch/groups/emmalu/hande_annotation/uni2_weighted_fds_runs/weighted_fds_job_FDS_ID
```

Output: uni2_oof_comparisons/weighted_fds_job_PLOT_ID/index.html. Columns: measured,
expression-weighted MLP, expression-weighted MLP+FDS. Defaults CAV CD8 CD138 COLIV
POSTN VIM; append --markers or --samples to change plotted selection. Top row all
locations, bottom row GT>0 at identical locations across models. Bottom row omits
false positives elsewhere, so inspect top row. The display threshold0 is NOT the
training90th percentile. Colors share one scale per figure; values are divided
by255 for display and secondary reports only. Missing coordinates are not zeros.
The optional dense ROI launcher below uses the local new-format loader; old
linear/MLP or HEX checkpoint loaders must not be substituted.

## Verification

Run doctor.py --device cpu, python -m unittest discover -s THIS_FOLDER -v,
python -m compileall -q THIS_FOLDER, and bash -n on launchers. See VALIDATION.md
for observed results. Synthetic success is not biological improvement. No remote
cohort run has been submitted by the coding agent while creating this folder.

## Generate both galleries from completed runs

Copy the entire updated expression_weighted_fds folder to the workstation.
The plotting launchers now accept NO positional arguments when there is exactly
one completed weighted run and one completed weighted_fds run under
/scratch/groups/emmalu/hande_annotation/uni2_weighted_fds_runs. They ignore
incomplete runs, verify the selected modes, record the exact paths, then execute
verify_pair.py. Multiple completed candidates produce an actionable error rather
than silently selecting a different seed or rerun. In that case provide BOTH
run directories as the first two arguments to either launcher.

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs

# CPU: same sparse OOF layout as previous comparisons.
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds/plot_comparison.sbatch

# GPU: same saved F03600 ROI, weighted-only versus weighted+FDS.
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/expression_weighted_fds/dense_comparison.sbatch
```

Outputs (different plotting job IDs):
- uni2_oof_comparisons/weighted_fds_job_JOBID/index.html
- uni2_dense_rois/f03600_weighted_fds_job_JOBID/index.html

Default markers for both: CAV CD8 CD138 COLIV POSTN VIM. Append --markers MARKER...
to restrict markers. Sparse plotting accepts --samples F03600 as well. For
explicit run selection use the same two run paths shown in the previous section;
additional plot flags follow them. No retraining occurs in either job.

### Dense mechanics and safety

`dense_from_review.py` follows source_dense_root in
uni2_dense_rois/f03600_cav_review_v1/manifest.json to its original inference run.
The exact native ROI, raw input/cache root, intensity divisor and encoder
checkpoint path are copied from saved metadata, not guessed from a screenshot.
The earlier ROI x3800..6200, y27500..31500 has150000 stride-8 locations; actual
coordinates always come from that reference. The resolver requires a completed
F03600 stride-8/8x8 run and rejects reference cycles or inconsistent ROI records.
An optional --review-root points to another completed F03600 dense review.

Both selected heads must be outer-held-out for F03600's patient and have matching
panel, folds, encoder SHA, transform, MPP, context/target and selection policy.
The strict local checkpoint loader supports the expression-weighted GELU head
and its z-score scaler, sets eval and freezes parameters. FDS/dropout are off;
no measured mIF values are passed to either head. Each predicts standardized
values, which are inverse-scaled using its saved training-only mean and SD.
Dividing final raw intensities by255 is only for display/secondary ROI metrics.

Old dense runs saved predictions but not the dense grid's embeddings. Therefore
UNI2-h is rerun once per ROI location, with exactly the frozen architecture and
224-square/ImageNet preprocessing recorded in the training caches. The same
features feed both heads. No filtering or training is repeated. All ROI blocks
are included, matching the original dense reviews, including background blocks.

Five-panel dense figures: H&E, native measured mIF, measured8x8 means,
expression-weighted MLP, expression-weighted MLP+FDS. Each map pixel is an8x8 block
mean; it is not cell-level/native-pixel expression. Percentile display limits
(1st/99th of combined measured-block and model values) are shared within each
figure. Display clipping can hide extremes; numeric arrays are NOT clipped.
Different galleries can have different limits, so compare models within the same
figure or inspect display_settings.json before comparing colors across figures.

Saved outputs: dense_predictions.npz (xy,target,weighted,weighted_fds for the whole
panel); he_roi.png; measured_mif_native_roi.npz (requested native channels);
roi_metrics_normalized.json; display_settings.json; reference_review.json; complete
manifest and source/input audit records. No dense embedding cache is retained.
Selected-ROI metrics are not substitutes for patient-level OOF evaluation.

Raw H&E/mIF are opened read-only with local image_io.py. raw_safety.py guards
output locations and before/after source metadata. All outputs must be new and
independent of raw sample folders, cache roots, source reviews and training runs.
Dense requests one GPU,64GB/fourCPUs/fourhours on emmalu. It needs the same timm,
torchvision, Pillow, tifffile, zarr, imagecodecs and Matplotlib packages as the
previous working dense pipeline; reuse that environment rather than replacing it.

Job logs: weighted-fds-plots-JOBID.out/.err or weighted-fds-dense-JOBID.out/.err
under uni2_slurm_logs. Source snapshots, selected_runs.txt, pair_check.json,
environment and exit status are in per-job record folders. Copy the entire
completed gallery directory to view index.html locally, including its images.
Actual remote patient figures cannot be generated by the coding agent locally.
