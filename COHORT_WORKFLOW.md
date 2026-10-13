# Audited cohort workflow for the remote CLI agent

Use the entire updated `kidney_uni2` directory, or merge its changes into the remote
version if the workstation agent has added its own memory preflight enhancements.
Never mix old background-pilot caches with cohort caches. Preserve earlier runs.
No dependency on any external preprocessing source directory exists.

## Protocol

1. Discover exact file pairs across uppercase B/F/K/M/R sample folders.
2. Scan fully in-bounds 128x128 H&E contexts, on a stride-128 grid by default.
3. Require >=50% tissue-like pixels in both context and its central 8x8 target.
4. Tissue-like pixels have mean RGB >20 and <235, and max-minus-min RGB >15.
   These are starting heuristics, not validated kidney segmentation thresholds.
   Black padding and white background are excluded; pale tissue may also be lost.
5. With a support mask, require the entire context valid. Without a mask, the
   explicit full-support assumption is recorded; tissue filtering cannot establish
   registration support. Never filter based on protein positivity.
6. Uniform reservoir sample up to 5,000 eligible grid centers per sample, seed 42.
   Record eligible-but-not-sampled count and exact final coordinates. This cap is
   an intentional first cohort experiment setting, not all-slide dense inference.
7. Compute raw 8x8 marker means, reordering by verified QPTIFF channel labels.
8. Cache frozen UNI2-h embeddings once and reuse them across folds.
9. Five outer patient folds. Each patient is evaluated once; all its samples stay
   together. Within each fold, use 20% of remaining patients for early stopping.
   A fresh head and training-only scaler are fitted each fold. The outer test
   patients never influence checkpoint selection. Hyperparameters must be fixed
   before reading outer results; tuning against these results would require nested
   inner CV or a separate untouched final evaluation cohort.
10. Compare raw-scale predictions to an equal-training-patient mean baseline.
    Combine all samples of a test patient before metrics; aggregate equally across
    patients. Correlation undefined on constant channels is null, not zero.

Default folds and thresholds are explicit configurable choices. Splits are randomly
shuffled, not acquisition/disease-stratified: add verified metadata and specify a
stratified design before reporting scientific conclusions if those groups matter.

## First check filtering on three patients

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/cohort_prepare.py \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --panel /scratch/groups/emmalu/hande_annotation/uni2_panel.json \
  --out /scratch/groups/emmalu/hande_annotation/uni2_filter_check \
  --mpp 0.5073077046562237 --samples-as-patients --assume-all-valid \
  --samples M5172118 M5173023 M5173144 \
  --stride 128 --max-per-sample 5000 --min-context 0.5 --min-center 0.5
```

Full-support assumption above is explicit, not proven by registration. Replace it
with `--valid-mask-name common_validity_mask.tiff` if support masks become available.
Open each sample's `filter_review.html` after copying its whole directory locally.
Inspect retained and rejected examples and `filtering.json` counts. Do not adjust
thresholds based on held-out marker prediction results. Lock settings after H&E QA.

## Prepare all eligible patients with fixed settings

Use a new root and omit `--samples` from the command above, setting
`--out /scratch/groups/emmalu/hande_annotation/uni2_cohort_prepared_v1`.
Missing pairs are explicitly reported as excluded/failed; all other samples continue.
The exit code is nonzero if any were excluded/failed, including known missing pairs.
Review `cohort_report.json` rather than assuming all patients succeeded. Each retained
sample's complete manifest is the gate for extraction. The user instructed sample
IDs be patient IDs; revise mapping if repeated patients are later identified.

## Extract selected tissue features

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/cohort_extract.py \
  --prepared-root /scratch/groups/emmalu/hande_annotation/uni2_cohort_prepared_v1 \
  --out /scratch/groups/emmalu/hande_annotation/uni2_cohort_cache_v1 \
  --checkpoint /scratch/groups/emmalu/hande_annotation/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin \
  --revision d517a8dd47902dd7c308b3c36f63bce47e7b9a43 \
  --batch-size 16 --device cuda
```

Extraction is sequential by sample and loads the encoder once per subprocess.
Per-sample logs include crop progress. Review `extraction_report.json`; any failure
must be resolved or explicitly accepted as a cohort exclusion before CV. The CV
inventory lists only completed caches. One patient's multiple samples can exceed
5,000 total crops; training weights still give that patient equal expected weight.

## Review patient fold plan, then train

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/cross_validate.py \
  --cache-root /scratch/groups/emmalu/hande_annotation/uni2_cohort_cache_v1 \
  --out /scratch/groups/emmalu/hande_annotation/uni2_cv_plan_v1 \
  --folds 5 --val-fraction 0.2 --hidden 0 --seed 42 --plan-only
```

Review `inventory.json` and `folds.json`. Execute with the same seed/settings and
new output path (identical inventory gives identical folds):

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/cross_validate.py \
  --cache-root /scratch/groups/emmalu/hande_annotation/uni2_cohort_cache_v1 \
  --out /scratch/groups/emmalu/hande_annotation/uni2_cv_linear_v1 \
  --folds 5 --val-fraction 0.2 --hidden 0 --seed 42 \
  --epochs 100 --patience 10 --batch-size 256 --device cuda
```

At 37x5,000 crops, feature data alone are approximately 1.14 GB. Training remains
in-memory but deliberately bounded by the selection cap. CV refuses a conservative
numeric array estimate >8 GB by default, configurable with `--max-numeric-gb`.
This estimate excludes Python/GPU/framework overhead; Slurm RAM limits still apply.
Do not increase selection size without checking memory and storage.

## Records and outputs

- `invocation.json`: arguments, Python version, argv, source SHA256 hashes.
- `source_snapshot/`: exact source and documentation used for the stage.
- `events.jsonl`: timestamps, stage transitions, failures, training epoch history.
- Discovery/cohort reports: all discovered samples and reasons for exclusion/failure.
- Per-sample `candidate_decisions.csv`: context/center tissue fractions and sequential
  first rejection reason for each probed location. Boundary centers are excluded by
  grid construction, not included in CSV; see filtering boundary rule.
- `filtering.json`: thresholds, counts, sampling/support assumptions, preview paths.
- `selected_coordinates.json`: final sampled locations; raw target shards and manifest.
- Extraction manifest: checkpoint SHA256, supplied revision, transform/software,
  sample/patient metadata, filtering policy. Filtering report is copied into cache.
- CV `folds.json`/`inventory.json`: patient partitions and exact included cache paths.
- Per fold: split, training log, scaler, epochs, best head checkpoint, validation
  metrics, outer predictions at exact coordinates, patient metrics with baseline.
- `oof_patient_metrics.json`: out-of-fold model/baseline metrics for every patient.
- `oof_summary.json`: equal-patient mean and descriptive patient SD per marker.
  SD is not a confidence interval. Sparse predictions are not dense marker maps.

Run tests and compilation before remote execution. Local tests cover filtering,
seeded sampling, marker targets, provenance, undefined correlations, and fold leakage.
Actual cohort image reads and GPU cross-validation are unverified locally and must
be tested on the workstation. No automatic resume/overwrite is supported. Failed
stage directories are preserved for diagnosis; rerun with a new output path.
