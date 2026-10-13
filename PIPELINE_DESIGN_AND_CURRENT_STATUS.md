# Kidney marker prediction with frozen UNI2-h: design, settings, and execution record

Updated **October 12, 2026**. This is the current technical explanation of the
pipeline, based on the local implementation and workstation outputs shared in this
chat. Exact remote source snapshots and run logs take precedence if the workstation
agent has made additional changes. Historical ROI results are preserved in
[PIPELINE_AND_SMOKE_TEST.md](PIPELINE_AND_SMOKE_TEST.md). Operational commands are
in [COHORT_WORKFLOW.md](COHORT_WORKFLOW.md) and [slurm/README.md](slurm/README.md).

## 1. Purpose and scope

The scientific task is to learn local associations between kidney H&E morphology
and measured protein-marker expression from registered H&E–mIF pairs. The chosen
adaptation strategy freezes a pathology foundation encoder and learns a new head.
UNI2-h is the initial backbone, not a demonstrated winner over Virchow2 on this task.
The encoder comparison has not been performed on the kidney cohort.

The current model estimates 42 numerical marker values for a small central tissue
block. Its input is a larger H&E context crop. The immediate goals are reproducible
training, patient-separated evaluation, and coordinate-aligned predictions. Dense
virtual staining, cell-level analysis, kidney compartment supervision, and calibrated
uncertainty are later developments; none is established by the current pilot results.

```text
Registered sample folders
  → exact pair discovery and metadata/marker checks
  → H&E tissue and support selection
  → reproducible spatial sampling
  → paired context crops and central marker targets
  → frozen UNI2-h feature extraction and caching
  → patient-separated regression-head training
  → checkpoint selection on inner validation patients
  → quantitative and spatial evaluation
  → later cohort out-of-fold aggregation
```

Only the head learns from our marker targets. We are not reproducing UNI2-h
pretraining and are not currently updating any encoder layer.

## 2. Data discovery, identities, and physical scale

Remote base directory: `/scratch/groups/emmalu/hande_annotation`.

| Item | Implemented rule |
|---|---|
| Sample folders | Immediate subdirectories starting with uppercase B, F, K, M, R |
| H&E file | Exactly one .tif/.tiff with case-sensitive `HandE` in the filename |
| mIF file | Exactly one .qptiff |
| Extensions | Matched case-insensitively |
| Missing/ambiguous files | Reported, not arbitrarily chosen |
| Current identity convention | Sample ID = patient ID, per user instruction |

Discovery found 37 labeled pairs with a consistent 42-marker set. That is a discovery
count, not a completed cohort training count. Missing QPTIFFs for M5172099 and
M5174001 were visible in the supplied discovery excerpt. More exclusions can occur
later because of metadata, tissue, decoding, or support failures.

Inspected H&E exports are uint8 RGB with `YXS` axes; mIF is uint8 with 42 channels,
usually `CYX`. Spatial dimensions match within inspected pairs. QPTIFF tags report
isotropic **0.5073077046562237 µm/pixel**. H&E has no physical-size tags in the inspected
exports. The supplied scale therefore assumes registration placed H&E onto the same
physical grid, origin, and orientation. Matching array dimensions alone cannot prove
this. Preparation rejects available TIFF-tag scale mismatches beyond 1%.

Sample IDs must be replaced by verified patient mapping if multiple samples belong
to one person. Patient grouping, not tile grouping, defines independent evaluation.

### Marker identities

`uni2_panel.json` stores the ordered marker identities checked against discovery.
Embedded QPTIFF labels are verified and reordered by exact identity. Matching channel
counts alone is insufficient. The initial full-panel adapter requires every panel
marker measured; missing values must not be represented as zeros.

The panel includes DAPI, which is a nuclear stain rather than a protein marker.
Thus “42 marker/stain channels” is the most precise description of the output panel.
The existing smoke-test table records all 42 names and metrics.

## 3. Registration support and tissue are different checks

Registration addresses spatial correspondence. Common support addresses whether
both image exports contain actual data at that location. Tissue selection addresses
whether the H&E example is biologically relevant. These checks are not interchangeable.

No registration-derived support mask has been supplied. Current runs explicitly use
`--assume-all-valid`. As a result, `invalid_support: 0` does not mean support was
measured and proved valid. It reflects that every position was treated as supported.
Visual correspondence has been checked in selected regions, not quantified for the
whole cohort. Tissue filtering excludes obvious black/white regions but cannot
establish registration quality or rule out missing modality coverage.

If support masks become available, supply `--valid-mask-name`. Cohort selection
requires the whole 128×128 context to be nonzero in that YX mask. This is conservative:
its context support requirement is broader than the 8×8 target itself. Mask dimensions
must match the pair. Do not construct validity from positive marker expression.

## 4. One example: context and target geometry

| Component | Geometry | Physical width at the supplied scale |
|---|---|---:|
| H&E context | 128×128 native RGB pixels | About 64.94 µm |
| mIF target | Central 8×8 native pixels | About 4.06 µm |
| Encoder input | Context resized to 224×224 | Still about 64.94 µm |

The larger image supplies surrounding morphology; the smaller target specifies
where expression is estimated. For marker m, average its 64 central mIF pixels.
Targets retain the exported intensity units. Because source marker images are uint8,
these values reflect that export's intensity representation, not absolute protein
concentrations. No additional background subtraction, blank-cycle correction,
acquisition calibration, or marker normalization is added at preparation.

Coordinates are `[x,y]` = `[column,row]`. Original dense target centers follow
`(4+8k,4+8j)`. A cohort example centered at `(x,y)` has target bounds
`[x-4:x+4, y-4:y+4]` and context bounds `[x-64:x+64, y-64:y+64]`.

The earlier ROI adapter permits zero-padded image contexts near boundaries. Cohort
selection instead uses only fully in-bounds contexts. The first cohort center is
`(68,68)`, maintaining the target grid's modulo-eight origin. No registration or
resampling is performed by this code.

## 5. Tissue filtering: exact rules and rationale

A pixel is tissue-like when all of the following are true:

```text
mean RGB > 20
mean RGB < 235
maximum RGB − minimum RGB > 15
```

For a candidate context:

| Check | Default acceptance |
|---|---|
| Context tissue-like fraction | >= 0.50 in 128×128 crop |
| Central tissue-like fraction | >= 0.50 in central 8×8 block |
| Common support | Whole context valid if an actual mask is supplied |
| Boundary | Whole context lies inside image |

Brightness excludes black padding and very bright background; color range favors
stained pixels. The context threshold avoids predominantly empty images. The center
threshold avoids predicting an empty target inside a larger tissue-containing image.
No protein-channel value influences eligibility. This retains marker-negative tissue
and avoids selection based on the outcome we want to predict.

These are simple H&E heuristics, not a validated kidney segmentation model. Pale
stroma, weakly stained cells, tissue borders, or lumens can be rejected, and artifacts
can pass. The user reviewed retained/rejected diagnostic examples and accepted the
thresholds for the current initial experiment. This is qualitative review, not a
measured sensitivity/specificity assessment. Rejected previews should be retained.

Rejection labels are sequential: `invalid_support`, then `low_context_tissue`, then
`low_center_tissue`. A position failing multiple tests is counted under the first
failure. Therefore these categories are mutually exclusive, not independent counts
of every violated criterion.

## 6. Sampling and data volume

Defaults:

- Native candidate stride: **128 pixels** in each direction.
- Maximum retained locations: **5,000 per sample**.
- Reservoir sampling seed: **42**.
- Prepared shard size: **1,000 examples**.

The stride reduces redundancy relative to the original ROI experiment, which used
8-pixel steps and heavily overlapping contexts. Nominal stride-128 contexts do not
overlap along that grid. However, examples from one slide/patient still are not
independent patients.

Uniform reservoir sampling considers eligible grid positions across the slide and
caps their number rather than keeping the first eligible locations. It is not
kidney-compartment-balanced sampling and can underrepresent rare structures. The
maximum is per sample, not per patient if multiple samples are later grouped.

The random generator is seeded using `[seed, sample enumeration index]`. Enumeration
is sorted discovery order after any explicit sample subset. Therefore a separate
recovery invocation can select different capped locations from a sample than a full
cohort invocation. Exact selected coordinates are the reproducibility record. In
M5173144 recovery, all 2,341 eligible locations were retained, so the cap did not
change that sample's location set.

Five thousand crops with 1,536 float32 features need 30,720,000 bytes of features plus
840,000 bytes of 42-channel float32 targets. At 37 samples reaching the cap, there
would be up to 185,000 examples and about 1.14 GB of features alone. Training currently
loads arrays into RAM; it is not an out-of-core training implementation. The cap and
CV numeric-memory guard limit the initial experiment rather than proving arbitrary
whole-slide scalability.

Diagnostic mode limits the scan to 64 spread-out grid positions per sample and caps
retention at 32. Those outputs are explicitly diagnostic-only and are rejected by
training/CV. Their retention percentages are not whole-slide estimates.

## 7. How our encoder use follows UNI2-h settings

Authoritative reference: [official UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h),
checked October 12, 2026. Our extractor follows its **manual local-checkpoint loading
example**, rather than retraining the published model.

| timm setting | Value |
|---|---|
| Model factory name | vit_giant_patch14_224 |
| img_size / patch_size | 224 / 14 |
| depth / num_heads | 24 / 24 |
| embed_dim | 1536 |
| init_values | 1e-5 |
| mlp_ratio | 2.66667 × 2 |
| num_classes | 0 |
| no_embed_class | True |
| mlp_layer / act_layer | SwiGLUPacked / SiLU |
| reg_tokens | 8 |
| dynamic_img_size | True |

`pretrained=False` constructs the architecture; we then load the downloaded state
with `strict=True`. It does not mean random features are used. `weights_only=True`
loads the tensor state. The final weights come from the approved checkpoint.
The factory name alone does not determine this custom architecture: supplied
arguments match the model-card configuration. These encoder settings and the
1,536-dimensional output are documented in the official card.

Checkpoint revision: `d517a8dd47902dd7c308b3c36f63bce47e7b9a43`.
Remote file:

```text
/scratch/groups/emmalu/hande_annotation/checkpoints/uni2-h/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin
```

The approximately 2.73 GB download is authenticated and stored on scratch. Extraction
computes the file SHA256 and records it. The supplied repository revision and file
hash are different provenance fields; the extractor cannot independently prove that
a user-supplied revision corresponds to the local file. The download procedure pinned
that revision before retrieving the file.

### Input transform and output

The manual example prescribes resize to 224, tensor conversion, and normalization
with mean `(0.485,0.456,0.406)` and SD `(0.229,0.224,0.225)`.
Our square input uses explicit `Resize((224,224))`, then `ToTensor`, then those values.
For a square 128×128 crop this produces the same output geometry as scalar
`Resize(224)`. There is no center crop, random augmentation, or separate stain
normalization in the extractor. Normalization is input-channel scaling, not stain
normalization or marker-target standardization.

We call `model(batch)` to obtain the model's pooled 1,536-number image embedding.
We do not reshape spatial patch tokens into a dense decoder or treat register tokens
as anatomical locations. Each embedding becomes one row of the feature cache.

### Frozen execution and deliberate project choices

`requires_grad_(False)`, `eval()`, and `torch.inference_mode()` keep the encoder fixed
and disable training-mode behavior/gradient tracking. We currently use float32
inference and save float32 features; no autocast is enabled. This is an implementation
choice, not a reproduction of the author's large-scale mixed-precision pretraining.

The 128-native-pixel field of view, central 8×8 protein target, tissue thresholds,
sampling cap, regression head, loss, optimizer, and patient splits are **our task
choices**. They are not UNI2-h's published kidney marker-regression recipe. Matching
its input size and normalization does not guarantee the optimal biological scale.
Field-of-view comparisons may be appropriate later.

## 8. Image I/O bottleneck, diagnosis, and caching

TIFF/Zarr reads ordinarily decode bounded regions without loading a slide into a
persistent array. The inspected M5172118 H&E, however, exposed shape AND chunk size
`[13680,16320,3]`: one whole-slide chunk, approximately 670 MB of raw RGB. A small crop
can repeatedly trigger decoding that chunk. mIF exposed smaller `[1,512,512]` chunks.

The full original filtering job took four hours and timed out after two samples.
A bounded diagnostic showed most measured time in H&E reads. The optimized path
optionally decodes a single-chunk H&E once into RAM, then slices the same pixels.
Conditions: single chunk and raw image size within `--he-cache-gb` budget. Current
batch scripts set that budget to **2 GiB** and request sufficient RAM. Tiled or larger
images use regional reads. Decoding has temporary-memory overhead beyond raw bytes.

This cache is not a changed image transform: no intensity or coordinate changes occur.
Local tests verify cached crop/padding correspondence on synthetic data. Remote
M5173144's optimized whole candidate scan completed in about 20 seconds, while target
reading/writing brought the recovery job total to 3 minutes 24 seconds. Do not turn
this into an exact old-vs-new speedup claim: the earlier job involved different
patients/nodes and included additional work.

Progress is now emitted before/after header inspection, reader opening, initial
reads, periodic scan counts, and marker-target reads. The CSV is flushed immediately.
`progress.json` stores current stage and timing data. Faulthandler stack dumps every
120 seconds diagnose slow operations; the message `Timeout (0:02:00)!` is not a Slurm
termination. Sample exceptions print tracebacks and are recorded.

## 9. Feature caching and extraction safeguards

`extract.py` reads prepared shards, reconstructs contexts at their coordinates,
processes batches, validates output shape `[B,1536]` and finite values, then saves
features with corresponding raw targets and coordinates. It writes the complete
manifest only after successful extraction. Existing outputs are refused.

| Setting | CLI default | Current larger pilot batch |
|---|---:|---:|
| Extraction batch size | 32 | 16 |
| Device | cuda | cuda |
| H&E read cache | Off | 2 GiB budget |
| Feature dtype | float32 | float32 |
| Crop progress | First, every 25 batches, final batch | Same |

The original tiny smoke test used batch size 4. Batch sizes affect memory/throughput,
not intended target geometry. Extraction reads fixed views; there is no random
training augmentation in cached features.

## 10. Head architecture, scaling, optimization, and stopping

Linear head (`--hidden 0`): `1536 → 42`, weights plus bias = **64,554 parameters**.
Default standalone trainer (`--hidden 256`) instead constructs:

`Linear(1536,256) → GELU → Dropout(0.1) → Linear(256,42)`.

That MLP has 404,266 parameters. The pilot and default CV commands explicitly choose
the linear head; the MLP is implemented but no comparative result has been reported.
No pretrained protein head is retained.

Targets are standardized marker-by-marker using training patients only. For each
patient, compute mean and second moment, then average those equally across patients.
The resulting SD includes within-patient and between-patient variation under equal
patient weighting. Floor SD at 1e-6. Optional `--log1p` precedes scaling and requires
nonnegative values; it was not used in the reported pilots. Save mean/SD/transform
and use them unchanged for validation/test; invert for raw-scale metrics.

| Optimization parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 default |
| Weight decay | 1e-4 |
| Loss | Huber delta 1, mean over batch and standardized channels |
| Head batch | 256 default |
| Seeds | torch and numpy; 42 in pilots |
| Sampling | WeightedRandomSampler with replacement |
| Per-example sampling weight | 1 / number of training examples for its patient |
| Epoch size | Number of training examples, sampled with replacement |
| Pilot max epochs / patience | 30 / 5 |
| Standalone/CV defaults | 100 / 10 |
| Scheduler, clipping, encoder updates | None |

Patient-balanced sampling makes total expected patient contribution equal. It does
not balance kidney compartments or marker-positive events. One epoch need not visit
every row exactly once. All training patients' selected examples feed one shared
head per round; images/targets are not averaged together before training.

Validation Huber is averaged within each patient, then equally across patients.
Any strictly lower score saves `best.pt`; there is no minimum-improvement tolerance.
Stop after patience consecutive epochs without improvement and reload the best head.
Only head parameters enter the optimizer. Seeds improve reproducibility, but GPU
bitwise determinism is not enforced.

## 11. Evaluation: validation versus independent testing

For the proposed three-patient pilot, train on M5172118 + M5173023 and validate on
M5173144. This split was recommended, but a completed training result or final split
has not yet been supplied. Do not claim it executed until the run records confirm it.
If that patient selects the checkpoint, its scores are validation analysis, not
independent testing. Training on all three would leave no independent validation
patient under the current design.

Evaluate per marker with raw-scale MAE, RMSE, and Pearson. Undefined correlations
(constant vectors or too few values) are null. Nonfinite measured/predicted data are
errors. The baseline predicts the equal-training-patient mean of raw intensities.
The baseline is a simple mean reference, not an optimized MAE/median predictor.

MAE captures magnitude error; correlation captures relative location-associated
variation. Low MAE with negative correlation need not produce useful marker maps.
Raw MAE should be compared within marker, not used to rank different markers with
very different scales. No clipping is applied to numerical predictions, so linear
outputs can be negative or exceed the acquisition export range.

`evaluate_head.py` checks checkpoint roles/provenance, reproduces the baseline,
exports measured/predicted arrays, and creates all-marker metrics plus selected
figures. It requires explicit `--role val` or `test`; test scoring requires actual
checkpoint test patients. Default preview channels: CD10, VIM, NES, UMOD, DAPI.
Plots use one shared 1st–99th percentile display range per marker/sample across
measured and predicted values. Display clipping does not alter metrics. Scatter
plots use all raw values. Matplotlib is optional; `--no-plots` produces numeric
outputs without it. CPU suffices for small-head inference on cached features.

Sparse displays leave unmeasured cells gray. Each square represents a sampled 8×8
target, not continuous expression throughout its larger sampling-stride cell. These
are sparse maps, not validated dense virtual staining. No calibrated uncertainty
is produced. Multi-head disagreement is separately available in `predict.py`, but
has not been assessed here and must not be presented as calibrated confidence.

## 12. Patient-level cross-validation

Implemented default: five outer folds, seed 42. Keep every sample/crop of a patient
in the same fold. Each round trains a fresh head. Set aside 20% of remaining patients
(rounded up, with at least one training patient left) for inner early stopping.
Training patients alone fit scaler and head. Outer patients influence neither.

At 37 patients, typical counts are 7–8 outer evaluation, 6 inner validation, and
23–24 training. Each patient is outer evaluated once; those predictions are
**out-of-fold**. Combine multiple samples within patient before scoring, then report
equal-patient means and descriptive patient SD. Undefined-correlation patient counts
are recorded; SD is not a confidence interval.

Five folds is a configurable practical default, not a UNI2-h requirement. The runner
requires >=4 independent patients, so the three-patient pilot is not its CV experiment.
Cached frozen features are reusable across folds; learned heads/scalers are not.
Current folds are random, not disease or acquisition-batch stratified. Verify such
metadata before adopting a stratified scientific evaluation.

This is outer CV with an inner holdout for early stopping, not full inner CV for
hyperparameter selection. Fix settings before reading outer scores. Tuning against
outer results requires a revised evaluation design. No completed cohort CV results
have been reported yet.

## 13. Execution history and verified results

### Early ROI smoke tests

See the historical README for all 42 metrics and exact learning curves.

- Initial training M5172118 ROI: `4096 3072 4608 3584`.
- Validation M5173023 ROI: `10536 8296 11048 8808`.
- Each 512×512 region yielded 4,096 target examples at stride 8.
- The initial training region was later visually confirmed to contain no tissue.
  Tiny training marker variance amplified validation error (about 76 standardized
  Huber). A constant-vector Pearson NaN also broke JSON export; corrected metric
  code and checkpoint report recovery fixed that engineering issue.
- Replacement training ROI: `14224 6584 14736 7096`, containing tissue.
- Corrected linear pilot selected epoch 11; best validation Huber 5.697199;
  early stopping after epoch 16. The model had lower MAE than the mean baseline
  for 22/42 markers. Selected Pearson values: CD10 0.375, VIM 0.354, NES 0.259,
  UMOD 0.237. This was one training and one validation ROI, not a cohort estimate.
- Standardized losses across those two runs use different scalers and are not
  directly comparable. Heavy crop overlap also limits effective biological diversity.

### Three-patient filtering diagnostic, job 47536559

| Sample | Probe candidates | Retained | Elapsed seconds |
|---|---:|---:|---:|
| M5172118 | 64 | 28 | 26.42 |
| M5173023 | 64 | 30 | 24.46 |
| M5173144 | 64 | 19 | 19.91 |

All completed. These 77 retained positions were diagnostic data, not training data.
H&E timing dominated this small probe. Visual review accepted filtering for now.

### Full filtering and recovery

Original job **47524714** hit its four-hour limit after completing two patients.
M5172118 checked 13,462 locations: 6,576 low-context rejections, 869 low-center
rejections, 6,017 eligible, and 5,000 retained (1,017 eligible not sampled).
M5173023's completed output retained 5,000 locations.

Recovery job **47563349** completed with exit 0:0 in **3:24**, peak reported RSS
943,728 K. It processed only M5173144 with H&E caching, then reused original completed
samples via symlinks and validated all three. M5173144 checked 9,215 locations:
5,872 low-context rejections, 1,002 low-center rejections, 2,341 eligible and retained.
Its whole scan reached completion in about 19.94 seconds before marker reads.
The stderr two-minute traceback was diagnostic, not a scheduler timeout.

| Completed preparation | Examples | Channels |
|---|---:|---:|
| M5172118 | 5,000 | 42 |
| M5173023 | 5,000 | 42 |
| M5173144 | 2,341 | 42 |
| Total | **12,341** | |

Assembled root:
`/scratch/groups/emmalu/hande_annotation/uni2_filter_recovered/job_47563349`.
Keep original linked directories present; the assembled root is not an independent
copy of all source arrays.

### Larger pipeline extraction, job 47563919

Both first-two-patient caches extracted and validated at 5,000 rows each. Then the
workstation `train.py` had a line-1 IndentationError, so this job failed before head
training. Elapsed 3:20; peak reported RSS 9,275,268 K. Existing feature caches remained
usable. A training-only recovery batch was added, and earlier syntax checks were
added before expensive extraction. Successful training from that recovery has not
yet been reported.

### Three-patient extraction, job 47565765

User supplied Slurm status **COMPLETED, exit 0:0, runtime 3:16**. It used one GPU,
four CPUs, 32 GB requested RAM, emmalu; actual submitted limit was 12 hours (the local
batch default is four). The wrapper validates prepared samples, CUDA, extraction,
and all output caches. Expected rows follow the prepared data: 5,000/5,000/2,341.
The exact final per-cache log lines should be archived alongside the job status.

Cache root:
`/scratch/groups/emmalu/hande_annotation/uni2_feature_checks/job_47565765`.
These are reusable features for subsequent heads; do not repeat extraction unless
inputs, encoder, or preprocessing change.

## 14. Audit trail and Slurm operation

Stages record arguments, input paths, sample outcomes, code snapshots/source hashes,
filter decisions, selected coordinates, marker order, physical scale, checkpoint
provenance, environment packages, job details, split/scaler, epochs, best checkpoint,
metrics, and prediction arrays. The earlier workstation agent added memory preflight
reporting; preserve its enhancements when merging updates.

| Stage artifact | Meaning |
|---|---|
| candidate_decisions.csv | Each probed center, tissue fractions, first decision |
| filtering.json / preview HTML | Final filter settings/counts/timings and QA examples |
| selected_coordinates.json | Exact retained set |
| manifest.json | Complete stage/sample with shard references |
| progress.json | Latest live stage; replaced atomically |
| invocation.json / source_snapshot | Settings and executed source |
| events.jsonl | Timestamped transitions and training epochs |
| environment.txt / slurm_job.txt | Remote packages and scheduler allocation |
| history.json / target_scaler.json / best.pt | Learning, training-only scaling, selected head |
| OOF reports / evaluation galleries | Patient-held-out metrics and spatial review |

A complete manifest is written only after successful sample extraction/preparation.
A root `finished` status can still contain failed samples; inspect counts. Diagnostic
outputs can have complete manifests but remain diagnostic-only. Existing paths are
refused. No automatic resume is implemented. Completed samples can be reused after
explicit validation; partial outputs stay preserved for diagnosis.

`sbatch` jobs survive terminal disconnection. `PD` means pending, `R` running;
disappearance from squeue does not establish success. Use sacct state/exit code plus
stage reports. afterok dependencies require successful preceding jobs, so the failed
original filtering job must not gate new extraction. Batch log directories must
exist before submission. Cache/token/tmp/Matplotlib configuration paths are redirected
to scratch as appropriate, avoiding a full home directory.

Approximate script requests (actual Slurm run records prevail):

| Batch | GPU | RAM | Local default time |
|---|---:|---:|---:|
| Full filtering check | None | 16 GB | 4 h |
| Bounded diagnostic | None | 16 GB | 45 min |
| Filtering recovery with H&E cache | None | 32 GB | 4 h |
| Three-patient feature extraction | 1 | 32 GB | 4 h |
| Two-patient extraction + head | 1 | 32 GB | 6 h |
| Cached-head recovery | 1 | 16 GB | 1 h |
| Evaluation | None | 16 GB | 1 h |

Requests are resource ceilings, not measured runtime guarantees. GPU allocation
alone does not accelerate NumPy filtering or TIFF I/O. Code is self-contained inside
`kidney_uni2`; no external preprocessing source directory is imported/accessed.

## 15. What is verified and what remains

Verified from reported workstation outputs: registered-file discovery, 42-channel
panel consistency, tiny paired preparation/extraction, ROI head training and metric
recovery, bounded filtering behavior, three-patient filtered target preparation,
H&E caching in recovery, and successful three-patient extraction job status.

Implemented but not yet demonstrated with supplied results: head training on the
larger three-patient filtered set, post-training evaluation galleries, full-cohort
preparation/extraction, and outer CV. Local tests currently cover 25 cases across
filtering, spatial sampling, crop caching, target arithmetic, leakage guards,
correlation handling, and sparse-map/role logic; compilation and batch syntax checks
passed. They do not prove biological accuracy or remote plotting behavior.

Not implemented/validated as completed outputs: full dense maps, H&E-only deployment
coordinate generation, cell segmentation/aggregation, compartment supervision,
calibrated confidence, independent encoder comparisons, or encoder fine-tuning.
The current cache-creation path uses paired preparation; predictions do not require
marker values as model inputs, but a separate H&E-only deployment path is still needed.

Key scientific qualifications: verify independent patient identities; assess
registration/support; account for batch/stain variation; review tissue sampling
biases; consider physical field of view; keep validation/test roles honest; and
avoid treating many correlated locations as many independent patients. Different
markers may have different morphological predictability. More data may help, but
improvement must be measured using patient-held-out results.

## 16. Immediate next actions

1. Confirm/archive final cache validation for job 47565765.
2. Record the chosen three-patient split; recommended pilot is two training patients
   and M5173144 validation, with no independent test claim.
3. Run a fresh linear head using those completed caches and corrected trainer.
4. Save losses, baseline metrics, and checkpoint; inspect measured-vs-predicted maps.
5. Expand to eligible patients with locked filtering and verified mapping.
6. Review outer folds, execute fixed-settings CV, and analyze out-of-fold predictions.

This document should be updated with new job IDs, splits, outcomes, and measured
performance as those stages run; do not replace pending statements with assumed success.
