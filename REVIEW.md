# Frozen UNI2-h kidney marker regression: first implementation

Architecture: registered 128×128 H&E context → square resize to 224×224 → frozen
UNI2-h CLS embedding (1536 dimensions) → linear or small MLP regression head →
measured kidney marker means in the central 8×8 block. This predicts the full
`panel` from prepared `measured` arrays, not ROSIE's 50-marker outputs.

This first milestone provides feature extraction, head training, held-out scoring,
and coordinate-aligned prediction shards. Cell segmentation, cell aggregation,
compartment heads, standalone map rendering, and calibrated prediction intervals
are subsequent milestones, not implemented here. Multiple heads provide only
uncalibrated disagreement. No encoder fine-tuning occurs.

## Inputs and assumptions

Use independent prepared data artifacts following `DATA_FORMAT.md`. No external
preprocessing source directory is required or accessed.
Each sample needs a complete manifest, an actual patient ID, registered H&E, and
NPZ shards containing `xy` and `measured`. All panel entries are assumed measured;
there is no missing-marker mask for this first full-panel adapter. Invalid support
blocks must already have been excluded in preprocessing. Do not encode missing
measurements as zeros. The current preprocessing rejects nonfinite intensities.

Use verified patient mappings, not sample IDs presumed to be patients. The local
single-sample example is an I/O check and cannot provide independent validation.

All caches in a training run must have identical marker order, encoder checkpoint,
transform, pixel size, context size, and target block. This deliberately rejects
mixed scales; resample registered images before preparation if needed. At
0.5073077 µm/pixel, context covers about 64.9 µm and targets about 4.06 µm.
Resizing the context does not change these physical dimensions. Field-of-view
experiments require separate, consistently prepared datasets.

An optional `--he` replaces the image path when transferring data to a cluster or
using an independently normalized H&E export. That image must preserve the exact
registered grid, orientation, and origin; dimensions alone cannot establish this.
Never fit normalization references on held-out patients.

## Cluster setup and weights

Copy only `kidney_uni2` to the workstation. All source dependencies are local
to this directory. Read `AGENTS.md` and `REMOTE_RUNBOOK.md` for the CLI handoff. Use a PyTorch environment and install the requirements there.
The requirement versions are constraints, not a validated complete environment
lock; export a lock from the successful cluster environment for reproducibility.

```bash
Can 

Request UNI2-h access using an institutional Hugging Face account and accept its
terms yourself. Authenticate with `hf auth login`, then download the official
`pytorch_model.bin` at a pinned full commit SHA. Do not put tokens in scripts.
The extractor loads that local checkpoint strictly, records its SHA256 and the
supplied revision, and never downloads weights automatically. The supplied SHA
must correspond to the downloaded file; its correspondence cannot be verified
from the weight file alone.

Architecture and preprocessing follow the official model card's manual loading
example: https://huggingface.co/MahmoodLab/UNI2-h . Square resize explicitly preserves
the whole square context. No random augmentation or additional stain normalization
is applied during extraction. Extraction is float32 initially; reduce batch size
if GPU memory is insufficient. UNI2-h extraction is expensive on CPU.

## 1. Cache each sample

```bash
python kidney_uni2/extract.py \
  --prepared /path/prepared/SAMPLE_A \
  --checkpoint /path/checkpoints/uni2-h/pytorch_model.bin \
  --revision REPLACE_WITH_FULL_40_CHARACTER_COMMIT_SHA \
  --he /path/registered/SAMPLE_A_HandE.tiff \
  --out /path/uni2_cache/SAMPLE_A \
  --batch-size 16 --device cuda
```

Repeat for every sample, including validation/test samples. Extraction does not
use targets to modify weights, so caching held-out data is allowed. Outputs are
`features`, raw `target`, and `xy` arrays in bounded per-shard files, plus metadata.
Incomplete extraction has no complete manifest and cannot be trained on. All
commands refuse existing output paths to preserve previous runs.

## 2. Define patient partitions

Create `split.json` with real patient IDs:

```json
{
  "train": ["PATIENT_A", "PATIENT_B"],
  "val": ["PATIENT_C"],
  "test": ["PATIENT_D"]
}
```

These IDs are illustrative, not a recommended cohort size. Supply caches for every
patient listed, including test patients; test arrays are not loaded during training.
Every supplied cache must belong to a listed patient. Overlap, duplicates, missing
caches, unknown IDs, and incompatible cache provenance fail before training.

## 3. Train a linear baseline and MLP

```bash
python kidney_uni2/train.py \
  --caches /path/uni2_cache/SAMPLE_A /path/uni2_cache/SAMPLE_B /path/uni2_cache/SAMPLE_C /path/uni2_cache/SAMPLE_D \
  --split /path/split.json --out /path/runs/linear_seed42 \
  --hidden 0 --seed 42 --device cuda
```

Repeat with `--hidden 256` and a new output directory for the MLP. Start with raw
intensity standardization; add `--log1p` only after confirming nonnegative,
skewed intensity distributions. Compare choices using validation patients.

Only head parameters enter AdamW. Huber loss treats standardized marker channels
equally. Target scaling is fitted on training patients only, with equal-patient
moments. Weighted sampling gives patients equal expected training contributions;
all samples from one patient contribute to that patient's sampling weight.
Validation loss is averaged within patient, then equally across patients. Early
stopping saves `best.pt`, history, and raw-scale per-marker validation metrics,
including an equal-patient training-mean baseline. Predictions are not clipped.

Training currently loads training and validation embeddings into RAM. Budget about
6 KB per crop for embeddings alone, plus concatenation and sampling overhead.
Use bounded pilot ROIs for the initial experiment. Full-slide, stride-8 extraction
can produce extremely large caches; an out-of-core loader and explicit spatial
sampling are needed before scaling to whole cohorts. This implementation does
not silently subsample patients or tissue.

## 4. Export predictions and score the held-out test set

```bash
python kidney_uni2/predict.py \
  --cache /path/uni2_cache/SAMPLE_D \
  --heads /path/runs/linear_seed42/best.pt \
  --out /path/predictions/SAMPLE_D --device cuda --evaluate
```

The export preserves `xy`, panel order, physical scale, and raw-scale predictions.
Place each vector at its central block to reconstruct marker maps; absent blocks
remain missing, not zero. `--evaluate` only permits patients in the test partition
of every supplied head. Choose the model before evaluating test patients.
Test metrics here are per sample; aggregate samples within patient before cohort
reporting. Pearson values are undefined for constant targets/predictions.

To estimate head disagreement, train the same selected architecture with several
seeds and supply multiple checkpoint paths to `--heads`. Exports additionally
contain sample standard deviation across raw-scale predictions. This is not a
probability, calibrated interval, or guarantee of accuracy. A single head emits
no uncertainty estimate.

Prediction currently consumes a cache produced from prepared pairs. An H&E-only
coordinate-generation/extraction path is a subsequent deployment task; inference
does not read cache targets unless `--evaluate` is supplied.

## Verification status

```bash
python -m unittest discover -s kidney_uni2 -v
python -m compileall -q kidney_uni2
```

Locally verified: patient leakage/provenance guards, equal-patient scaling, negative
log-input rejection, transform inversion, CLI help, and Python compilation.
PyTorch/timm and approved UNI2-h weights are unavailable locally; actual encoder
loading, GPU extraction, training, and prediction have not been run. First cluster
check: extract one small ROI, confirm `[N,1536]` finite features and aligned targets,
then train on independent patient ROIs and compare against the mean baseline.

## Workstation preflight

```bash
python kidney_uni2/doctor.py
python kidney_uni2/validate.py --datasets /path/prepared/SAMPLE_A
```

`doctor.py` exits nonzero for missing or broken dependency imports. Install a
CUDA-compatible PyTorch/torchvision pair before the remaining requirements.
`validate.py` checks data without loading PyTorch or TIFF images. Run it again
on caches with `--split /path/split.json` before training.

## Raw registered sample-folder discovery and preparation

`prepare.py` is standalone and scans immediate sample subdirectories starting with
uppercase B, F, K, M, or R. Each must have exactly one .tif/.tiff whose filename
contains case-sensitive `HandE`, and exactly one .qptiff. Extensions are matched
case-insensitively; extra files are ignored. Ambiguous pairs are reported, never
selected arbitrarily. Samples with errors do not stop other samples, and the final
exit code is nonzero if any failed. No registration is performed.

First run on the workstation (replace the script location):

```bash
python /path/to/kidney_uni2/prepare.py \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --dry-run --inspect \
  --report /scratch/groups/emmalu/hande_annotation/uni2_discovery.json
```

Discovery reads metadata only with `--inspect`. Without it, no TIFF dependency or
image decoding is required. The report shows marker labels, dimensions, axes, and
TIFF-tag pixel size where available. Missing tag scale must be confirmed externally;
matching image dimensions do not prove full common support.

Create `panel.json` as an ordered list of exact antibody/channel names after checking
QPTIFF metadata. Embedded marker identities must match this list; channels are
reordered by label, not assumed positional. Unlabeled files need explicit
`--confirm-unlabeled-panel-order` after verifying acquisition order.

For the initial pilot, use an explicitly selected sample and an inspected ROI:

```bash
python /path/to/kidney_uni2/prepare.py \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --out /scratch/groups/emmalu/hande_annotation/uni2_prepared_pilot \
  --panel /path/to/panel.json --samples-as-patients \
  --mpp 0.507 --samples REPLACE_WITH_SAMPLE_ID \
  --roi X0 Y0 X1 Y1 \
  --valid-mask-name common_validity_mask.tiff
```

Replace the approximate scale with the verified scale if needed; TIFF metadata
mismatches beyond 1% fail. ROI coordinates must be multiples of 8 and inside the
selected pair. `--samples-as-patients` records the user's one-sample-per-patient
assumption. Alternatively supply `--patient-map` CSV for repeated patient samples.

If full support was independently verified, `--assume-all-valid` can replace
`--valid-mask-name`. Registration alone does not establish this. An interior ROI
also requires checking its common support. Preparation averages raw marker intensity
in fully valid 8x8 blocks; no marker normalization or background subtraction is
performed. All spatial blocks are retained by default; optional `--white-threshold`
is only a basic RGB heuristic and needs review for tissue retention.

Preparation reads 64x1024-pixel slabs and writes bounded NPZ shards. Run the normal
`validate.py` check afterward, then `extract.py` with the approved local checkpoint.
The manifest records physical scale, ROI, patient assumption, channel mapping, and
support choice. Existing sample outputs are preserved and reported as errors.

## Visual ROI review before preparation

`view_rois.py` needs only the existing image-reading dependencies and Pillow, not
PyTorch, a GPU, a display server, or model weights. It obeys the same pair discovery
rules as preparation and requires embedded QPTIFF marker labels. Run:

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/view_rois.py \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --samples M5172118 M5173023 M5173144 \
  --out /scratch/groups/emmalu/hande_annotation/uni2_roi_review \
  --markers DAPI CDH1 UMOD COLIV \
  --size 512 --grid 7 --top-k 6
```

The script probes 49 bounded H&E windows per sample, ranks tissue-like windows,
avoids overlapping suggestions, and exports up to six matched ROI panels per
sample. This is a sparse heuristic search, not exhaustive tissue segmentation.
Try a larger grid or lower `--min-tissue` if no regions pass. Whole slides are never
loaded as arrays; each mIF read includes all marker channels only in the chosen ROI.

Copy the entire output directory to a laptop, then open `index.html` in a browser.
PNG images are linked locally; no server, network access, or CDN is required.
Each card gives `--samples` and `--roi` arguments to copy after review. `rois.json`
records suggestions and display scaling, with review/support flags initially false.
There is no automatic approval or save-selection feature in the gallery; record
verified choices in a separate file or update those fields explicitly.

To inspect the already extracted pilot region, use a new output directory:

```bash
python /scratch/groups/emmalu/hande_annotation/kidney_uni2/view_rois.py \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --samples M5172118 \
  --out /scratch/groups/emmalu/hande_annotation/uni2_existing_roi_review \
  --roi 4096 3072 4608 3584
```

Panels show H&E and independently percentile-scaled markers at identical native
coordinates. The DAPI cyan overlay slider aids visual alignment inspection.
Brightness across different ROIs is not quantitatively comparable because display
limits differ; raw data are unchanged. Constant channels display black. Absence of
signal can be biological and does not alone prove missing coverage. Reject windows
with padding, tears, uncovered regions, or mismatch between corresponding nuclei
and structures. Candidate tissue fraction does not establish validity or alignment.

Locally tested: candidate bounds/grid alignment, black/white heuristic exclusions,
display scaling and input preservation. Real TIFF rendering needs workstation
verification. No additional dependency installation is required beyond requirements.

## Recover metric export after a completed training run

Undefined Pearson correlation (e.g. constant baseline predictions) is JSON `null`.
`metrics.py` computes centered correlations in float64 and checks constant vectors
explicitly, avoiding float32 std/correlation inconsistencies. Nonfinite raw inputs
still fail rather than being silently hidden.

If training saved `best.pt` but failed while writing reports, run `recover_report.py`
with `--checkpoint`, `--caches`, and a NEW `--out` directory. It loads the selected
head, reconstructs the raw-scale mean baseline, regenerates validation metrics,
and writes `scale_diagnostics.json` ranked by each marker's standardized validation
loss. It does not train or modify the original checkpoint. Diagnostics report raw
ranges and training standard deviations; a near-zero training variance combined
with validation variation can magnify standardized error. Do not modify scaling
using held-out patients to reduce that error.

Remote code may contain workstation-agent enhancements absent from this checkout.
For recovery, copy only `metrics.py` and `recover_report.py`; these reuse existing
`train.py` helpers but import the corrected metrics directly. Merge future metric
changes into the workstation trainer rather than overwriting its enhancements.

## Cohort filtering and cross-validation

See `COHORT_WORKFLOW.md` for executable commands and the complete audit trail.
`cohort_prepare.py` adds H&E context/center filtering and seeded spatial sampling;
`cohort_extract.py` records batch extraction; `cross_validate.py` performs outer
patient evaluation with a separate inner early-stopping holdout and saves OOF
predictions/metrics. `audit.py` records source snapshots, arguments, and events.
