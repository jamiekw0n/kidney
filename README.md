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

For registered raw pairs, `discover.py --base /absolute/data/path` inventories
immediate sample folders beginning with uppercase B, F, K, M, or R. It requires
one `.tif`/`.tiff` filename containing case-sensitive `HandE` and one `.qptiff`,
and reports ambiguous or missing pairs without selecting a candidate. Use
`--out /new/inventory.json` to save the inventory without overwriting a file.
This inventory is not a prepared dataset: patient mapping, TIFF channel layout,
valid support, and bounded pilot ROIs are still needed to build target shards.
`panel.json` records the user-supplied channel order and pixel size; those values
have not yet been independently checked against image metadata.

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
python -m pip install -r kidney_uni2/requirements.txt
```

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
