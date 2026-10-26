# Annotation-linked UNI2-h embedding QA

## Purpose and correct analysis unit

The existing cache has one 1536-dimensional embedding per 128×128 H&E context,
not per cell. This analysis reuses those cached embeddings. A point in UMAP is a
**tissue context** colored by annotations of cell centroids inside that context.
It is not an embedding of an individual annotated cell and does not automatically
verify encoder accuracy. A genuine cell-centered study would require a new encoder
pass at cell centroids, with context size and label interpretation specified.

Cell labels are reference annotations, not error-free independent ground truth.
The labels derive from marker/morphology rules upstream; their uncertainty and
biological validity need separate review. The analysis does not add supervision to
UNI2-h or refit the regression head.

## Annotation inputs supplied by user

Remote root: `/scratch/groups/emmalu/hande_annotation/PRMNT_SP_annotated_cells_39samples`.
There is one sample directory per sample and one combined CSV inside each sample
directory. `annotation_files` requires exactly one CSV per selected sample; it will
not silently choose between multiple candidate files. An alternative `--annotations`
argument accepts explicit nonoverlapping combined/per-sample files.

Required columns: `sample_id`, `cell_id`, `x`, `y` and each requested label column.
Defaults: `cell_type_merged`, `cell_type_new`, `compartment_refined`,
`cell_state_primary`. The supplied schema has all four. Other columns can be selected
with `--label-columns`, including refined marker labels or cell-cycle state, as
categorical colors. Annotation confidence is not automatically thresholded: values
such as high/medium/soft are categorical and must not be treated as numeric scores.

## First resolve coordinate units and source grid

A coordinate written as `5194.0` does **not** imply micrometers. Pixel centroids can
be stored as floating-point numbers. Confirm units and the registered image's
origin, orientation, scale and any crop/registration transform from the exporter.
Multiplying/dividing by pixel size cannot fix unknown offsets, flips or rotations.

`annotation_coordinates.py` shows the same cells under two hypotheses: x/y already
in native pixels, or x/y in micrometers converted using the cache's µm/pixel. It
reports coordinate quantiles and out-of-bounds counts and plots H&E and native DAPI
crops with centroid crosses. A nucleus need not coincide precisely with a
whole-cell centroid. Both hypotheses can appear superficially plausible; bounds
and a few overlays alone are not proof. Inspect segmentation overlays/exporter
metadata if needed. The script does not choose units automatically.

Copy the updated standalone source, then submit the coordinate diagnostic (no GPU):

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slurm/check_annotation_coordinates.sbatch K2110291
```

Output: `uni2_annotation_coordinate_checks/job_JOBID/index.html` and
`coordinate_report.json`. Logs: `uni2_slurm_logs/annotation-xy-JOBID.out/.err`.
This requires valid CSV headers. It reads raw H&E/DAPI read-only and uses a bounded
8 GiB optional H&E cache for single-chunk files. Source metadata is checked before/
after. No model inference or image rewriting occurs. The sample CSV and paired image
must be on the same registered grid under the confirmed unit interpretation.

### Uploaded example has a formatting issue

The local example `/Users/jamiekwon/Downloads/K2110291.csv` begins with a data row.
Its header appears inside cell 2's record and splits that record across lines.
The original input is untouched. `ANNOTATION_EXAMPLE_QC.json` reports only the
195,315 intact rows, excluding the malformed record fragments; it is **not** a
cleaned replacement dataset or an authoritative full cell count.

Intact-row observations: 136 columns; x=66..15028, y=562..17905; all x/y values are
integer-valued floats; no explicit coordinate-unit metadata; no duplicate sample/
cell IDs among intact records. Median whole-cell `size`=502 and median available
nuclear `size_nuclear`=144. These are compatible with pixel-based segmentation
exports, but units are not established by this observation. The uploaded copy may
have been damaged during editing/copying; do not assume every remote CSV is broken.
Check the original first record; if it is also malformed, recover a clean original
or make an explicitly audited new derived copy. Do not silently skip bad rows for
UMAP. Current analysis scripts reject malformed/missing headers or row lengths.

## Annotation-to-context matching

- Select up to 1,000 existing cached locations per sample without replacement,
  seed 42 plus sorted sample index. Record source manifest SHA, selected cache row
  indices and sample inventory. This bounds initial RAM/UMAP cost; it does not
  modify the training cache or select by cell label. Maximum 39,000 UMAP points.
- Match cells by **sample ID first**, then centroid containment in the half-open
  window `[x−64,x+64) × [y−64,y+64)` for each selected context. Never use an unlimited
  nearest-cell match that assigns empty tissue to a distant cell.
- Default `--join-region context` matches the encoder's full input support. Optional
  `target` uses `[x−4,x+4) × [y−4,y+4)`; sparse 8×8 targets may contain few centroids.
  Neither mode uses cell masks or proves which cell covers the center point.
- Keep cell counts and the full categorical composition for each label field.
  Assign a dominant display label only if at least **3 labeled centroids** and
  **60% dominance**, with no tie. Otherwise use `MIXED_OR_LOW_SUPPORT` or
  `UNANNOTATED`. These thresholds are adjustable QA heuristics, not biological truth.
- Purity denominator is the number of nonblank annotations for that field, not
  tissue area or all cells when labels are missing. Counts are cell-number weighted,
  not cell-size/area weighted. Missing label counts are reported.
- Keep every selected embedding in UMAP, including mixed and unannotated contexts;
  do not manufacture visual separation by keeping only pure examples.
- Coordinates must be finite and in bounds after declared unit conversion; failures
  stop explicitly. Use the confirmation flag only after verifying registration.
- Duplicate `(sample_id,cell_id)` records fail. A disk SQLite uniqueness index avoids
  keeping all cell IDs in RAM. Input CSVs stream in chunks of 10,000 rows. No marker
  intensity columns are used to fit the dimensionality reduction.

## UMAP procedure and plots

1. Require finite, nonzero cached feature vectors with consistent UNI2-h signature.
2. L2-normalize each feature vector; no marker-target scaling is applied to features.
3. Default PCA: up to 50 components, randomized solver, no whitening, seed 42.
   Record explained variance and transform. `--pca-components 0` bypasses PCA.
4. Fit **unsupervised** 2D UMAP with Euclidean distance, neighbors 15, min_dist 0.1,
   seed 42 and one UMAP worker. Euclidean distance on L2-normalized full vectors is
   related to cosine distance; PCA truncation approximates that geometry and can
   discard useful variation. Record the retained variance and assess sensitivity
   before drawing conclusions.
5. Fit once and color the identical coordinates by sample ID, patient ID, the
   requested cell-type/state/compartment summaries, centroid count, feature norm,
   dominant fractions and measured CAV/CD8 central-block targets. Labels, target
   measurements and native coordinates never enter the PCA or UMAP fit.

UMAP parameters and seed behavior follow the
[official parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
and [reproducibility documentation](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).
A fixed seed trades some parallel speed for repeatability; versions are recorded.
A UMAP plot is a projection, not an objective cluster count or classifier. No
HDBSCAN/K-means labels or cell-type accuracy claims are generated here.

## Dependencies and launch after coordinates are verified

Coordinate QA needs the existing image-reading/plotting environment. Full UMAP
additionally needs the optional `requirements-embedding.txt`. Check imports first:

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python -c 'import scipy, sklearn, umap, matplotlib; print("UMAP imports passed")'
```

If missing, install into the analysis environment deliberately and retain a package
export before/after; no installation is performed by batch jobs:

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python -m pip freeze > /scratch/groups/emmalu/hande_annotation/uni2_before_umap_packages.txt
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python -m pip install -r /scratch/groups/emmalu/hande_annotation/kidney_uni2/requirements-embedding.txt
```

After establishing units/grid, start with one sample. **Choose only the verified
unit argument**, not based on floating-point formatting:

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slurm/embedding_umap.sbatch native-pixels --confirm-registered-coordinates --samples K2110291
```

If units are micrometers, replace `native-pixels` with `micrometers`. Omit `--samples`
for all available cache samples with matching annotation folders. The script refuses
missing samples/ambiguous CSVs. Optional `--join-only` produces annotation QA without
PCA/UMAP and without requiring sklearn/umap. The flag confirming the grid is never
inserted by the wrapper; users must explicitly provide it after verification.

CPU allocation: four cores, 32 GB, two hours; no GPU. Timing depends on annotation
CSV size, context counts and package JIT/PCA/UMAP overhead; start bounded, inspect
progress, then scale. No existing training or dense-inference job is altered.

## Outputs and reading the result

`uni2_embedding_umap/job_JOBID/` contains:

- `inventory.json`: exact selected cache inventory and counts.
- `annotation_join_report.json`: matching rates, missing labels, category counts,
  ignored annotation samples, feature norms and constant dimensions.
- `joined_tiles.json`: source coordinates/cache rows, centroid counts, annotation
  compositions, dominant labels/purity and UMAP coordinates when completed.
- `umap_points.npz`, PCA transform, `reduction_settings.json`: reusable coordinates,
  transform/settings and software versions. The fitted UMAP object is not pickled.
- HTML/PNG plots, invocation/source snapshot, events and completion manifest.
- `annotation_ids.sqlite3`: uniqueness-audit index; a successful join-only run does
  not produce a completed UMAP manifest.

Look for biological consistency **and** sample separation. A cluster enriched for
a cell type is useful descriptive evidence; it may also reflect cell density,
tissue composition or acquisition effects. Mixed contexts can bridge clusters
without indicating a bad encoder. Separation by sample is not automatically a
batch artifact because composition can genuinely differ. No separation does not
prove the full 1536-dimensional representation lacks predictive signal.

This study is exploratory on already-viewed cohort data and cannot validate a new
cell-type predictor or rescue CD8 accuracy by itself. Useful next steps after join
QA are sensitivity checks and, if justified, patient-held-out quantitative probes
or a new cell-centered embedding extraction. Do not train probes and report their
training-set accuracy as generalization.

## Verification status

Local tests cover spatial boundary inclusion/exclusion, unit conversion, duplicate
IDs, mixed-label handling, deterministic cache sampling and annotation-directory
ambiguity. A synthetic directory-to-cache annotation join completed. The local
environment lacks sklearn/umap, so actual PCA/UMAP fitting has **not** been verified
locally. No real workstation annotation overlay or UMAP result has been supplied.
