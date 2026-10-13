# Portable data contract

Prepared inputs are independent data artifacts. They may already exist; no access
to another preprocessing code directory is required. This package does not register
images or generate paired marker targets from raw mIF. A remote agent building a
new data adapter must follow this contract and validate biological alignment.

Each sample directory contains `manifest.json` and one or more NPZ shards.
Required manifest fields:

```json
{
  "complete": true,
  "sample_id": "sample_A",
  "patient_id": "verified_patient_A",
  "he": "/workstation/data/registered_he.tiff",
  "he_axes": "YXS",
  "he_series": 0,
  "he_level": 0,
  "he_mpp": 0.5,
  "shape": [1000, 2000],
  "input_patch": 128,
  "target_block": 8,
  "panel": ["ECad", "Vimentin"],
  "shards": ["targets_00000.npz"]
}
```

Values here are illustrative. `shape` is `[height,width]`; `xy` columns are `[x,y]`.
Axes describe the selected TIFF level, and singleton non-spatial dimensions are
accepted. Specify `CYX` explicitly for multi-page channel TIFFs only after checking
channel identity. H&E must be uint8 RGB. The TIFF reader reads bounded regions via
Zarr; no full-slide image array is loaded.

Each prepared shard contains:

- `xy`: integer `[N,2]` centers of central target blocks, normally `(4+8k,4+8j)`.
- `measured`: finite numeric `[N,M]` raw mean intensity for each marker in panel order,
  averaged over the central 8×8 block on the SAME grid as H&E.

The H&E context is centered at `xy` and covers 128×128 native pixels, with zero
padding at image boundaries. Retain only target blocks fully supported by both
registered modalities. Missing measurements are not zeros; the initial full-panel
implementation requires all channels measured. Verify source channel labels before
reordering into `panel`. Do not derive validity from marker positivity.

Caches preserve this metadata and replace `measured` with `target`, adding
`features` float32 `[N,1536]`, `encoder`, `revision`, `checkpoint_sha256`, `transform`,
and software versions. The H&E path can be remapped with extractor `--he` without
changing target coordinates. Sample IDs must be unique across supplied caches.

Prediction shards contain `xy` and `prediction` `[N,M]` in original intensity units;
multiple heads additionally produce `head_disagreement_std`. These are sparse
block maps: uncovered locations remain missing. Output maps are not RGB renderings,
cell-level measurements, or calibrated confidence probabilities.
