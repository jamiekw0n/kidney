# CLI agent handoff

This directory is the entire UNI2-h implementation. Work only here for this task.
Do not access or import `rosie_preprocess` or its numbered copies. Shared functionality
has been replicated locally in `image_io.py`. Inputs are data directories conforming
to `DATA_FORMAT.md`; no source checkout outside this directory is needed.

Read `README.md`, `DATA_FORMAT.md`, and `REMOTE_RUNBOOK.md` before running experiments.
Commands are scripts, run as `python /absolute/path/kidney_uni2/SCRIPT.py`; module-mode
execution is not supported. CLI help works without PyTorch. Never invent patient IDs,
physical scales, panel identities, registration validity, or checkpoint revisions.

Start with `doctor.py` and `validate.py`. Install a CUDA-compatible PyTorch build
for the workstation using official PyTorch instructions; do not silently replace an
existing environment. Use a dedicated environment. Obtain approved weights through
the user's institutional Hugging Face access. Never print or save authentication tokens.

The encoder stays frozen. Do not add encoder fine-tuning to this baseline. Only
training patients fit target transformations. Every sample from a patient stays in
one partition. Do not score test data until architecture/settings are selected.

Begin with bounded pilot data: training loads cached embeddings into RAM. Whole-slide
stride-8 caches may exceed memory/storage. Record GPU and RAM/storage availability,
row counts, and estimated memory before training. No automatic ROI truncation.
Do not overwrite outputs: interrupted runs must use a new output directory.

Verification: `python -m unittest discover -s /absolute/path/kidney_uni2 -v` and
`python -m compileall -q /absolute/path/kidney_uni2`. Run actual ROI extraction,
training, and export smoke checks on the workstation; these have not run locally.
Report which tests ran, exact input/checkpoint provenance, and any unverified behavior.
No cell/compartment supervision or calibrated confidence is implemented yet.

For cohort tissue filtering and patient-level cross-validation, use
`COHORT_WORKFLOW.md` as the authoritative stage runbook. Record all exclusions and
review retained/rejected previews before full extraction. Default 5-fold outer CV
uses an inner holdout for early stopping; it is not a hyperparameter-tuning nested
CV implementation. Keep head settings fixed for its outer evaluation.
