# Remote workstation execution checklist

1. Copy only `kidney_uni2/` plus selected data artifacts to the workstation. Do not
   copy `__pycache__`. Run from any working directory using absolute script paths.
2. Read `AGENTS.md` and `DATA_FORMAT.md`. Identify actual prepared-data locations,
   registered H&E paths, independent patient mapping, local approved checkpoint,
   full checkpoint commit SHA, GPU model, available RAM, and free disk space.
3. Create a dedicated Python environment. Install a compatible PyTorch/torchvision
   pair for the installed NVIDIA driver first. Install `requirements.txt` afterward.
   Run `doctor.py`; inspect its JSON and exit status. CUDA absence is permitted only
   if choosing `--device cpu` explicitly. Save `pip freeze` and doctor output.
4. Run unit tests and compilation. Run `validate.py --datasets /data/sample_A ...`
   on prepared datasets. Metadata validity does not prove registration: independently
   inspect H&E/mIF alignment and patient identities before training.
5. Extract one bounded ROI using the README command. Check finite `[N,1536]`
   embeddings, unchanged coordinates/targets, checkpoint SHA256, input scale, and
   marker order. Start at batch size 1 or 4 if GPU memory is uncertain, then increase.
6. Extract all bounded pilot samples. Build a split JSON using verified patient IDs.
   Run `validate.py --datasets /cache/sample_A ... --split /data/split.json`.
7. Train linear and MLP heads to separate output directories. Compare validation
   per-marker metrics against the training-mean baseline. Review curves for
   overfitting and inspect biological localization, not just aggregate errors.
8. Freeze model selection, then export held-out predictions with `--evaluate`.
   Save commands, logs, split, software lock, checkpoint metadata, and metrics.

No automatic downloads, remote job submission, resume, or overwrite occurs. Cache
and prediction writes are per shard; `manifest.json` is written only after successful
completion. A directory missing that manifest is incomplete. Use a new path for
reruns. If memory is insufficient, stop and design an out-of-core dataset rather
than silently reducing the patient cohort or sampling arbitrary patches.

First required runtime proof: a real frozen UNI2-h checkpoint loads strictly,
features are finite, head-only training completes on independent patient ROIs,
validation metrics are finite, and prediction coordinates match their source.
Synthetic tests alone cannot establish biological accuracy.
