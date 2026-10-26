# Fixed color scale for existing dense ROI caches

New standalone renderer; existing code and outputs are unchanged. Reads only saved
NPZ, JSON and H&E PNG artifacts. No raw image access, model inference, retraining,
or metric recomputation. Dependencies: NumPy, Pillow, matplotlib. No GPU.

Default: intensity /255, vmin=0, vmax=1, viridis for EVERY marker, model, and ROI.
There is no per-marker percentile contrast adjustment. Yellow has the same numeric
meaning everywhere. Faint markers will appear faint. This improves numeric display
comparability; it does NOT make antibody-channel fluorescence biologically equivalent.
H&E retains RGB. Negative/>255 predictions saturate at the color limits only;
display_statistics.json records outside-limit fractions. Saved predictions unchanged.
This is display rescaling, NOT the new per-slide P99 target normalization. This
renderer is for existing raw-intensity dense caches, not P99-trained checkpoints.

Copy this complete folder to kidney_uni2/dense_fixed_scale on the workstation.

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/dense_fixed_scale/render.sbatch
```

Discovers completed direct-child dense caches in uni2_dense_rois, including original
linear/MLP and cached experiment pairs. Defaults: CAV CD8 CD138 COLIV POSTN VIM.
Partial/unsupported discovered caches are listed in skipped.json; explicit inputs
fail on invalid references. Only cached native-marker images are available. For
other markers, show measured block means and predictions without the native panel.
No source images are reread. All measured block targets/predictions are required
finite, shape-matched and on the expected native coordinate grid; native maps when
present are checked against cached target means. Incomplete output has no final
complete manifest. Existing output paths are refused. Scan is one directory deep.

Outputs: uni2_dense_fixed_scale/job_JOBID/index.html plus one subgallery per source.
Logs: uni2_slurm_logs/fixed-scale-JOBID.out/.err. Final manifest and skipped.json
identify rendered and omitted sources. No real workstation figures generated locally.

To render all 42 markers, pass their exact names with --markers. For a different
shared range, use e.g. --vmax 0.1 (yellow=0.1 for ALL figures). Keep 0..1 as the main
comparison; label any supplementary 0..0.1 figures explicitly.

To restrict inputs, use Python directly with --dense-roots PATH [PATH ...] instead
of --scan-root, and --out NEW_PATH. The Slurm wrapper always scans; Python allows
explicit input selection. No source folder imports required.

Local verification: synthetic cached-grid render test (including native-marker
agreement, shared limits, missing native panel, negatives/above-range values and
source-byte preservation); CLI help, compileall and bash syntax. These checks do
not verify real workstation artifacts or biological predictions.
