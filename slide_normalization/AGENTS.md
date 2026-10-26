# Self-contained per-slide-normalized target experiment

Read README.md and VALIDATION.md. No parent imports, source symlinks or outside
scripts. Raw images and old caches are READ ONLY. Output guards, source metadata/
cache hashes and final completion manifests are mandatory. Never rewrite, retile,
chmod or normalize original TIFFs in place. No encoder training or re-extraction.

Target = mean8x8(clip(pixel / EACH SLIDE's channel divisor,0,1)). P99 is exact over
ALL native pixels; zeroP99 falls back to dtype maximum with a flag. Do not silently
change support masks, percentile, clipping order, sparse policy or channel order.
Raw cached means must match freshly read original blocks before accepting targets.

Regression z scores fit only training patients AFTER normalization; FDS binning
uses normalized block target /1. No extra /255. Per-slide mIF-derived factors define
relative labels, not model input or deployable raw-intensity calibration. Report
relative units and do not claim raw-scale MAE improvements. Reject mixed recipes/
units in partitions and checkpoint provenance. Use the local new-format loader.
Run all tests, compileall, shell checks and workstation pixel-reading checks. Local
synthetic CPU checks are not proof of real TIFF/CUDA or biological performance.
