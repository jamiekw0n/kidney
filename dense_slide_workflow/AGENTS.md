# Independent dense slide experiment

Read README.md. Never access rosie_preprocess. Do not edit parent experiments or raw
images. All modules are local copies; run/source freezes executable code at planning.
Targets use completed per-slide P99 factors, pixel clipping BEFORE central8x8 mean.
Patient folds stay fixed; whole-slide predictions use the outer-test checkpoint.
Never silently convert unsupported locations to zero expression. Preserve valid.npy
and show missing predictions in gray. Full native extent is displayed except <8px
bottom/right remainder; measured mIF figures are 8x8 means, NOT native cell predictions.
Training stride32 does NOT imply stride32 targets cover every pixel. Dense inference
stride8 changes evaluation support; whole-slide images are qualitative, not new OOF
metrics comparable to earlier sampled summaries. Do not tune on outer-test images.
No source imports outside this folder. Originals, prior caches, failed outputs stay
unchanged; use new paths on retries. Do not promise Monday completion before pilot
throughput, cohort sizes and Slurm queue delays are known. Test locally and state
real-image/CUDA work still needs workstation verification.
