# Self-contained augmentation-only experiment

Read README.md and VALIDATION.md. No imports/source scripts outside this folder.
Installed packages and read-only data/checkpoint artifacts are allowed. Never access
rosie_preprocess, modify raw images, edit old code/snapshots, delete/replace existing
embeddings or overwrite existing model results. No Slurm needed. The code-server
allocation must stay active; each worker owns one logical cuda device.

Identity views are existing cached features. Generate only two new feature views:
nonidentity center-preserving D4, and SAME D4 plus mild brightness/contrast/saturation/
hue. Fixed native128 context/8 target. Do not translate/zoom/crop or alter targets,
normalization, splits, architecture, loss or FDS in this matched experiment. FDS OFF.
Target transforms/weights use unique original training locations only. Augmented
views may enter ONLY training. Validation/outer test must use original features.
50/50 identity/augmented sampling uses independent NumPy RNG; sampler Torch RNG and
initialization are matched across variants. No triple-sized epochs. Early stopping
is identical but can produce different total epoch counts; record them honestly.

Immutable plans freeze code and provenance. Resume only hash-verified complete
features/committed shards. Preserve any uncommitted orphan embedding files; new
attempt names avoid collisions. Partial training outputs require review; never
silently delete them. Paired patient-bootstrap intervals condition on fixed models,
not full training/CV/model-selection uncertainty. Outer test images are not a tuning
set. Local synthetic CPU checks do not establish real UNI2-h GPU performance.
