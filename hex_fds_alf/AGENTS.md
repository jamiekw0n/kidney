# Self-contained experiment handoff

Run scripts from this folder using absolute paths. Import only local modules or
installed scientific packages. Never add parent-directory imports, sys.path hacks,
symlinks to other source folders, or access to rosie_preprocess. Cached NPZ data and
reviewed fold JSONs may live elsewhere; they are inputs, not code dependencies.
Never modify raw data or caches. All outputs must be independent new directories.
Keep UNI2-h frozen. Head training reads only caches. Dense inference may load the
frozen encoder and open registered raw images read-only, with raw_safety guards
and before/after metadata checks. Never fine-tune the encoder.

Read README.md, HEX_CROSSCHECK.md and VALIDATION.md before changing the method.
The checkpoint format is distinct from the old linear/MLP format. Always use local
model.load_checkpoint; calling parent make_head on these weights is incorrect.
FDS targets/statistics are TRAINING ONLY. Inference must work with no targets.
ALF must include rho + log(scale) + logZ(alpha), with gradients through all terms.
Do not silently replace it with an unnormalized learned robust penalty.
Run all test_*.py tests, doctor.py and compileall before workstation training.
Report synthetic versus real-data execution honestly. Local tests cannot establish
kidney performance. No remote job was submitted while creating this folder.
