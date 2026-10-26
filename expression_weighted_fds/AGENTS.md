# Standalone matched experiment handoff

Only local source imports and installed packages. No parent sys.path manipulation,
symlinks, outside scripts or encoder fine-tuning. Training reads only caches.
Optional dense inference can load the frozen encoder and read raw images only
with local raw_safety checks and read-only TIFF readers.
Existing NPZ caches and reviewed fold JSON are read-only data inputs. Outputs must
be new and independent. Read README.md and VALIDATION.md before edits/runs.

Keep weighted and weighted_fds identical except the FDS treatment. Regression uses
training-only equal-patient z scores; FDS binning uses raw/intensity_max. Never
confuse those arrays. Loss weights stay training-fitted and are not multiplied
into FDS moments. No ALF or optimizer scheduler belongs in this experiment.
Keep warm-up/checkpoint eligibility matched in both modes. Dropout and FDS must
be disabled at inference, which must require no ground-truth labels.
Use local model.load_checkpoint for the unique format. Run all tests, doctor,
compileall and shell syntax checks. Actual CUDA/cohort evidence must be reported
separately from synthetic CPU checks. No automatic overwrite or resume.
