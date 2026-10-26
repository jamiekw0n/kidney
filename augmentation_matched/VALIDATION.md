# Validation record

The standalone suite covers:
- All8 D4 transformations preserve the exact central8x8 channel sums.
- Deterministic per-location parameters, realistic configured bounds, source RGB
  arrays unchanged; geometry and geometry+color share the same D4 orientation.
- Augmented shard coordinates align with original rows, including shuffled/repeated
  sampled indices and masked identity draws in the mmap store.
- Cache checksums/completed-shard resume and original-cache byte preservation.
- REAL CPU optimization/checkpoint/original-view outer export for all3heads on one
  reviewed synthetic fold;51epochs each. Same initialization/scalers/weights, same
  location sampler draws; augmentation views used only by training.
- Synthetic4fold/3variant aggregate + paired MAE differences/bootstrap arithmetic.
- Inherited expression-weighting/FDS/architecture tests. FDS remains off in this
  experiment; helper tests cover the copied model implementation, not a new FDS arm.

Images and encoder used in cache tests are synthetic; encoder forward is mocked.
Real UNI2-h GPU/torchvision performance, actual stain appearance, CUDA memory,
shared-filesystem throughput and complete39patient jobs remain unverified locally.
No remote job or real patient augmentation was run by the coding agent.
CPU trainer subprocess tests require outside-sandbox execution for macOS OpenMP.

No claim of model improvement follows from numerical tests. Inspect pilot preview
and actual cached feature logs, then compare held-out outcomes and inner validation.
Head budgets match per epoch and stopping criteria; actual early-stopped epoch
counts may differ. This is explicitly reported by the audit.

Final full suite:24tests passed in16.128seconds. Includes corrupt-view rejection and
mocked two-GPU queue/no-Slurm test. A follow-up context-manager fix closes the plan
lock cleanly on return; targeted launcher test rerun afterward. Python compilation
and CLI help checked. No actual GPU throughput or biological accuracy measured.

After adding identity-feature reproduction and ordered CPU prefetch, the full24-test
suite passed again in8.625seconds. Tests use consistent synthetic identity features
and a mocked encoder, not real UNI2 weights. Source/CLI Python compilation passed.
