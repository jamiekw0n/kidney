# Cross-check against released HEX and the adaptive-loss reference

Reviewed on 2026-10-20 using the public source below. Branch URLs are mutable;
a repository commit SHA could not be resolved through the available source-access
route, so **no immutable upstream revision or bit-for-bit equivalence is claimed**.
The local runnable implementation is independently written and records its own
source hashes per experiment. Upstream code is not imported, cloned or downloaded
at runtime. These references are audit evidence, not execution dependencies.

## Authoritative sources

- [HEX model and FDS](https://github.com/lilab-stanford/HEX/blob/main/hex/utils.py):
  CustomModel, calibrate_mean_var, FDS and update_running_stats_from_moments.
- [HEX training](https://github.com/lilab-stanford/HEX/blob/main/hex/train_dist_codex_lung_marker.py):
  FDS_ACTIVE_MARKERS, FDS_OFF_EPOCH, optimizer, criterion and epoch loop.
- [Barron adaptive parameterization](https://github.com/jonbarron/robust_loss_pytorch/blob/master/robust_loss_pytorch/adaptive.py):
  AdaptiveLossFunction initialization and lossfun.
- [Barron parameter transforms](https://github.com/jonbarron/robust_loss_pytorch/blob/master/robust_loss_pytorch/util.py):
  affine_sigmoid, inv_affine_sigmoid and affine_softplus.
- [Normalized likelihood](https://github.com/jonbarron/robust_loss_pytorch/blob/master/robust_loss_pytorch/distribution.py):
  Distribution.log_base_partition_function and nllfun.
- [Underlying FDS research](https://proceedings.mlr.press/v139/yang21m.html),
  Yang et al., ICML 2021.

## Mapping from reference to local implementation

| Component | Reference behavior | Local decision |
|---|---|---|
| Hidden head | 256/128, ReLU, dropout .5 | Same; input becomes 1536 and output follows kidney panel |
| Marker conditioning | Separate FDS per output row | Same mathematical mapping, stored in batched buffers |
| Intensity bins | 50 equal-width bins on [0,1] | Same, strict range validation |
| Moment tracking | Detached hidden features, per-epoch moments, EMA .9 | Same structure; float64 accumulation |
| Kernel | Gaussian delta filter, size9, sigma2 | Coefficients independently matched to SciPy in tests |
| Calibration | Diagonal variance ratio clipped [.1,10] | Same supported-bin formula, zero-variance bypass |
| Scheduling | Start smoothing10, off50 | Same zero-based schedule |
| Evaluation | No target-conditioned smoothing | Same; label-free API and state-invariance tests |
| Loss | Per-output adaptive robust NLL | Same density/objective, different numerical normalizer |
| Alpha bounds | .001 through 1.999, initialized1 | Same affine-sigmoid transform |
| Scale | Floor1e-5, initialized1 | Same shifted affine-softplus transform |
| Loss optimization | Criterion parameters included in optimizer | Same; checkpoint also saves criterion state |

The relevant parameter transform is precisely:

```
scale = floor + (initial_scale-floor) * softplus(raw_scale + log(expm1(1)))
```

A rescaled softplus divided by log(2) would initialize to the same scale but have
a different derivative/optimization path. The final implementation uses the
reference's shifted transform, with an explicit independent numerical test.

## Explicit adaptations and safeguards

1. **Frozen UNI2-h caches replace MUSK image training.** No encoder layers are
   fine-tuned and no image augmentation runs. This respects the current project
   design but differs from HEX end-to-end learning. Shared context/central-target
   limitations remain. No reported HEX performance is inferred from this adaptation.
2. **Training labels use fixed raw/255 scaling.** No target z score is inserted
   into FDS binning. The old kidney z-scored MLP is a historical comparator, not a
   matched loss ablation. All local modes share the new scaling and architecture.
3. **Empty bins are not assigned arbitrary placeholder feature distributions.**
   Kernel averaging excludes unsupported bins and renormalizes over valid support.
   Calibration of unsupported bins bypasses. A minimum of two within-epoch samples
   is required; singleton bins do not create zero-variance pseudo-evidence.
4. **A newly observed bin initializes from its own moments**, even if it first
   appears after epoch0. Otherwise applying EMA to default zeros can bias estimates.
5. **Finite-precision safeguards:** float64 moment accumulation; cancellation-induced
   negative variances floored at zero; near-zero variance dimensions bypass; finite
   targets, predictions, objectives and gradients checked. Ratios retain [.1,10].
6. **ALF normalization uses quadrature rather than the external spline.** The
   likelihood, learnable bounds and transforms are retained. CPU-only independent
   adaptive integration and gradients validate the approximation. No official spline
   asset or package is vendored, and no full numerical equivalence claim is made.
7. **Single-GPU cached-head optimization** defaults to Adam lr1e-3 and gamma.95;
   HEX's image-training lr is1e-5. Both loss and head groups have zero weight decay.
   There is no distributed reduction or AMP; only head matmuls are float32.
8. **Sampling and evaluation are patient-aware.** Default inverse-patient-row
   replacement sampling follows our earlier protocol rather than HEX's distributed
   shuffled image batches. Unique row and patient bin support is recorded separately
   from sampled moment counts. Outer partitions reuse the reviewed kidney folds.
9. **Early stopping is an addition.** Fixed equal-patient validation MSE selects a
   checkpoint only after epoch50 has switched FDS off. The released script instead
   periodically saves checkpoints. Training NLL cannot serve as a fixed comparable
   score while its own shape/scale parameters change.
10. **Fresh checkpoints for every fold and mode.** No ALF or FDS state crosses
    patient folds; checkpoint loader supports only this format and inference bypasses
    FDS. Existing head loaders must not be used on the new architecture.

## What the cross-check does and does not establish

It establishes matching architecture/conditioning intent, schedules, robust-density
objective and parameter transforms, plus tests of the independently implemented
math. It does not establish identical random minibatches, distributed numerics,
pretraining, biological data, image scale, registration, normalization of public
training CSVs, or paper benchmark results. Empty-bin safeguards intentionally
prevent claiming exact FDS source reproduction. GPU and real kidney evaluation
still need to run on the workstation.
