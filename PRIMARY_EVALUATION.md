# Primary patient-level quantitative evaluation

Primary assessment is accuracy/agreement of marker estimates on unseen patients,
not the number of markers beating a constant baseline. Baselines remain secondary
reference outputs. No positivity, compartment, or cell-level metrics are introduced
without appropriate labels. No prediction confidence model is introduced here.

## Metrics

Each marker is scored within each patient using all that patient's sample locations:

- MAE: mean absolute intensity error.
- RMSE: square root of mean squared intensity error; emphasizes large mistakes.
- Pearson: linear correlation of measured and predicted intensity.
- Spearman: Pearson of average ranks, with ties handled explicitly.
- Mean bias: mean(predicted − measured); positive means overprediction.
- Prediction-on-measured slope/intercept: descriptive least-squares line
  `predicted = intercept + slope × measured`. Ideal slope 1/intercept 0.

The last line is an agreement diagnostic, not a fitted correction applied to model
outputs, and not the reverse regression sometimes called a calibration slope.
It can reveal compressed predictions (slope below 1) and intensity offsets. It
is not calibrated prediction uncertainty or evidence of correct compartment biology.

Metrics use original exported marker-intensity units, not standardized head targets.
No numerical clipping is performed. Marker scales differ, so report per marker;
no overall average MAE across all channels is presented as a universal performance
score. All channels, including DAPI, are retained. Tied ranks and constant signals
are handled without extra scipy dependency. Constant-vector correlation and
constant-measured slope/intercept are null. For constant predicted values with
variable measured values, descriptive slope 0/intercept constant remains defined.

## Aggregation and confidence intervals

Compute metrics within each patient first, then equally average patient metrics.
This does not pool all tiles as independent patients. Pearson/Spearman here are
arithmetic patient means, not pooled-tile correlations or Fisher-z pooled estimates.
Multiple samples belonging to the same patient are combined before its metrics.
Report patient SD and number with defined values for every metric.

Bootstrap: sample patient IDs with replacement, using the same bootstrap draws
across markers/metrics, and recalculate equal-patient means. Default 2,000 draws,
seed 42; report percentile 2.5th–97.5th bounds. Undefined metrics are omitted within
replicates, and effective replicate/defined-patient counts are saved. No interval
is emitted with fewer than two defined patients. Very small patient counts still
produce unstable intervals; a numeric interval is not evidence of adequate power.

These are conditional evaluation-patient sampling intervals for fixed predictions.
They do not retrain heads, resample training patients, capture all model/fold
uncertainty, or estimate individual-prediction confidence. CV models share training
patients, so these intervals should not be claimed to capture every dependency or
the full uncertainty of a model-development procedure. No tile bootstrap is used.

## Where it runs

`metrics.py` now computes the additional metrics everywhere existing metrics are
called, retaining old MAE/RMSE/Pearson keys for compatibility. `evaluate_head.py`
adds primary output files for explicit validation/test roles. `cross_validate.py`
adds primary OOF summaries after all outer folds finish. Its outer patients remain
unseen by the corresponding head/scaler; inner validation controls stopping.

`evaluate_oof.py` can re-score saved completed CV prediction arrays without GPU
or retraining. It validates fold roles, unique outer patient assignment, sample
inventory completeness, panel order, and array shapes. Example arguments are
`--cv-root /actual/completed/cv/root --out /new/evaluation/path`. Never claim OOF
results before actual CV has completed.

## Current pilot submission

Copy updated metrics.py, evaluate_head.py, and cross_validate.py; new
primary_evaluation.py, evaluate_oof.py, and slurm/primary_evaluation.sbatch.
Preserve workstation additions if merging cross_validate.py. Source snapshots of
already-running jobs are not changed by these copies.

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slurm/primary_evaluation.sbatch
```

The CPU batch scores head 47664301 against validation M5173023, using the original
three-patient caches for baseline/provenance. This does not fit or alter the head.
It is labeled validation because M5173023 selected the checkpoint, and its one
patient receives NO cohort confidence interval. Matplotlib is not needed for this
numerical batch (`--no-plots`). Plots can still be generated with evaluate_head.py.

Outputs under `uni2_primary_evaluations/job_N`:

- primary_patient_metrics.json/.csv: all metrics, per marker and patient.
- primary_summary.json/.csv: equal-patient means, SD, defined counts, CIs/method.
- Existing metrics.json/marker_summary.csv: secondary mean baseline comparison.
- Prediction arrays, checkpoint-role summary, invocation/source/event records.

The main scientific run will use all eligible patients' OOF predictions from CV,
not pilot early-stopping validation. Current 39-sample inventory and fold plan must
be reviewed before training. Hyperparameter choice must not exploit outer results.

Tests cover tied ranks, known monotonic correlation, known bias/slope/intercept,
equal-patient weighting despite unequal location counts, reproducible bootstrap,
constant/null behavior, and suppressed single-patient intervals. These tests do
not establish actual cohort accuracy or remote bootstrap results.
