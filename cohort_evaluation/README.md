# Harmonized quantitative review of kidney head modifications

CPU-only, standalone analysis of saved OUTER-TEST prediction arrays. No PyTorch,
GPU, encoder, raw image reads, retraining, threshold fitting, clipping, or image
rendering. Local modules are copied here; no code from other folders is imported.
Inputs and prior runs remain unchanged. NumPy and Python standard library suffice.

## Why recompute rather than stack existing summary tables?

Different run families have different heads, scaling, stopping and objectives.
Also, a high/low-expression table is comparable only if thresholds, sampled
locations, patient folds and aggregation match. This evaluator verifies exact
xy, measured targets, panel/encoder/physical provenance and sample/patient row
inventories, then recomputes all metrics in original measured intensity units.
It supports multiple samples per patient by concatenating before patient scoring.

Every fold uses one shared threshold vector from the fresh weighted-only run's
TRAINING expression_state.json. Its recorded split must match the reviewed fold.
Those same thresholds evaluate EVERY model. They are not learned from outer data
and do not enter any prediction calculation. They describe high versus low
expression, NOT biological positivity. A common threshold is needed even for
older models that did not use expression weighting during training.

## Default run plan

- original_mlp: uni2_cv_runs/cohort39_mlp_job_47815087.
- expression_earlier: the unambiguous completed weighted run in uni2_expression_runs;
  excludes an explicitly unweighted control if present.
- hex_control: uni2_hex_runs/control_job_48680643.
- hex_fds: uni2_hex_runs/fds_job_48822433.
- hex_alf: uni2_hex_runs/alf_job_48822437.
- hex_fds_alf: uni2_hex_runs/fds_alf_job_48680648.
- weighted_matched: completed mode weighted in uni2_weighted_fds_runs.
- weighted_fds: completed mode weighted_fds in that directory.

Automatic selection refuses multiple completed candidates rather than choosing a
convenient result. Incomplete prediction inventories are rejected. Original MLP
and earlier expression runs need not have the newer final manifest format; their
full OOF inventories/folds/arrays are still validated. Plan resolution requires
the saved primary summary as a basic completion indicator.

Comparisons within the HEX family and within the fresh weighted pair are declared
matched settings: every fold's arguments (except mode/output/split paths), head
config and scaler are checked. Source hashes for relevant training/math files are
also compared and differences recorded for review. A settings mismatch aborts a
claimed matched pair. A code-hash difference is reported, not automatically judged
benign. Historical original-versus-earlier-weighted and earlier-versus-fresh-weighted
comparisons are DESCRIPTIVE, not isolated causal effects. Source records distinguish
changes in head, preprocessing, loss, optimizer and stopping. Do not attribute a
cross-family difference solely to FDS, ALF, or expression weighting.

## Metrics and aggregation

For every patient and marker, over the SAME retained sampled centers:
- MAE, RMSE, Pearson, Spearman.
- Bias = mean(predicted-measured), plus separate high/low-region bias.
- OLS predicted-on-measured slope/intercept (descriptive calibration).
- High-expression and low-expression MAE, using shared measured threshold.
- Precision, recall, FPR at predicted intensity > that same threshold.
- AP (non-interpolated average precision, tied scores grouped) and ROC AUC.
- Measured/predicted high-expression fractions.
- diagnostic_recall_at_fpr_0.01: maximum recall at observed ROC thresholds with
  FPR <=1%; includes the all-negative prediction option. This uses the OUTER ROC
  curve, is descriptive, and is NOT a validated deployment threshold. It is not
  a replacement for future inner-validation operating-point selection.

Metrics are calculated WITHIN PATIENT first, then averaged equally across patients.
No pooling thousands of tiles as independent patients. Undefined correlations,
empty high/low strata, precision with no predicted highs, and AP/AUC with one
measured class are null and their defined counts are reported. AP should be read
alongside measured positive fraction. Native-pixel or cell-level performance is
not measured by these sampled-center block targets.

For each requested comparison, form candidate-minus-baseline metric differences
within each patient, then average paired differences. Only patients with BOTH
values defined contribute. This can differ from subtracting independently
averaged correlations. Same-seed patient-bootstrap draws give percentile95% CIs
for means and paired differences (default2000 draws, seed42). They condition on
fixed trained predictions and do not capture full model-training, hyperparameter
selection or shared-CV-fold uncertainty. No p values, multiple-testing-adjusted
claims, or universal best-model ranking are made. Raw MAE is not averaged across
markers to declare a winner: marker intensity ranges/prevalences differ.

Patient wins are reported for error metrics (lower) and correlations/AP/ROC
(higher). Bias deltas and FPR changes have no automatic winner label: closeness
to zero and detection tradeoffs need joint interpretation. Reduced FPR alone can
reflect a model predicting low everywhere.

## Launch

Copy the WHOLE cohort_evaluation folder to:
/scratch/groups/emmalu/hande_annotation/kidney_uni2/cohort_evaluation

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/cohort_evaluation/run_evaluation.sbatch
```

Requests4 CPUs,16GB,two hours on emmalu, no GPU. Existing environment supplies
NumPy; no installation. Every saved prediction is reread, so filesystem speed
matters. Progress is printed per sample. Output must be a new path; no resume or
overwrite. A partial directory does not establish successful completion.

New output: uni2_quantitative_evaluation/job_JOBID/
Logs: uni2_slurm_logs/quant-eval-JOBID.out/.err.
Job record: uni2_slurm_logs/quant_eval_JOBID/ (source/environment/tests/status).

Files:
- REPORT.md: readable six-marker tables for every model and paired comparison.
- summary.json/.csv: ALL markers/metrics, equal-patient means,95% CIs and counts.
- patient_metrics.json: all per-model/patient/marker values and subgroup counts.
- paired_differences.json/.csv: ALL paired deltas/CIs/defined counts/patient wins.
- resolved_runs.json: exact source run plan.
- protocol_audit.json: per-fold settings, scaler, source hashes, pair differences.
- common_expression_states.json: shared per-fold training threshold provenance.
- inventory.json: validated sample locations/counts.
- manifest.json complete only after all steps succeed; invocation/source snapshot.

Copy REPORT.md plus summary/paired CSVs back for review. The coding agent cannot
read /scratch on the remote workstation directly. No real patient evaluation
has run locally as part of creating this tool.

## Explicit run plan / missing or repeated experiments

Create a JSON object:

```json
{
  "runs": {"baseline": "/absolute/run_A", "candidate": "/absolute/run_B"},
  "threshold_reference": "baseline",
  "pairs": [{"baseline":"baseline", "candidate":"candidate",
             "matched_settings":true, "interpretation":"Prespecified matched comparison"}]
}
```

Use actual paths; the threshold_reference must have per-fold training expression
states. Set matched_settings false (or omit it) for historical comparisons, retaining
an honest interpretation. Evaluate even a subset explicitly if some experiments
are unavailable; the plan/output must name that subset. Do not silently exclude
patients or alter matched settings to make the evaluator pass.

```bash
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/cohort_evaluation/run_evaluation.sbatch \
 --runs-json /absolute/path/selected_runs.json
```

For plan-only validation, append --plan-only. It checks inventory/folds and writes
the plan; it does NOT validate array matching or compute quantitative results.

## Local verification

Four tests passed, including an actual synthetic two-model/four-patient evaluation,
paired improvement/CI calculations, row-order-independent matching, deliberate
changed-ground-truth rejection, tied-score ROC and missing classes, missing paired
values and ambiguous run selection. compileall and bash -n passed. These are
numerical/data-contract checks, not evidence that any kidney model is better.
