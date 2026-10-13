# One-sample filtering tasks with a Slurm array

Use `filter_array.py` and `slurm/filter_sample_array.sbatch`, plus updated
`cohort_prepare.py` (adds seed-index offset). Copy the current source dependencies,
including `raw_safety.py`, `image_io.py`, `audit.py`, and `validate.py`.
This isolates a native image-reader crash to one sample; it does not fix the
underlying B06458 segmentation fault. That sample still needs diagnosis if it fails.
Raw images remain read-only and derived paths are guarded.

## Launch a fresh cohort run

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
  /scratch/groups/emmalu/hande_annotation/kidney_uni2/filter_array.py plan \
  --base-dir /scratch/groups/emmalu/hande_annotation \
  --panel /scratch/groups/emmalu/hande_annotation/uni2_panel.json \
  --out /scratch/groups/emmalu/hande_annotation/uni2_filter_arrays/cohort_v1 \
  --samples-as-patients --assume-all-valid \
  --he-cache-gb 8 --mem 64G --time 06:00:00 --concurrency 2 --submit
```

Use a new output path for every plan/retry. `plan` without `--submit` records the
plan and command without launching. It creates the logs directory before sbatch.
The launcher prints the array job ID and records Slurm's response. An optional
`--samples B06458 ...` submits only specified samples. It discovers all folders first
so per-sample RNG indices remain aligned to the original sorted cohort ordering.
Known missing/ambiguous pairs are excluded from the task list and recorded in
`plan.json`; no silent selection of uncertain files occurs.

The launcher snapshots the source at planning time; tasks execute that snapshot,
not changing working source. Marker panel is also copied. Raw source paths are
checked again when each task starts. Full source integrity metadata checks occur
inside cohort preparation.

Default settings match current filtering: stride 128, 50% context and center,
5,000 retained maximum, seed 42. The array runs at most two samples concurrently
(each requesting 64 GB, four CPUs, no GPU). Increase concurrency only if resources
and shared-storage throughput justify it. Memory/time limits are configurable and
not guaranteed runtime requirements.

The task subprocess scans only one sample. A native SIGSEGV returns -11 to the
parent worker, which writes `status.json` with signal 11 and exits nonzero. Other
array tasks continue. If Slurm kills the whole task for memory/time, that status
record may be absent; scheduler accounting and logs then determine the cause.

## Monitor and inspect

```bash
squeue -u "$USER" -r -o "%.20i %.9T %.30j %.10M %.40R"
sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Replace ARRAY_JOB_ID with the returned ID. Logs are inside the plan's `logs/`:
`ARRAY_JOB_ID_TASK_ID.out`, `.err`, and `_environment.txt`. `plan.json` maps
zero-based task IDs to sample IDs.

```text
uni2_filter_arrays/cohort_v1/
  plan.json                       # fixed task list, source paths, config, exclusions
  submission.json                 # actual submitted ID/error
  submission_command.json
  source_snapshot/
  logs/
  tasks/
    0000/
      task.json
      command.json
      status.json
      events.jsonl
      prepared/
        cohort_report.json
        cohort_status.json
        SAMPLE_ID/
          progress.json
          candidate_decisions.csv
          filtering.json
          raw_input_check.json
          targets_*.npz
          filter_review.html
          manifest.json
```

This layout differs from the earlier single-process cohort root. Never pass the
nested tasks directory directly to cohort extraction. Use the collector below.
Each task refuses existing output paths; automatic overwriting/requeue resume is
not supported. Failed tasks are retried through a new plan with `--samples`.
The plan reruns valid samples freshly unless a subset is explicitly chosen; earlier
completed runs are preserved but are not automatically reused.

## Collect task status and prepare an extraction root

```bash
/scratch/groups/emmalu/hande_annotation/envs/kidney-uni2/bin/python \
  /scratch/groups/emmalu/hande_annotation/kidney_uni2/filter_array.py collect \
  --run /scratch/groups/emmalu/hande_annotation/uni2_filter_arrays/cohort_v1
```

Writes `array_report.json` with complete/failed/not-started/running-or-interrupted
states. A missing status cannot distinguish live processing from scheduler-killed
processing without Slurm accounting. Completed outputs are revalidated.

After all submitted samples pass, add:

```text
--assemble /scratch/groups/emmalu/hande_annotation/uni2_cohort_prepared/array_cohort_v1
```

Assembly creates a new root containing sample-directory symlinks and cohort/array
reports. Keep the original task outputs present. By default failed/incomplete tasks
block assembly. `--allow-partial` permits explicitly reviewed partial inclusion,
retains the exclusion report, and must not be used to silently omit difficult
patients. Retry failures and review the final patient inventory before training/CV.
Discovery exclusions remain recorded even when all launched tasks succeed.

Later `cohort_extract.py --prepared-root` should point at the assembled root. Failed
samples do not get a usable complete manifest; partial shards are not training data.

Local verification covers plan exclusions, seed indices, isolated native failure
status, output preservation, existing pipeline tests, and Bash/Python syntax. Real
Slurm array execution and the cause of the mIF crash still require workstation checks.
