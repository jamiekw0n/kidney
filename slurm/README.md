# Submit the three-patient filtering check with Slurm

Copy this directory and the updated cohort Python files to the remote
`/scratch/groups/emmalu/hande_annotation/kidney_uni2` directory.

This stage reads H&E and mIF and writes selected targets, filtering records, and
previews. It does not extract UNI2-h features or train a head. It needs no GPU.
Starting resource request: one task, four CPUs, 16 GB RAM, four hours. These are
configurable requests, not measured runtime/memory requirements. Account and
partition are site-specific and are intentionally not guessed.

Before submitting on the login node:

```bash
mkdir -p /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs
sbatch /scratch/groups/emmalu/hande_annotation/kidney_uni2/slurm/filter_check.sbatch
```

If Slurm requires a particular CPU partition/account, supply the site's actual
values with `--partition` and `--account` to sbatch. Inspect a previously accepted
job with `scontrol show job JOB_ID` or query partitions with `sinfo`; GPU-job
partitions need not be suitable for this CPU stage. Do not paste placeholder values.

Once sbatch returns `Submitted batch job N`, the controller owns the job. Disconnecting
your terminal does not stop it. Pending jobs have not started yet; log files may not
exist until resources are allocated. No nohup or background ampersand is needed.

```bash
squeue -u "$USER"
tail -n 50 /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-check-N.out
tail -n 50 /scratch/groups/emmalu/hande_annotation/uni2_slurm_logs/filter-check-N.err
sacct -j N --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Replace N with the real returned job ID. Filtering currently prints results per
sample, so a quiet log can be normal during scanning. Each sample's
`candidate_decisions.csv` grows as windows are processed. Avoid assuming completion
from an empty queue; check accounting state and the reports.

Unique output per submission:

```text
/scratch/groups/emmalu/hande_annotation/uni2_filter_checks/job_N/
  cohort_report.json
  cohort_status.json
  invocation.json
  events.jsonl
  source_snapshot/
  M5172118/...
  M5173023/...
  M5173144/...
```

Additional job records live under `uni2_slurm_logs/job_N`: submitted batch script,
package versions, Slurm allocation details, and exit status. Wall-time termination
or forced kills can prevent the shell exit record, so Slurm accounting is authoritative
for scheduler status. Completed samples have `manifest.json`; a finished status
report can still include failed samples. Success requires exit code zero and all
three samples complete and validated.

Copy each sample directory to your laptop and open `filter_review.html`. Inspect
retained AND rejected tissue examples, then review `filtering.json` counts. These
previews are examples, not exhaustive quality checks. Default RGB criteria may
reject pale tissue. A GPU/full-cohort run should follow only after filtering QA.

The batch uses the environment's Python directly, avoiding dependence on login-shell
Conda activation. It redirects caches/tmp to scratch and prevents user-site package
imports. If your site's environment requires module loading or activation hooks,
add those explicitly before the Python calls based on your environment configuration.
It never installs packages or downloads weights. Scratch temp files are preserved;
remove them later according to your storage policy.

The job records `--samples-as-patients` and `--assume-all-valid` as existing workflow
assumptions. Registration/tissue filtering do not prove whole-slide common support.
Replace the latter with `--valid-mask-name` when reliable support masks are available.

Validation performed locally: Bash syntax check only. Remote Slurm submission and
real filtering runtime are not verified by that check.
