---
name: fine-tuning
description: Use before discussing, submitting, or checking on a fine-tuning run — GPU training jobs on pi05/robot-policy checkpoints, submit_finetune_run, get_finetune_run_status, or questions like "how's my fine-tune doing" or "kick off training on X dataset."
---
FINE-TUNING

## Scripts

Every capability below is a standalone script under `scripts/`, run via the
shell tool as `python3 "$SKILLS_ROOT/fine-tuning/scripts/<name>.py" <flags>`.
Each is self-contained and does its whole job end-to-end.

| Script | Purpose |
| --- | --- |
| `submit_finetune_run.py` | Submit a fine-tuning pipeline run against a staged dataset |
| `get_finetune_run_status.py` | Check a run's progress, stages, and eval results |
| `list_finetune_runs.py` | List all fine-tuning experiments on this cluster |

In order:

0. Only 'pi05' is supported as a fine-tuning target right now — say so up front if the user names a different model.
1. Confirm the dataset is staged (`pull_dataset.py`) and, for robot-policy models, validated (`validate_dataset.py --dataset-format lerobot`) before ever running `submit_finetune_run.py` — see the datasets skill for that workflow. If the dataset repo bundles multiple independent LeRobot datasets as subfolders (see the datasets skill), pass the subfolder name as `--dataset-subset`.
2. Discuss the recipe with the user first — model, dataset, that this runs real GPU-hours on the shared cluster for potentially hours.
3. EXCEPTION TO RULE 1: never run `submit_finetune_run.py` in the same turn as the initial fine-tuning request, for any reason, including to "check" whether the dataset/config is valid — that speculative call IS the forbidden action, whether or not it succeeds. Use `get_dataset_job_status.py`, or the datasets skill's `resources_list` check for what's already staged, instead to check preconditions. Wait until the user explicitly says to proceed — same carve-out as pull_dataset, higher stakes (GPU-hours, not just storage).
4. The pipeline advances through its own stages on its own — `get_finetune_run_status.py` is a read-only progress check, not something that needs repeated calls to make a stage happen.
5. If the exact exp_name isn't known, or the question is general ("what's running", "any fine-tunes in progress"), run `list_finetune_runs.py` — don't guess a name or say there's no way to check.
6. Relay final eval numbers and the checkpoint PVC name only once `get_finetune_run_status.py` reports all stages complete.
7. If `get_finetune_run_status.py` reports a FAILED stage, it now includes that stage's recent log lines -- read them. If they mention a LeRobot dataset-format/version error (e.g. BackwardCompatibilityError, "v2.1", "v3.0"): run `convert_dataset_to_v3.py --dataset-pvc-name <name>`, wait for `get_dataset_conversion_status.py` to report succeeded, then run `submit_finetune_run.py` again with the same `--dataset-pvc-name` and `--exp-name` -- resubmitting under the same exp_name is only allowed once the prior run's state is FAILED. For any other failure reason, don't guess a fix -- report the log contents to the user.
