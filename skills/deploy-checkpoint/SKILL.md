---
name: deploy-checkpoint
description: Use when standing up a finished fine-tuning checkpoint as a live model endpoint for testing or comparison against the base model.
---
DEPLOY CHECKPOINT — for standing up a fine-tuned checkpoint as a real,
callable model endpoint.

## Scripts

Every capability below is a standalone script under `scripts/`, run via the
shell tool as `python3 "$SKILLS_ROOT/deploy-checkpoint/scripts/<name>.py" <flags>`.
Each is self-contained and does its whole job end-to-end, including
submitting to the cluster where relevant.

| Script | Purpose |
| --- | --- |
| `deploy_checkpoint_model.py` | Start copying a checkpoint into a live endpoint |
| `get_checkpoint_deployment_status.py` | Check/advance a deployment's progress |
| `takedown_checkpoint_model.py` | Tear down a checkpoint deployment |
| `list_checkpoint_deployments.py` | List currently-deployed checkpoints |

In order:

1. Confirm the fine-tuning run actually succeeded first — call
   `get_finetune_run_status.py --exp-name <name>` (or `list_finetune_runs.py`
   if the exact exp_name isn't known). Don't deploy a checkpoint from a run
   that's still in progress or failed. Only 'pi05' is supported (same
   restriction as the fine-tuning skill).
2. Run `deploy_checkpoint_model.py --exp-name <name>`. This is a LIVE
   action, not a GitOps draft like the deploy-model skill — it copies the
   checkpoint into the models namespace, converting it along the way into a
   format openpi-runtime can serve, and reuses the base pi05_droid
   checkpoint's own normalization stats (not stats recomputed from this
   run's dataset -- mention that if asked how trustworthy the deployed
   checkpoint's predictions are). It's ephemeral by design: never committed
   to git, so it won't show up in a PR and ArgoCD will never touch it.
3. Run `get_checkpoint_deployment_status.py --exp-name <name>` repeatedly to
   advance and check progress — unlike get_finetune_run_status, nothing
   else drives this forward on its own; each call both reports status and,
   once the current stage is ready, kicks off the next one. Keep calling it
   until it reports the model deployed. Copying a full checkpoint takes a
   couple of minutes, so "still copying" on an early poll is expected, not
   a failure.
4. The deployed model is scale-to-zero, same as catalog models — use the
   models skill's SCALING UP OR DOWN steps to actually warm it up before
   testing or calling it, using the isvc_name reported by
   `get_checkpoint_deployment_status.py`. First startup can take several
   minutes (image pull plus the server's own warmup inference) — use the
   models skill's status/log steps on isvc_name to check real progress
   rather than assuming failure.
5. "Deployed" only means the InferenceService/pod came up healthy -- it does
   not mean a real inference request has been verified. If asked to confirm
   the checkpoint actually serves correctly, that needs an actual request
   against the predictor (or a WebSocket handshake as a lighter check), not
   just a healthy pod.
6. This is NOT yet wired into the robotics playground's model list — that's
   a manual follow-up, not something this skill does automatically.
7. When done comparing, run `takedown_checkpoint_model.py --exp-name <name>`
   to tear it down and free the GPU/storage it was using.
