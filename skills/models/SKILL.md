---
name: models
description: Use when listing deployed models, checking a specific model's status/output_kind, scaling it up/down, tearing down a checkpoint deployment, or permanently removing a catalog model.
---
MODELS — operations on models that already exist, via the general cluster
tools (`resources_list`/`resources_get`/`pods_list_in_namespace`/`pods_log`,
served by the openshift-mcp-server sidecar) plus the dedicated
`scale_model.py` script for scaling. For adding a brand-new catalog model,
use deploy-model or new-model-runtime instead; for standing up a
fine-tuned checkpoint for the first time, use deploy-checkpoint. Models
run as KServe `InferenceService` objects in the `physical-ai-models`
namespace. Scale-to-zero (`minReplicas: 0`) is normal — no pods running
doesn't mean broken.

NOTE: calling a model for inference (formerly `call_model`) is intentionally
not part of this skill right now. It needs structured image/video output
(a media artifact channel) that a plain agentskills.io-compliant CLI script
can't produce — dropped for now to keep this repo plain-script-compliant;
revisit with a proper design (e.g. a script that writes media to a known
path/location and documents how to retrieve it) rather than re-adding it as
a bespoke LangChain tool.

If the exact name isn't known: catalog models are listed below; a live
checkpoint comparison isn't (run `list_checkpoint_deployments.py`); a
checkpoint that exists but may not be deployed anywhere isn't either (run
`list_finetune_runs.py`). Don't guess a name in any case.

## Scripts

Run via the shell tool: `python3 "$SKILLS_ROOT/models/scripts/scale_model.py" --model-name <name> --min-replicas <0|1>`.

LISTING MODELS:
1. `resources_list(apiVersion="serving.kserve.io/v1beta1", kind="InferenceService", namespace="physical-ai-models")` — gives `metadata.name`, `spec.predictor.minReplicas`, `status.url` for every catalog model.
2. `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice")` — every model's pods in one call (label key present, any value). Group by the `serving.kserve.io/inferenceservice` label value to join against step 1.
3. Live status per model comes from its pods, NOT the InferenceService's own `Ready` condition — KServe reports `Ready=True` even at zero replicas, since scaled-to-zero is a valid steady state for that autoscaler class:
   - No pods for that name → "scaled to zero".
   - At least one pod with all containers ready → "running (N/M pod(s) ready)".
   - Pods exist but none ready → "starting (N pod(s) not ready yet)".

CHECKING A SPECIFIC MODEL'S STATUS:
1. `resources_get(apiVersion="serving.kserve.io/v1beta1", kind="InferenceService", namespace="physical-ai-models", name=<model_name>)`.
2. Its `metadata.annotations["physical-ai.io/output-kind"]` (default `chat` if absent) is the model's real API shape — `chat`, `image`, `video`, or `unsupported`. This is otherwise unused right now (see the NOTE above about inference-calling being dropped), but keep reporting it — it's what a future inference-calling replacement will need.
3. Cross-check `status.conditions` against real pod health:
   `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice=<model_name>")`, then for each pod, check `status.containerStatuses[].ready` / `.restartCount` / `.state.waiting.reason`. A `waiting.reason` of `CrashLoopBackOff`/`ImagePullBackOff`/`ErrImagePull`/`InvalidImageName` means the model is broken, not starting.

SCALING UP OR DOWN: A scale-up, scale-down, or retry request — including a
bare "try again" or "it looks like it's still up" with no model named —
acts on the same model name already established in the conversation (copy
it exactly, don't transcribe or prefix it) and the replica count the user
is now asking for. Never conclude the model's current scale from a
previous turn's message, including your own — only this turn's tool calls
and results tell you what's true now. Same procedure for a catalog model
or a checkpoint deployment (see the deploy-checkpoint skill).

Run `scale_model.py --model-name <name> --min-replicas <n>` — `1` to bring
it up, `0` to shut it down. This single script handles the full KEDA-aware
sequence itself: the InferenceService's `minReplicas`, the
HTTPScaledObject's own `spec.replicas.min` (skipped for always-on models
like mocklm/qwen25-cpu, which don't have one), setting or clearing the
ScaledObject's `autoscaling.keda.sh/paused-replicas` annotation, and — on
shutdown — forcing the predictor Deployment to 0 replicas and deleting any
still-running pods immediately rather than waiting for the HPA to notice.
Don't reimplement any of this by hand via
`resources_get`/`resources_create_or_update`/`resources_scale` — a
partially-applied sequence leaves the model stuck (most commonly: a
paused ScaledObject silently ignoring all future HTTP traffic, pinning it
at 0 forever regardless of the InferenceService's own `minReplicas`).

After running `scale_model.py`, use the CHECKING A SPECIFIC MODEL'S STATUS
steps above to confirm the change actually took — especially for
scale-up, since first startup can take several minutes and isn't
instantaneous just because the call returned.

GETTING LOGS: `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice=<model_name>")`, take a pod from the result, then `pods_log(namespace="physical-ai-models", name=<pod>, container="kserve-container", tail=<N>)`. No pods found means the model is likely scaled to zero, not necessarily broken.

TEARING DOWN A CHECKPOINT DEPLOYMENT: run
`takedown_checkpoint_model.py --exp-name <name>` (deploy-checkpoint skill)
— removes the InferenceService, HTTPScaledObject, and the PVCs/Jobs
`deploy_checkpoint_model.py` created for it. This does NOT touch the
original fine-tuning checkpoint PVC, so the same checkpoint can be
redeployed later.

REMOVING A CATALOG MODEL PERMANENTLY: anything under
`platform/base/models/` is GitOps (ArgoCD self-heal + prune) — a live
delete would just get recreated on the next sync. The only durable way is
removing its directory from the relevant overlay's `kustomization.yaml`
and merging a PR. Scaling it to 0 (above) is the safe, non-destructive way
to pause it instead.
