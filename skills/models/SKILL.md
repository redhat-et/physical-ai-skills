---
name: models
description: Use before listing deployed models, checking a specific model's status/output_kind, calling it for inference, scaling it up/down, tearing down a checkpoint deployment, or permanently removing a catalog model.
---
MODELS — operations on models that already exist, via the general cluster
tools (`resources_list`/`resources_get`/`resources_scale`/
`resources_create_or_update`/`pods_list_in_namespace`/`pods_log`, served by
the openshift-mcp-server sidecar). For adding a brand-new catalog model,
use deploy-model or new-model-runtime instead; for standing up a
fine-tuned checkpoint for the first time, use deploy-checkpoint. Models
run as KServe `InferenceService` objects in the `physical-ai-models`
namespace. Scale-to-zero (`minReplicas: 0`) is normal — no pods running
doesn't mean broken.

If the exact name isn't known: catalog models are listed below; a live
checkpoint comparison isn't (call `list_checkpoint_deployments`); a
checkpoint that exists but may not be deployed anywhere isn't either (call
`list_finetune_runs`). Don't guess a name in any case.

LISTING MODELS:
1. `resources_list(apiVersion="serving.kserve.io/v1beta1", kind="InferenceService", namespace="physical-ai-models")` — gives `metadata.name`, `spec.predictor.minReplicas`, `status.url` for every catalog model.
2. `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice")` — every model's pods in one call (label key present, any value). Group by the `serving.kserve.io/inferenceservice` label value to join against step 1.
3. Live status per model comes from its pods, NOT the InferenceService's own `Ready` condition — KServe reports `Ready=True` even at zero replicas, since scaled-to-zero is a valid steady state for that autoscaler class:
   - No pods for that name → "scaled to zero".
   - At least one pod with all containers ready → "running (N/M pod(s) ready)".
   - Pods exist but none ready → "starting (N pod(s) not ready yet)".

CHECKING A SPECIFIC MODEL'S STATUS:
1. `resources_get(apiVersion="serving.kserve.io/v1beta1", kind="InferenceService", namespace="physical-ai-models", name=<model_name>)`.
2. Its `metadata.annotations["physical-ai.io/output-kind"]` (default `chat` if absent) is the model's real API shape — `chat`, `image`, `video`, or `unsupported`. You need this before every single call_model, no exception for names you're already confident about — if it's `unsupported`, stop and tell the user, don't call the model.
3. Cross-check `status.conditions` against real pod health:
   `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice=<model_name>")`, then for each pod, check `status.containerStatuses[].ready` / `.restartCount` / `.state.waiting.reason`. A `waiting.reason` of `CrashLoopBackOff`/`ImagePullBackOff`/`ErrImagePull`/`InvalidImageName` means the model is broken, not starting.

CALLING A MODEL: `call_model` still requires the `output_kind` determined
above — mandatory before every single call, no exception for names you're
already confident about. If `output_kind` is "unsupported", stop and tell
the user. Relay `call_model`'s result exactly, without paraphrasing.

SCALING UP OR DOWN: A scale-up, scale-down, or retry request — including a
bare "try again" or "it looks like it's still up" with no model named —
acts on the same model name already established in the conversation (copy
it exactly, don't transcribe or prefix it) and the replica count the user
is now asking for. Never conclude the model's current scale from a
previous turn's message, including your own — only this turn's tool calls
and results tell you what's true now. Same procedure for a catalog model
or a checkpoint deployment (see the deploy-checkpoint skill).

To bring a model up (target 1 replica):
1. `resources_get` the InferenceService (as above), set `spec.predictor.minReplicas` to `1` in the fetched object, then `resources_create_or_update(resource=<the full modified object>)` — this is Server-Side Apply, so re-apply the COMPLETE resource, not a partial patch, or you'll wipe out any field you don't include.
2. If an HTTPScaledObject exists for it — `resources_get(apiVersion="http.keda.sh/v1alpha1", kind="HTTPScaledObject", namespace="physical-ai-models", name="<model_name>-http-scaler")`, a 404 is expected for always-on models like mocklm/qwen25-cpu, skip the rest of this step if so — set `spec.replicas.min` to `1` and re-apply via `resources_create_or_update`.
3. Clear any leftover KEDA pause from a previous shutdown: `resources_get(apiVersion="keda.sh/v1alpha1", kind="ScaledObject", namespace="physical-ai-models", name="<model_name>-http-scaler")`, remove the `autoscaling.keda.sh/paused-replicas` annotation if present, re-apply. Without this, a paused ScaledObject ignores incoming HTTP traffic and stays pinned at 0 forever, regardless of the InferenceService's own `minReplicas`.

To shut a model down (target 0 replicas):
1. Same InferenceService `resources_get` → set `spec.predictor.minReplicas` to `0` → `resources_create_or_update`.
2. Same HTTPScaledObject, `spec.replicas.min` to `0` (skip on 404).
3. Pin the ScaledObject to exactly 0 via the KEDA pause annotation: `resources_get` the ScaledObject, set `metadata.annotations["autoscaling.keda.sh/paused-replicas"]` to `"0"`, re-apply. The HTTPScaledObject's own idle cooldown (often ~1hr) otherwise keeps its generated HPA pinned at `minReplicas=1` while "active", fighting the direct scale-down above — the pause annotation is the only way to force an exact replica count regardless of that cooldown.
4. Force the predictor Deployment to 0 right now rather than waiting for the HPA to notice: `resources_scale(apiVersion="apps/v1", kind="Deployment", namespace="physical-ai-models", name="<model_name>-predictor", scale=0)`.
5. Delete any still-running pods so the shutdown is immediate: `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice=<model_name>")`, then `resources_delete(apiVersion="v1", kind="Pod", namespace="physical-ai-models", name=<each pod>)` for each one found.

GETTING LOGS: `pods_list_in_namespace(namespace="physical-ai-models", labelSelector="serving.kserve.io/inferenceservice=<model_name>")`, take a pod from the result, then `pods_log(namespace="physical-ai-models", name=<pod>, container="kserve-container", tail=<N>)`. No pods found means the model is likely scaled to zero, not necessarily broken.

TEARING DOWN A CHECKPOINT DEPLOYMENT: call `takedown_checkpoint_model(exp_name)`
— removes the InferenceService, HTTPScaledObject, and the PVCs/Jobs
`deploy_checkpoint_model` created for it. This does NOT touch the original
fine-tuning checkpoint PVC, so the same checkpoint can be redeployed
later.

REMOVING A CATALOG MODEL PERMANENTLY: anything under
`platform/base/models/` is GitOps (ArgoCD self-heal + prune) — a live
delete would just get recreated on the next sync. The only durable way is
removing its directory from the relevant overlay's `kustomization.yaml`
and merging a PR. Scaling it to 0 (above) is the safe, non-destructive way
to pause it instead.
