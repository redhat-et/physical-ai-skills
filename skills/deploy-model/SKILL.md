---
name: deploy-model
description: Use when sizing hardware or drafting manifests for a model that fits an existing runtime pattern (stock vLLM chat, or vLLM-Omni multimodal). For a model needing a runtime this platform hasn't run before, use new-model-runtime instead.
---
DEPLOY MODEL — for hardware sizing or adding a new model to the catalog that
fits an existing runtime pattern.

## Scripts

Every capability below is a standalone script under `scripts/`, run via the
shell tool as `python3 "$SKILLS_ROOT/deploy-model/scripts/<name>.py" <flags>`.

| Script | Cluster access | Purpose |
| --- | --- | --- |
| `list_cluster_gpus.py` | Yes (read-only) | Real GPU capacity by product type |
| `estimate_model_footprint.py` | No (HF Hub only) | VRAM sizing + tensor_parallel_size recommendation |
| `generate_model_manifests.py` | No (pure text generation) | Draft the full Kustomize/KServe file set |

In order:

1. Run `list_cluster_gpus.py` to see real GPU capacity — never guess it.
2. Run `estimate_model_footprint.py --hf-repo-id <id>` with the target
   Hugging Face repo id to get a real recommended tensor_parallel_size —
   never guess that either.
3. Only then run `generate_model_manifests.py` using the values from steps
   1-2. It only knows two runtime templates: stock vLLM (`--output-kind
   chat`) and vLLM-Omni (`--output-kind image`/`video`). If the target
   model's real serving mechanism is neither of those — e.g. it needs its
   own native server/CLI — stop here and use the new-model-runtime skill
   instead; forcing an unfamiliar runtime through these templates produces
   a broken deployment.
4. `generate_model_manifests.py` always returns the full file set for the
   model directory in one call — pvc.yaml, model-download-job.yaml,
   servingruntime.yaml, inferenceservice.yaml, httpscaledobject.yaml,
   kustomization.yaml, AND the four MaaS catalog-registration files
   (external-model.yaml, model-ref.yaml, subscription.yaml,
   auth-policy.yaml) — there's no flag to get just the InferenceService
   half. Return all of it to the user verbatim in fenced code blocks, one
   per file — do not paraphrase, shorten, summarize, or drop the MaaS files
   as boilerplate.
5. Tell the user this is a draft only: this platform uses GitOps (ArgoCD
   self-heal + prune), so nothing is actually deployed until a human saves
   ALL of the generated files, wires them into an overlay, and merges a PR.
   Nothing in this skill can deploy a model itself — `generate_model_manifests.py`
   only prints text, it never touches the cluster or git.
