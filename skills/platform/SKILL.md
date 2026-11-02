---
name: platform
description: Use when a request lands on the Physical AI platform and it isn't already clear which skill applies — "how do I fine-tune", "where do I start", "run the whole pipeline", "what's on this cluster", or any request spanning more than one stage. Routes to the narrowest skill and names the order; this skill executes nothing itself.
---
PLATFORM — the entry point and router for every other skill in this repo.

## Purpose

Orient from the current request, then hand execution to one narrow skill.
Answer "where do I start?" and "what comes after this?"; never do the work
here.

This file is a map, not a manual. Every row below names the skill that owns
the actual procedure, invariants, and scripts — read that skill before
acting. Where this router and a skill disagree, the skill wins.

## What this platform is

- An OpenShift AI (RHOAI) cluster. Datasets and fine-tuning run in the
  `physical-ai` namespace; served models are KServe `InferenceService`
  objects in `physical-ai-models`.
- Catalog models are **GitOps**: nothing is deployed until a human merges a
  PR and ArgoCD applies it (`deploy-model`, `new-model-runtime`).
- Fine-tuned checkpoints are the opposite: `deploy-checkpoint` stands one
  up **live and ephemerally**. It is never committed to git and ArgoCD
  never sees it.
- Scale-to-zero is normal. No pods for a model does not mean it is broken.
- Datasets are user-supplied Hugging Face repo ids. There is no search or
  recommendation; never invent one.
- Only `pi05` is fine-tunable today. Do not imply otherwise.

## Route the next action

Rows are in rough pipeline order. Use the narrowest row that matches what
the user is asking for *now*, not the whole arc.

| Goal | Skill |
| --- | --- |
| Ask what a catalog model is, or what data it was trained on | `model-specs` |
| Inspect, validate, stage, or convert a dataset | `datasets` |
| Submit, check, or list fine-tuning runs | `fine-tuning` |
| Stand up a fine-tuned checkpoint as a live endpoint | `deploy-checkpoint` |
| List, inspect, scale, or tear down a deployed model | `models` |
| Add a catalog model that fits stock vLLM or vLLM-Omni | `deploy-model` |
| Onboard a model whose runtime this platform has never run | `new-model-runtime` |

## End-to-end: fine-tune a robot policy

The order callers most often want. Each step is a separate skill; do not
skip a gate because the next step looks safe.

1. `model-specs` → `get_model_reference.py --model-name pi05`. Establishes
   what the user's dataset must look like before anything is pulled.
2. `datasets` → check already-staged dataset PVCs first; then
   `get_dataset_info` / `validate_dataset` on the repo id the user gave,
   then `pull_dataset` — wait for explicit confirmation. A LeRobot
   v2.1-vs-v3.0 mismatch is fixed later, reactively, not here.
3. `fine-tuning` → discuss the recipe and the GPU-hours, then
   `submit_finetune_run.py`. Never in the same turn as the initial request.
   `get_finetune_run_status.py` reads progress; it does not advance it. On a
   dataset-version failure, convert through `datasets` and resubmit the same
   exp_name (allowed only once the prior run is FAILED).
4. `deploy-checkpoint` → confirm the run succeeded, deploy, then poll
   `get_checkpoint_deployment_status.py` — that call advances the
   deployment. Scale it up through `models` before calling it.
5. `models` → scale up, read pod status and logs, tear down when done.

Stopping points: `takedown_checkpoint_model.py` for a checkpoint;
`scale_model.py --min-replicas 0` for a catalog model.

## End-to-end: add a catalog model

1. `models` → list InferenceServices first. If the model already exists, do
   not add a duplicate.
2. `deploy-model` when it serves under stock vLLM (`--output-kind chat`) or
   vLLM-Omni (`image` / `video`): real GPU capacity, then footprint, then
   manifests — in that order, never guessing either number.
3. `new-model-runtime` when neither template fits. Get the real image,
   entrypoint, port, env, and API shape before drafting anything.
4. Return every generated file verbatim, in fenced blocks. Nothing is
   deployed until a human merges the PR; no skill here can deploy a catalog
   model itself.
5. `model-specs` → write `references/<model>.md` if the new model should
   have a static spec, so later questions don't fall back to guesswork.

## Choosing between similar-sounding skills

| Request | Use | Not |
| --- | --- | --- |
| "What is pi0.5 / DreamZero, what was it trained on?" | `model-specs` | `models` (live instances only) |
| "Is my model up? Scale it. Kill it." | `models` | `deploy-model` (drafts only) |
| "Deploy my fine-tune so I can try it." | `deploy-checkpoint` | `deploy-model` (catalog + GitOps) |
| "Add this model to the catalog." | `deploy-model` or `new-model-runtime` | `deploy-checkpoint` |
| "Which dataset should I use?" | `datasets` — which must not recommend one | any |

## Readiness vocabulary

Never report a later level than was actually reached; say which one.

- **Dataset**: discovered (HF metadata read) → staged (PVC `Bound`,
  download Job succeeded) → validated (format compatible with the target
  model) → converted (v3.0, where needed).
- **Fine-tune**: submitted (KFP run exists) → running → complete (every
  stage reported done, eval numbers present) → checkpoint usable (PVC holds
  the checkpoint; pods gone).
- **Model**: drafted (manifests printed) → merged (PR and ArgoCD) → live
  (InferenceService exists) → serving (a replica is ready). "Deployed" does
  not mean a real request has been verified.

## Invariants

1. Never guess a name — model, dataset repo id, exp_name, PVC. Resolve it
   from a list call or ask.
2. Never submit GPU work (`submit_finetune_run.py`) in the same turn as the
   request that prompted it. Discuss scope and cost first.
3. Never suggest or invent a dataset. The user supplies the repo id.
4. `deploy-model` prints text only; `deploy-checkpoint` is live and
   ephemeral. Never describe one as the other.
5. Never conclude current state from an earlier turn — scale, status, and
   pod health come only from this turn's tool calls.
6. Relay generated manifests verbatim. Do not summarize or drop the MaaS
   catalog-registration files as boilerplate.

## Limitations

- This skill does not install, submit, train, scale, or deploy anything.
  It routes.
- There is no policy evaluation skill yet.
- It lists only skills in this repo, not external or NVIDIA skills.
- It does not restate each skill's procedure. Read the skill it names.

## Completion gate

Answer with the single next skill to read and the readiness level currently
established from live evidence. If nothing in the table matches, say the
request is out of scope for this repo instead of improvising a procedure.
