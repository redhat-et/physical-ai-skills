---
name: datasets
description: Use before inspecting, validating, or pulling a user-supplied dataset for fine-tuning or general use, or when asked what data a robot-policy model (pi0.5, DreamZero) was trained or fine-tuned on.
---
DATASETS — this platform requires users to bring their own dataset (a
Hugging Face repo id they supply). There is no dataset search/discovery
tool; never suggest candidate datasets or a repo id you weren't given
directly by the user.

## Scripts

Every capability below is a standalone script under `scripts/`, run via the
shell tool as `python3 "$SKILLS_ROOT/datasets/scripts/<name>.py" <flags>`.
Each is self-contained (no shared helper module) and does its whole job
end-to-end — including submitting to the cluster where relevant — rather
than returning a manifest for a separate step to apply.

| Script | Cluster access | Purpose |
| --- | --- | --- |
| `get_dataset_info.py` | No (HF Hub only) | Inspect a dataset without downloading it |
| `validate_dataset.py` | No (HF Hub only) | Check compatibility against a target model's expected format |
| `pull_dataset.py` | Yes | Stage a dataset onto the cluster as a PVC + download Job |
| `get_dataset_job_status.py` | Yes | Poll a `pull_dataset` download's progress |
| `convert_dataset_to_v3.py` | Yes | Convert an already-staged LeRobot v2.1 dataset to v3.0 in place |
| `get_dataset_conversion_status.py` | Yes | Poll a `convert_dataset_to_v3` conversion's progress |

In order:

1. Check what's already staged first — don't re-pull an already-staged
   dataset. No dedicated script for this: call
   `resources_list(apiVersion="v1", kind="PersistentVolumeClaim", namespace="physical-ai", labelSelector="physical-ai.io/dataset-cache=true")`
   (the general cluster tool served by the openshift-mcp-server sidecar,
   same one the models skill uses). For each PVC returned:
   `metadata.name` is the PVC name `pull_dataset.py`/`convert_dataset_to_v3.py` take
   as `--dataset-pvc-name`; `metadata.labels["physical-ai.io/dataset-repo"]`
   is the source HF repo id, but with every `/` swapped for `--` (a K8s
   label value can't contain `/`) — swap it back before showing it to the
   user or comparing against a dataset_repo_id; `spec.resources.requests.storage`
   is the requested size; `status.phase` is the bind status (`Bound` is
   healthy). No results means nothing is staged yet.
2. For a named catalog model (e.g. 'pi05', 'dreamzero'), call
   get_skill('model-specs') once, then get_model_reference(model_name) —
   model_name is the catalog directory name, not a Hugging Face repo id.
   When a model has a real fine-tuning recipe to check the user's dataset
   against, this reference is a table with the exact same `#`/Dimension
   rows as the DATASET COMPATIBILITY CHECKLIST table below, plus a
   model-specific Priority column (Critical / Adjustable / Minor) — read
   the two tables side by side by row number, don't treat every row as
   equally load-bearing just because it's in the generic table. A model
   with no fine-tuning recipe (e.g. an inference-only model) may instead
   have plain prose about training-data provenance, not a table — that's
   expected, not a gap. Read this fresh every time, even for a model
   discussed earlier in the same conversation; don't recall specs from
   general knowledge or a prior turn. If there's no reference for the
   model at all, or it isn't in the catalog, say so and ask the user for
   the missing specifics rather than guessing.
3. Once the user has given you a dataset_repo_id, run
   `get_dataset_info.py --dataset-repo-id <id> --view summary` for size,
   license, gated status, and schema. See DATASET COMPATIBILITY CHECKLIST
   below — this is one dimension, not the whole picture.
4. Run `validate_dataset.py --dataset-repo-id <id> --dataset-format lerobot
   --expected-exterior-cameras <n> --expected-wrist-cameras <n>
   --expected-action-dim <n>` using the values from the model's reference
   (step 2) — read them fresh each time, every single call, even ones later
   in the same conversation. `--dataset-format lerobot` is the default and
   applies to robot-policy models; for a non-LeRobot target (e.g. a video
   world model expecting 'video'/'caption' columns), use
   `--dataset-format generic --expected-feature-keys <col1> <col2> ...`
   instead. NEVER invent `--expected-action-dim`/`--expected-exterior-cameras`/
   `--expected-wrist-cameras`/`--expected-feature-keys` from memory or general
   knowledge of the model -- these are caller-supplied inputs this script
   trusts verbatim and echoes back in its verdict, so a wrong value you
   supplied produces a confident-looking but false result. Omit a flag
   entirely and read the schema yourself if you don't have a real value for it.
4a. A 0 count for `--expected-exterior-cameras`/`--expected-wrist-cameras` does
   NOT mean the dataset has no camera there -- `validate_dataset.py`'s lerobot
   format counts by substring match on the key name ('exterior'/'wrist'), and
   plenty of real datasets name cameras something else entirely, so the count
   can come back 0/0 even when an equivalent camera is genuinely present under
   a different name. Before reporting a camera-count mismatch as an
   incompatibility, always read the raw Features list the same call already
   returned and check by eye whether an image/video feature just has a
   different name -- don't take the count alone as the verdict.
4b. A `codebase_version` that isn't v3.x (`validate_dataset.py` will flag it as
   "INCOMPATIBLE") is not by itself a reason to reject the dataset or tell the
   user to find a v3.x-native alternative -- `convert_dataset_to_v3.py`
   converts a v2.1 LeRobot dataset to v3.0 in place on its PVC, so this is a
   fixable tooling gap, not a dead end. Report it as "will need conversion
   before fine-tuning," not as a blocking incompatibility. Before doing
   fine-tuning, you will have to make the conversion.
4c. Before passing a reference-derived number into `--expected-action-dim`/
   `--expected-exterior-cameras`/`--expected-wrist-cameras`, check HOW the
   reference states it: a number given as a hard requirement is fine to pass
   through, but a number mentioned only to describe internal recipe behavior
   (e.g. "auto-pads to N", "not required", "no fixed requirement") means that
   parameter has no correct value to pass at all -- omit it, don't pass the
   number anyway. `validate_dataset.py` always reports a raw-value mismatch as
   flatly "INCOMPATIBLE" with no way to soften it, so passing a number the
   reference itself says isn't binding produces a confident-looking but false
   blocker (confirmed: this happened twice in a row for pi0.5's action
   dimensionality, which auto-pads to 32 regardless of the dataset's raw
   shape). Also don't assume every checklist parameter bundled into one
   Dimension row inherits that row's single Priority label uniformly -- a
   row's Priority can be qualified in parentheses (e.g. "Critical (encoding
   only)", "Critical (version only)") to flag that only one sub-fact in that
   row is the actual blocker; read the qualifier and the row's full prose,
   not just the headline word, before treating a `validate_dataset.py`
   mismatch as disqualifying.
5. EXCEPTION TO RULE 1: never run `pull_dataset.py` same-turn as
   `get_dataset_info.py`. Show size/license/gated status, get explicit
   go-ahead first.
6. After pulling, run `get_dataset_job_status.py` to confirm success before
   saying the dataset is ready. A repo with a very large file count
   (thousands-plus -- e.g. one video+parquet per episode) can exhaust
   Hugging Face's account-tier API rate limit (1000 requests/5min) partway
   through: huggingface_hub resolves each file individually before
   downloading it, so file count, not GB, is what matters.
   `pull_dataset.py` retries snapshot_download a few times (verifying actual
   local file count against the repo's real file count each time, since
   snapshot_download can itself silently return the existing local_dir
   instead of raising when it can't reach the repo) and, if it's still
   failing, falls back to a plain `git clone` + `git lfs pull` -- git-lfs
   resolves object URLs via a batch API instead of one request per file, so
   it isn't subject to the same per-file rate limit (confirmed live: 18min/
   zero 429s vs. an ~8h throttled crawl for a ~53k-file repo). Still worth
   spot-checking actual file counts on the PVC before trusting "succeeded"
   for anything unusual, but the common case is now self-healing.
7. If `validate_dataset.py` (or `get_dataset_info.py`) fails to find
   meta/info.json at the repo root ("Could not fetch meta/info.json ... Is
   this actually a LeRobot-format dataset?"), don't conclude the repo isn't
   LeRobot-format yet — some repos bundle several independent LeRobot
   datasets as subfolders instead of one dataset per repo. Check
   `get_dataset_info.py --dataset-repo-id <id> --view file --filename README.md`
   or the repo's file listing for subfolder names, then retry with the
   subfolder appended directly to `--dataset-repo-id` (e.g.
   '<org>/<repo>/<subfolder>'). `pull_dataset.py` still takes just the real
   two-segment repo id; the subfolder choice comes back at fine-tuning time
   via `submit_finetune_run`'s `dataset_subset` (see the fine-tuning skill).

DATASET COMPATIBILITY CHECKLIST — use for any robot-policy dataset + model
pairing on this platform, not just pi0.5/DROID. Weight dimensions by how
adaptable the model is: a model pretrained across many embodiments
tolerates deviation in Embodiment, Perceptual Setup, and Environment
Diversity. A model fine-tuned on one narrow dataset can't lean on that, so
nearly every dimension matters — a model's own model-specs reference (step
2 above) says which case it is. Action Space & Representation and Format &
Tooling Compatibility are critical regardless of adaptability: a mismatch
there corrupts the training signal itself, not just downstream
generalization.

| # | Dimension | Checklist parameters | How to check it here |
|---|---|---|---|
| 1 | Embodiment & Kinematics | Arm/platform, DoF, kinematic chain, gripper type, gripper encoding + polarity, joint/workspace limits | `robot_type` from `validate_dataset.py` (often unpopulated) + the Features list. DoF/kinematics/limits: no tool exposes these — use the target model's model-specs reference or ask the user. |
| 2 | Action Space & Representation | Joint vs. cartesian, absolute vs. delta, coordinate frame origin, action dimensionality, chunk horizon | `validate_dataset.py` reports action shape. Encoding and frame aren't in the schema — use `get_dataset_info.py --view rows` to compare `action` against a labeled `observation.state.*` field at adjacent row indices. |
| 3 | Perceptual Setup | Camera count/mounting, extrinsics/intrinsics, FOV, resolution, stereo/mono, depth/tactile | `validate_dataset.py` counts camera features by name substring. For calibration/FOV/depth/tactile, `get_dataset_info.py --view file --filename README.md` — not always documented, but worth checking before assuming. |
| 4 | Dynamics & Control Quality | Recording frequency, control latency, motion smoothness, teleop noise, idle-frame density | `fps` from `validate_dataset.py`. Latency/smoothness/teleop noise: not exposed — check documented collection methodology (`get_dataset_info.py --view file --filename README.md`). |
| 5 | Normalization & Statistics | Precomputed stats availability, normalization scheme the recipe needs | `get_dataset_info.py --view file --filename meta/stats.json` to check for precomputed mean/std or q01/q99 — confirm the target recipe actually needs it (per its model-specs reference) before treating absence as disqualifying. |
| 6 | Format & Tooling Compatibility | Storage format, schema/codebase version, feature key naming, metadata completeness | `validate_dataset.py` checks `codebase_version` and returns the raw Features list — exact key names and gaps are visible there. A v2.1-vs-v3.0 version mismatch specifically is fixable via `convert_dataset_to_v3.py` (run reactively after a fine-tuning failure, not during this check — see step 4b), not a blocking incompatibility. |
| 7 | Task Structure & Annotations | Episode/trajectory length, success/failure labeling, instruction presence + specificity, task metadata | `total_episodes` from `validate_dataset.py`; `get_dataset_info.py --view rows` across a few indices shows whether a task/language field is populated and how specific it actually is. |
| 8 | Scale & Composition | Total episodes/frames/hours, minimum viable episode count, storage size, success/fail composition | `get_dataset_info.py --view summary` reports GB; `validate_dataset.py` reports `total_episodes` — check both against what the recipe needs (per its model-specs reference). |
| 9 | Environment & Task Diversity | Scene/task/object count, visual clutter, lighting, domain randomization | Not in the schema — `get_dataset_info.py --view file --filename README.md` for the dataset's own collection-methodology description. |
| 10 | Provenance, Identity & Licensing | Exact source/lineage, naming-confusion risk, license | `get_dataset_info.py --view summary` reports `Gated`/`Tags`/`Created`/`Last modified`/license; cross-check episode/frame counts against what a repo's name implies. |

This platform doesn't search or recommend datasets — the user supplies a
dataset_repo_id, and this checklist is for validating it against what the
target model's model-specs reference says it actually needs. A
dry-run/smoke-test scale is not the same as a real production fine-tune,
and the reference should say which this is.

BROADER LANDSCAPE (if asked about a model/dataset not on this platform):
DROID is one of ~60 datasets in Open X-Embodiment (OXE), alongside
Bridge-v2, RT-1, and others. Format/schema compatibility doesn't guarantee
training-compatibility — OpenVLA dropped DROID from its own OXE mixture
partway through training because it hurt accuracy, despite matching
format. Don't assert a dataset for a model not covered here without
checking its actual documentation first.
