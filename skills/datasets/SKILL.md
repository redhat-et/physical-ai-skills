---
name: datasets
description: Use before searching for, inspecting, validating, or pulling a dataset for fine-tuning or general use, or when asked what data a robot-policy model (pi0.5, DreamZero) was trained or fine-tuned on.
---
DATASETS — in order:

1. list_staged_datasets first — don't re-pull an already-staged dataset.
2. For a named catalog model (e.g. 'pi05', 'dreamzero'), call
   get_model_readme(model_name) and read its Dataset Compatibility section
   for that model's checklist specifics — model_name is the catalog
   directory name, not a Hugging Face repo id. Read this fresh every time,
   even for a model discussed earlier in the same conversation; don't recall
   specs from general knowledge or a prior turn. If the README has no
   Dataset Compatibility section, or the model isn't in the catalog, say so
   and ask the user for the missing specifics rather than guessing.
3. Search using the facts from that README, not the model's own name as the
   query. 'pi05' as a query returns any embodiment anyone used with it;
   'droid' + expected_robot_type='franka' returns what the recipe actually
   needs.
4. get_dataset_info for size, license, gated status, and schema. See
   DATASET COMPATIBILITY CHECKLIST below — this is one dimension, not the
   whole picture.
5. validate_lerobot_dataset(dataset_repo_id=..., expected_exterior_cameras=..., expected_wrist_cameras=..., expected_action_dim=...) using the values from the model's README (step 2) — read them fresh each time, every single call, even ones later in the same conversation. NEVER invent expected_action_dim/expected_exterior_cameras/expected_wrist_cameras/expected_feature_keys from memory or general knowledge of the model -- these are caller-supplied inputs the tool trusts verbatim and echoes back in its verdict, so a wrong number you supplied produces a confident-looking but false "INCOMPATIBLE" against a dataset that may actually be fine. Omit expected_feature_keys entirely and read the schema yourself if you don't have a real one.
5a. A 0 count for expected_exterior_cameras/expected_wrist_cameras does NOT mean the dataset has no camera there -- validate_lerobot_dataset counts by substring match on the key name ('exterior'/'wrist'), and plenty of real datasets name cameras something else entirely, so the count can come back 0/0 even when an equivalent camera is genuinely present under a different name. Before reporting a camera-count mismatch as an incompatibility, always read the raw Features list the same call already returned and check by eye whether an image/video feature just has a different name -- don't take the count alone as the verdict.
6. EXCEPTION TO RULE 1: never call pull_dataset same-turn as get_dataset_info. Show size/license/gated status, get explicit go-ahead first.
7. After pulling, get_dataset_job_status to confirm success before saying the dataset is ready. A repo with a very large file count (thousands-plus -- e.g. one video+parquet per episode) can exhaust Hugging Face's account-tier API rate limit (1000 requests/5min) partway through: huggingface_hub resolves each file individually before downloading it, so file count, not GB, is what matters. pull_dataset now retries snapshot_download a few times (verifying actual local file count against the repo's real file count each time, since snapshot_download can itself silently return the existing local_dir instead of raising when it can't reach the repo) and, if it's still failing, falls back to a plain `git clone` + `git lfs pull` -- git-lfs resolves object URLs via a batch API instead of one request per file, so it isn't subject to the same per-file rate limit (confirmed live: 18min/zero 429s vs. an ~8h throttled crawl for a ~53k-file repo). Still worth spot-checking actual file counts on the PVC before trusting "succeeded" for anything unusual, but the common case is now self-healing.
8. If validate_lerobot_dataset (or get_dataset_info) fails to find meta/info.json at the repo root ("Could not fetch meta/info.json ... Is this actually a LeRobot-format dataset?"), don't conclude the repo isn't LeRobot-format yet — some repos bundle several independent LeRobot datasets as subfolders instead of one dataset per repo. Check get_dataset_file('README.md') or the repo's file listing for subfolder names, then retry with the subfolder appended directly to dataset_repo_id (e.g. '<org>/<repo>/<subfolder>'). pull_dataset still takes just the real two-segment repo id; the subfolder choice comes back at fine-tuning time via submit_finetune_run's dataset_subset (see the fine-tuning skill).

DATASET COMPATIBILITY CHECKLIST — use for any robot-policy dataset + model
pairing on this platform, not just pi0.5/DROID. Weight dimensions by how
adaptable the model is: a model pretrained across many embodiments
tolerates deviation in Embodiment, Perceptual Setup, and Environment
Diversity. A model fine-tuned on one narrow dataset can't lean on that, so
nearly every dimension matters — a model's own catalog README (step 2
above) says which case it is. Action Space & Representation and Format &
Tooling Compatibility are critical regardless of adaptability: a mismatch
there corrupts the training signal itself, not just downstream
generalization.

| # | Dimension | Checklist parameters | How to check it here |
|---|---|---|---|
| 1 | Embodiment & Kinematics | Arm/platform, DoF, kinematic chain, gripper type, gripper encoding + polarity, joint/workspace limits | `robot_type` from validate_lerobot_dataset (often unpopulated) + the Features list. DoF/kinematics/limits: no tool exposes these — use the target model's catalog README or ask the user. |
| 2 | Action Space & Representation | Joint vs. cartesian, absolute vs. delta, coordinate frame origin, action dimensionality, chunk horizon | validate_lerobot_dataset reports action shape. Encoding and frame aren't in the schema — use get_dataset_rows to compare `action` against a labeled `observation.state.*` field at adjacent row indices. |
| 3 | Perceptual Setup | Camera count/mounting, extrinsics/intrinsics, FOV, resolution, stereo/mono, depth/tactile | search_compatible_lerobot_datasets/validate_lerobot_dataset count camera features by name substring; search_datasets's `tags` param can pre-filter by `modality:video`/`modality:image` at search time. For calibration/FOV/depth/tactile, get_dataset_file('README.md') — not always documented, but worth checking before assuming. |
| 4 | Dynamics & Control Quality | Recording frequency, control latency, motion smoothness, teleop noise, idle-frame density | `fps` from validate_lerobot_dataset. Latency/smoothness/teleop noise: not exposed — check documented collection methodology (get_dataset_file('README.md')). |
| 5 | Normalization & Statistics | Precomputed stats availability, normalization scheme the recipe needs | get_dataset_file('meta/stats.json') to check for precomputed mean/std or q01/q99 — confirm the target recipe actually needs it (per its catalog README) before treating absence as disqualifying. |
| 6 | Format & Tooling Compatibility | Storage format, schema/codebase version, feature key naming, metadata completeness | validate_lerobot_dataset checks `codebase_version` and returns the raw Features list — exact key names and gaps are visible there. search_datasets's `tags` param can pre-filter by `format:parquet`/`LeRobot` at search time. |
| 7 | Task Structure & Annotations | Episode/trajectory length, success/failure labeling, instruction presence + specificity, task metadata | `total_episodes` from validate_lerobot_dataset; get_dataset_rows across a few indices shows whether a task/language field is populated and how specific it actually is. |
| 8 | Scale & Composition | Total episodes/frames/hours, minimum viable episode count, storage size, success/fail composition | get_dataset_info reports GB; validate_lerobot_dataset reports `total_episodes` — check both against what the recipe needs (per its catalog README). search_datasets's `size_category` pre-filters by row/frame-count bucket at search time — NOT a GB or episode-count proxy, use max_size_gb (search_compatible_lerobot_datasets) for an actual GB limit. |
| 9 | Environment & Task Diversity | Scene/task/object count, visual clutter, lighting, domain randomization | Not in the schema — get_dataset_file('README.md') for the dataset's own collection-methodology description. |
| 10 | Provenance, Identity & Licensing | Exact source/lineage, naming-confusion risk, license | get_dataset_info now also reports `Gated`/`Tags`/`Created`/`Last modified`; cross-check episode/frame counts against what a repo's name implies. search_datasets/search_compatible_lerobot_datasets can filter by `license`/`gated` at search time instead of checking each candidate after the fact. |

Don't recommend a fixed list of "known good" repo ids — Hub datasets get
renamed, replaced, and superseded. Use search_compatible_lerobot_datasets to
find current candidates, and size them against what the target model's
catalog README says it actually needs — a dry-run/smoke-test scale is not
the same as a real production fine-tune, and the README should say which
this is.

BROADER LANDSCAPE (if asked about a model/dataset not on this platform):
DROID is one of ~60 datasets in Open X-Embodiment (OXE), alongside
Bridge-v2, RT-1, and others. Format/schema compatibility doesn't guarantee
training-compatibility — OpenVLA dropped DROID from its own OXE mixture
partway through training because it hurt accuracy, despite matching
format. Don't assert a dataset for a model not covered here without
checking its actual documentation first.
