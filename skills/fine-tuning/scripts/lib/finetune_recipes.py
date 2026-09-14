"""Fine-tuning recipes: per-model ordered stage lists for submit_finetune_run.

A recipe's stages run as separate Kubernetes Jobs, in order, each overriding
the same training image's command -- see the platform_agent fine-tuning
plan for why (no Kubeflow/Tekton available on this cluster; Tekton's CRDs
aren't installed despite pre-provisioned RBAC, and KFP/DSPA would need a new
SDK dependency and an unconfirmed auth story). Kept as a plain Python
constant, not a generic multi-architecture schema, until a second recipe
exists to generalize from.

pi0.5's recipe trains via LeRobot's own native `lerobot-train` CLI (pure
PyTorch, https://huggingface.co/docs/lerobot/pi05), not openpi's JAX
scripts -- confirmed that `lerobot/pi05_base` (the checkpoint our
openpi-runtime already serves live) is exactly LeRobot's own
PreTrainedPolicy save format (config.json + model.safetensors +
pre/post-processor json), so a lerobot-train checkpoint should load with
zero conversion via that same serving setup. No custom TrainConfig shim
needed either -- lerobot-train is a normal CLI.
"""

import json
from pathlib import Path

LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"


def split_dataset_repo_id(dataset_repo_id: str) -> tuple[str, str | None]:
    """A real Hugging Face repo id is always exactly two slash-separated
    segments (org/name) -- anything past that in dataset_repo_id is a
    subfolder within the repo, not part of the id, and needs splitting back
    off before any call that actually hits the Hub API (e.g.
    hf_hub_download). Exists because some repos (e.g. nvidia's
    PhysicalAI-Robotics-Manipulation-SingleArm) bundle several independent
    LeRobot datasets as subfolders of one repo instead of one dataset per
    repo -- confirmed live via the Hub API's file listing: each subfolder
    has its own meta/info.json, not the repo root.
    """
    parts = dataset_repo_id.split("/", 2)
    if len(parts) <= 2:
        return dataset_repo_id, None
    return "/".join(parts[:2]), parts[2]


def _fetch_lerobot_info(dataset_repo_id: str) -> dict | str:
    """Returns the parsed meta/info.json for a LeRobot-format dataset repo
    (or subfolder), or an error string.
    """
    from huggingface_hub import hf_hub_download

    real_repo_id, subset = split_dataset_repo_id(dataset_repo_id)
    filename = f"{subset}/meta/info.json" if subset else "meta/info.json"

    try:
        info_path = hf_hub_download(repo_id=real_repo_id, repo_type="dataset", filename=filename)
    except Exception as e:
        return f"Could not fetch {filename} for '{dataset_repo_id}': {e}. Is this actually a LeRobot-format dataset?"

    with open(info_path) as f:
        return json.load(f)

DATASET_MOUNT_ROOT = "/mnt/lerobot_home"
SOURCE_DATASET_ROOT = "/mnt/source_dataset"
PREPARED_DATASET_ROOT = "/mnt/prepared_dataset"
CHECKPOINT_MOUNT_PATH = "/mnt/checkpoint"

# The base checkpoint to fine-tune from -- same HF repo our pi05
# InferenceService already downloads and serves (platform/base/models/pi05/
# model-download-job.yaml, on the unmerged origin/feat/add-pi05-model
# branch we don't touch). Fine-tuning from this exact checkpoint, in the
# exact same checkpoint format, is what makes the "no conversion needed"
# assumption hold.
PI05_PRETRAINED_PATH = "lerobot/pi05_base"

# Per-model dataset-compatibility requirements (embodiment, camera counts,
# action space, dataset format) live in the `datasets` skill
# (platform_agent/skills/datasets.md), not here -- that content needs to
# express real uncertainty/caveats a Python dict can't, and shouldn't imply
# machine-checked ground truth it isn't.

# Shared between _train_script's actual --policy.normalization_mapping flag
# and get_recipe's logged params -- a single source of truth so the two can't
# drift apart.
NORMALIZATION_MAPPING = '{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}'

# Confirmed live via lerobot-train against lerobot/pi05_base with no
# n_action_steps override: PI05Config.validate() reports its own default as
# 50, matching its default chunk_size. Used only to decide whether
# _train_script needs to auto-cap n_action_steps when chunk_size is lowered
# without an explicit n_action_steps -- see _train_script's docstring.
PI05_BASE_DEFAULT_N_ACTION_STEPS = 50


def dataset_mount_path(dataset_repo_id: str) -> str:
    """Where a dataset PVC is mounted in every finetune stage's pod -- single
    source of truth shared between the actual volume mount (kfp.kubernetes
    mount_pvc call in finetune_pipeline.py's submit_pipeline_run) and the
    training/eval scripts below that need to tell lerobot-train/
    LeRobotDataset the same path via --dataset.root / root=.

    That explicit root is not optional: LeRobotDatasetMetadata only trusts a
    local path when it's passed as `root` -- otherwise it checks for
    `<path>/.cache/huggingface/download/`, the marker left by
    snapshot_download(local_dir=...) (exactly what pull_dataset uses), and
    treats its PRESENCE as "old, non-revision-safe download, re-fetch from
    the Hub instead" (confirmed live: without --dataset.root, lerobot-train
    ignored this mount entirely and tried to re-download over the network,
    which then failed anyway since the mount is read-only).
    """
    return f"{DATASET_MOUNT_ROOT}/{dataset_repo_id}"


def _pi05_spec_yaml() -> str:
    """Load the executable Pi0.5 compatibility contract from model-specs.

    The controller embeds this text into the prep stage because pipeline pods
    run the LeRobot image, not this repository checkout.
    """
    spec_path = Path(__file__).resolve().parents[3] / "model-specs" / "references" / "pi05.yaml"
    return spec_path.read_text(encoding="utf-8")


# How many trailing episodes to reserve for eval. Previously the eval script
# computed its own "held_out = last 5 episodes" at runtime while the train
# script had no episode filter at all -- training used ALL episodes, so
# eval's "held-out" set had already been seen during training. That made the
# eval numbers an in-sample fit check, not a real generalization measure.
NUM_EVAL_EPISODES = 5


def split_episodes(total_episodes: int) -> tuple[list[int], list[int]]:
    """Split a dataset's episodes into a training set and a genuinely
    held-out eval set. The eval episodes get passed to --dataset.episodes
    at training time to exclude them, and the exact same list gets passed
    to the eval script -- one computation, shared by both stages, so they
    can't drift apart the way the old two-independent-computations version
    could.
    """
    num_eval = min(NUM_EVAL_EPISODES, total_episodes - 1) if total_episodes > 1 else 0
    if num_eval < 1:
        raise ValueError(
            f"Dataset has only {total_episodes} episode(s) -- too few to split into a "
            f"non-empty train set and a non-empty eval set."
        )
    train_episodes = list(range(total_episodes - num_eval))
    eval_episodes = list(range(total_episodes - num_eval, total_episodes))
    return train_episodes, eval_episodes


def _checkpoint_dir(exp_name: str) -> str:
    """lerobot-train's own convention: {output_dir}/checkpoints/last/pretrained_model
    always points at the most recent checkpoint (a directory containing
    config.json + model.safetensors + pre/post-processor json -- the same
    layout as lerobot/pi05_base itself)."""
    return f"{CHECKPOINT_MOUNT_PATH}/{exp_name}/checkpoints/last/pretrained_model"


def _prepare_script(
    dataset_repo_id: str,
    chunk_size: int | None,
    n_action_steps: int | None,
    empty_cameras: int | None,
    training_profile: str,
    training_steps: int,
    batch_size_per_gpu: int | None,
    num_workers: int | None,
    save_freq: int | None,
    compile_model: bool | None,
) -> str:
    """Prepare an isolated, validated LeRobot working copy for Pi0.5.

    The source PVC is never modified. The generated script copies it to the
    run-specific prepared PVC, converts the copy if needed, derives camera
    configuration from the dataset and checkpoint, computes missing numeric
    statistics, and writes the resolved config consumed by train/evaluate.
    """
    parts = dataset_repo_id.split("/", 2)
    subset = parts[2] if len(parts) == 3 else ""
    source_root = f"{SOURCE_DATASET_ROOT}/{subset}" if subset else SOURCE_DATASET_ROOT
    spec_yaml = _pi05_spec_yaml()
    requested_chunk = "" if chunk_size is None else str(chunk_size)
    requested_action_steps = "" if n_action_steps is None else str(n_action_steps)
    requested_empty_cameras = "" if empty_cameras is None else str(empty_cameras)
    requested_batch_size = "" if batch_size_per_gpu is None else str(batch_size_per_gpu)
    requested_num_workers = "" if num_workers is None else str(num_workers)
    requested_save_freq = "" if save_freq is None else str(save_freq)
    requested_compile_model = "" if compile_model is None else str(compile_model).lower()
    return f'''\
set -euo pipefail
export HOME=/tmp
export HF_LEROBOT_HOME={PREPARED_DATASET_ROOT}
SOURCE_ROOT="{source_root}"
WORK_ROOT="{PREPARED_DATASET_ROOT}"
MODEL_REPO_ID="{PI05_PRETRAINED_PATH}"
DATASET_REPO_ID="{dataset_repo_id}"

test -f "$SOURCE_ROOT/meta/info.json" || {{
  echo "Prepared-dataset source is missing $SOURCE_ROOT/meta/info.json" >&2
  exit 1
}}

# This directory belongs exclusively to this run's prepared-dataset PVC.
mkdir -p "$WORK_ROOT"
find "$WORK_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
cp -a "$SOURCE_ROOT"/. "$WORK_ROOT"/

cat > /tmp/pi05.yaml <<'SPEC_EOF'
{spec_yaml}SPEC_EOF

DATASET_VERSION="$(python - "$WORK_ROOT/meta/info.json" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    info = json.load(f)
print(info.get("codebase_version") or info.get("version") or "")
PYEOF
)"
if [[ "${{DATASET_VERSION#v}}" != 3.* ]]; then
  echo "Converting copied dataset from $DATASET_VERSION to LeRobot v3.0"
  python -m lerobot.scripts.convert_dataset_v21_to_v30 \\
    --repo-id="$DATASET_REPO_ID" \\
    --root="$WORK_ROOT" \\
    --push-to-hub=false
fi

python - "$WORK_ROOT" "/tmp/pi05.yaml" "{requested_chunk}" "{requested_action_steps}" "{requested_empty_cameras}" "{training_profile}" "{training_steps}" "{requested_batch_size}" "{requested_num_workers}" "{requested_save_freq}" "{requested_compile_model}" <<'PYEOF'
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

root = Path(sys.argv[1])
spec = yaml.safe_load(Path(sys.argv[2]).read_text())
requested_chunk = int(sys.argv[3]) if sys.argv[3] else None
requested_action_steps = int(sys.argv[4]) if sys.argv[4] else None
requested_empty_cameras = int(sys.argv[5]) if sys.argv[5] else None
training_profile = sys.argv[6]
training_steps = int(sys.argv[7])
requested_batch_size = int(sys.argv[8]) if sys.argv[8] else None
requested_num_workers = int(sys.argv[9]) if sys.argv[9] else None
requested_save_freq = int(sys.argv[10]) if sys.argv[10] else None
requested_compile_model = sys.argv[11].lower() == "true" if sys.argv[11] else None
info_path = root / "meta" / "info.json"
info = json.loads(info_path.read_text())
features = info.get("features", {{}})
version = str(info.get("codebase_version") or info.get("version") or "")
if not version.startswith("v3"):
    raise SystemExit(f"Prepared dataset is still not LeRobot v3.x: {{version!r}}")

def fail(message):
    raise SystemExit("Pi0.5 dataset validation failed: " + message)

profiles = spec.get("training_profiles", {{}})
if training_profile not in profiles:
    fail(f"unknown Pi0.5 training profile {{training_profile!r}}; available profiles={{sorted(profiles)}}")
profile = profiles[training_profile]

def feature_dtype(specification):
    return str(specification.get("dtype", "")).lower() if isinstance(specification, dict) else ""

action = features.get("action")
if not action or not action.get("shape"):
    fail("missing shaped 'action' feature")
state_key = spec["compatibility"]["state"]["feature_key"]
if state_key not in features:
    fail(f"missing required state feature {{state_key!r}}")
task_candidates = [k for k in ("task", "task_index") if k in features]
if not task_candidates and not (root / "meta" / "tasks.parquet").exists():
    fail("no task/task_index feature or meta/tasks.parquet was found")

camera_keys = [
    key for key, value in features.items()
    if feature_dtype(value) in {{"image", "video"}}
]

def camera_role(name):
    name = name.lower()
    if any(token in name for token in ("wrist", "hand", "gripper")):
        return "wrist"
    if any(token in name for token in ("base", "front", "world", "exterior", "overhead", "top")):
        return "exterior"
    return "unknown"

model_config = {{}}
try:
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(
        repo_id="{PI05_PRETRAINED_PATH}", filename="config.json", token=os.environ.get("HF_TOKEN") or None
    )
    model_config = json.loads(Path(config_path).read_text())
except Exception as exc:
    fail(f"could not download model config for camera validation: {{exc}}")

input_features = model_config.get("input_features", {{}})
model_camera_keys = [
    key for key, value in input_features.items()
    if "image" in key.lower() or "camera" in key.lower()
    or (isinstance(value, dict) and str(value.get("type", "")).lower() in {{"image", "video", "visual"}})
]
if not model_camera_keys:
    fail("base model config did not expose any camera input features")

mapping = {{}}
unused = set(camera_keys)
warnings = []
for target in model_camera_keys:
    target_role = camera_role(target)
    candidates = [key for key in sorted(unused) if camera_role(key) == target_role]
    if not candidates and target_role == "unknown":
        candidates = sorted(unused)
    if len(candidates) != 1:
        if not candidates and len(unused) == 1:
            candidates = sorted(unused)
        else:
            fail(f"camera mapping for model slot {{target!r}} is ambiguous; dataset cameras={{camera_keys}}, model slots={{model_camera_keys}}")
    mapping[target] = candidates[0]
    unused.remove(candidates[0])

derived_empty_cameras = max(0, len(model_camera_keys) - len(mapping))
empty_cameras = requested_empty_cameras if requested_empty_cameras is not None else derived_empty_cameras
if empty_cameras < derived_empty_cameras:
    fail(f"explicit empty_cameras={{empty_cameras}} is below the required derived value {{derived_empty_cameras}}")
if unused:
    warnings.append("unused dataset camera features: " + ", ".join(sorted(unused)))

readme_text = ""
for candidate in (root / "README.md", root.parent / "README.md"):
    if candidate.exists():
        readme_text += candidate.read_text(errors="ignore").lower()
if re.search(r"\\b(delta|relative|velocity)\\b", readme_text):
    fail("dataset documentation indicates delta/relative/velocity actions; Pi0.5 recipe requires absolute actions")
if not re.search(r"\\b(absolute|joint position|joint_position)\\b", readme_text):
    warnings.append("could not prove absolute joint-position action encoding from dataset metadata/documentation")

fps = float(info.get("fps") or 0)
chunk_size = requested_chunk or (max(1, round(fps * 5)) if fps else spec["compatibility"]["control"]["default_chunk_size"])
n_action_steps = requested_action_steps or min(chunk_size, int(spec["compatibility"]["control"]["default_chunk_size"]))
if n_action_steps > chunk_size:
    fail(f"n_action_steps={{n_action_steps}} exceeds chunk_size={{chunk_size}}")

batch_size_per_gpu = requested_batch_size or int(profile["batch_size_per_gpu"])
num_workers = requested_num_workers if requested_num_workers is not None else int(profile["num_workers"])
save_freq = requested_save_freq or int(profile["save_freq"])
compile_model = requested_compile_model if requested_compile_model is not None else bool(profile["compile_model"])

# Produce the same stats.json shape LeRobot expects, without decoding video.
stats_path = root / "meta" / "stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {{}}
numeric_columns = {{"action", state_key}}
required_stat_names = ("mean", "std", "q01", "q10", "q50", "q90", "q99")
missing_stats = [
    key for key in numeric_columns
    if key not in stats or not all(name in stats[key] for name in required_stat_names)
]
if missing_stats:
    import pyarrow.parquet as pq
    chunks = {{key: [] for key in missing_stats}}
    for shard in sorted((root / "data").glob("**/*.parquet")):
        parquet = pq.ParquetFile(shard)
        available = set(parquet.schema_arrow.names)
        columns = [key for key in missing_stats if key in available]
        for batch in parquet.iter_batches(columns=columns):
            for key in columns:
                arr = batch.column(key).to_numpy(zero_copy_only=False)
                if arr.dtype == object:
                    arr = np.stack([np.asarray(row, dtype=np.float64) for row in arr])
                else:
                    arr = np.asarray(arr, dtype=np.float64).reshape(-1, 1)
                chunks[key].append(arr)
    for key in missing_stats:
        if not chunks[key]:
            fail(f"could not find numeric Parquet column {{key!r}} to compute normalization stats")
        values = np.concatenate(chunks[key], axis=0)
        stats[key] = {{
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "count": [int(values.shape[0])],
        }}
        quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        for name, values_at_quantile in zip(("q01", "q10", "q50", "q90", "q99"), quantiles):
            stats[key][name] = np.atleast_1d(values_at_quantile).tolist()
    stats_path.write_text(json.dumps(stats, indent=2) + "\\n")

resolved = {{
    "model": "pi05",
    "dataset_version": version,
    "fps": fps,
    "action_shape": action.get("shape"),
    "state_key": state_key,
    "camera_mapping": mapping,
    "empty_cameras": empty_cameras,
    "chunk_size": chunk_size,
    "n_action_steps": n_action_steps,
    "training_profile": training_profile,
    "training_steps": training_steps,
    "batch_size_per_gpu": batch_size_per_gpu,
    "num_workers": num_workers,
    "save_freq": save_freq,
    "freeze_vision_encoder": bool(profile["freeze_vision_encoder"]),
    "train_expert_only": bool(profile["train_expert_only"]),
    "gradient_checkpointing": bool(profile["gradient_checkpointing"]),
    "dtype": profile["dtype"],
    "compile_model": compile_model,
    "normalization_mapping": {{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}},
    "warnings": warnings,
}}
(root / "dataset-manifest.json").write_text(json.dumps({{"info": info, "camera_features": camera_keys}}, indent=2) + "\\n")
(root / "resolved-training-config.json").write_text(json.dumps(resolved, indent=2) + "\\n")
print(json.dumps(resolved, indent=2))
PYEOF
'''


def _train_script(
    dataset_repo_id: str,
    exp_name: str,
    num_train_steps: int,
    batch_size: int,
    train_episodes: list[int],
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    empty_cameras: int | None = None,
    training_steps: int = 50,
) -> tuple[str, int | None]:
    """Training stage script: runs lerobot-train directly -- a plain CLI, no
    custom Python config-construction shim needed unlike the old openpi-based
    recipe. Uses Pi0.5's QUANTILES normalization for action/state, with the
    preparation stage computing missing quantile statistics directly from
    Parquet without decoding video.

    huggingface/lerobot-gpu:latest already ships lerobot with pi0.5 support
    preinstalled (confirmed live: `import lerobot.policies.pi05` and
    PI05Policy both import with zero extra installs) in a uv-managed venv
    that has no `pip` binary at all -- a prior version of this script ran
    `pip install -q "lerobot[pi]"` here, which failed immediately with
    "pip: command not found" (exit 127) before training ever started. Do
    NOT re-add a pip/uv install line for this -- it's unnecessary.

    Confirmed live (full dry run, actual training steps executing on GPU)
    that three more fixes were needed beyond removing pip install:
    --dataset.root (without it, LeRobotDatasetMetadata ignores the mounted
    PVC entirely -- see dataset_mount_path's docstring -- and tries to
    re-download over the network, failing on the read-only mount);
    --policy.push_to_hub=false (cfg.validate() otherwise demands a
    --policy.repo_id to push the checkpoint to the Hub); and HF_TOKEN in the
    pod env (finetune_pipeline.py's submit_pipeline_run -- pi0.5's tokenizer
    processor loads config from PaliGemma's gated HF repo and 401s without it).

    STILL NEEDS VERIFICATION: whether lerobot-train handles DROID's
    camera/state layout correctly over a full 3000-step run (only the first
    ~10 steps were observed directly), and whether train_expert_only fits a
    single 48GB L40S for the full run (only ~25GB used in early steps).

    train_episodes excludes whatever split_episodes reserved for eval, via
    --dataset.episodes -- confirmed this flag's list-literal CLI syntax
    against LeRobot's own Makefile/CI examples (--dataset.episodes="[0]").
    Without this, the eval stage's "held-out" episodes were actually part of
    the training set the whole time (see split_episodes' docstring).

    chunk_size/empty_cameras are opt-in overrides for datasets that don't
    share DROID's fps or camera count -- confirmed real flags via
    `lerobot-train --policy.type=pi05 --help` on this same image, and a full
    (non-training) config-resolution dry run against a real --dataset.root
    confirmed they parse and resolve correctly together. A different fps
    changes the real-world time a fixed chunk_size covers; a dataset with
    fewer camera views than pi05_base's pretrained input_features needs
    padding via empty_cameras. Left unset (the default), the generated
    command is byte-identical to the pre-existing droid_100-only script,
    which needs neither.

    There used to be a rename_map flag here too, to remap a dataset's own
    camera key names to pi05_base's pretrained naming (e.g. 'world_camera'
    -> 'base_0_rgb'). Removed after confirming live it actively breaks
    training rather than helping: it renames the keys the DataLoader yields
    at batch time, but cfg.input_features (what PI05Policy._preprocess_images
    checks the batch against) gets resolved from the dataset's RAW,
    un-renamed meta/info.json names earlier in argument parsing -- the two
    sides then share zero key names, so every training step failed with
    "All image features are missing from the batch". Confirmed unnecessary
    besides: pretrained weight transfer from lerobot/pi05_base doesn't need
    matching camera key names at all ("Remapped 812 state dict keys / All
    keys loaded successfully" happened fine using a dataset's own raw
    camera names) -- that transfer is positional/structural, not
    name-matched. A dataset's own camera keys should just be left as-is.

    chunk_size and n_action_steps are related but not the same knob:
    chunk_size changes what the flow-matching loss actually supervises the
    model to predict (a training-time hyperparameter), while
    n_action_steps only controls how many of those predicted steps get
    executed before the policy replans against a fresh observation (a
    receding-horizon control choice pi05's own inference loop makes --
    confirmed live that n_action_steps=15 with chunk_size=50 parses and
    resolves fine, i.e. executing a short prefix of a longer prediction is
    a normal, valid combination, not a fallback). The one hard constraint
    (confirmed live via PI05Config.validate(), which pi05_base hits at its
    own default of 50/50): n_action_steps must not exceed chunk_size --
    "n_action_steps (50) cannot be greater than chunk_size (30)". So this
    only auto-caps n_action_steps down to chunk_size when the caller
    lowered chunk_size below 50 (pi05_base's confirmed pretrained default)
    without giving an explicit n_action_steps of their own -- an explicit
    n_action_steps always wins, letting a caller deliberately keep
    replanning more frequent than the prediction horizon.

    Returns (script, effective_n_action_steps) rather than just the script --
    get_recipe logs the latter into its MLflow params so an auto-capped run's
    provenance still records the value that actually got passed to
    lerobot-train, not just whatever the caller (or lack thereof) supplied.
    """
    optional_flags = ""
    effective_n_action_steps = n_action_steps
    if effective_n_action_steps is None and chunk_size is not None and chunk_size < PI05_BASE_DEFAULT_N_ACTION_STEPS:
        effective_n_action_steps = chunk_size

    script = f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={PREPARED_DATASET_ROOT}
PREPARED_CONFIG={PREPARED_DATASET_ROOT}/resolved-training-config.json
test -f "$PREPARED_CONFIG"
EMPTY_CAMERAS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["empty_cameras"])' "$PREPARED_CONFIG")"
RESOLVED_CHUNK_SIZE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["chunk_size"])' "$PREPARED_CONFIG")"
RESOLVED_N_ACTION_STEPS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_action_steps"])' "$PREPARED_CONFIG")"
BATCH_SIZE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size_per_gpu"])' "$PREPARED_CONFIG")"
NUM_WORKERS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_workers"])' "$PREPARED_CONFIG")"
SAVE_FREQ="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["save_freq"])' "$PREPARED_CONFIG")"
FREEZE_VISION_ENCODER="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["freeze_vision_encoder"]).lower())' "$PREPARED_CONFIG")"
TRAIN_EXPERT_ONLY="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["train_expert_only"]).lower())' "$PREPARED_CONFIG")"
GRADIENT_CHECKPOINTING="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["gradient_checkpointing"]).lower())' "$PREPARED_CONFIG")"
DTYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dtype"])' "$PREPARED_CONFIG")"
COMPILE_MODEL="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["compile_model"]).lower())' "$PREPARED_CONFIG")"
lerobot-train \\
    --dataset.repo_id={dataset_repo_id} \\
    --dataset.root={PREPARED_DATASET_ROOT} \\
    --dataset.episodes="{train_episodes}" \\
    --policy.type=pi05 \\
    --policy.push_to_hub=false \\
    --policy.pretrained_path={PI05_PRETRAINED_PATH} \\
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \\
    --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \\
    --policy.dtype="$DTYPE" \\
    --policy.device=cuda \\
    --policy.normalization_mapping='{NORMALIZATION_MAPPING}' \\
    --policy.empty_cameras="$EMPTY_CAMERAS" \\
    --policy.chunk_size="$RESOLVED_CHUNK_SIZE" \\
    --policy.n_action_steps="$RESOLVED_N_ACTION_STEPS" \\
    --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \\
    --policy.compile_model="$COMPILE_MODEL" \\
{optional_flags}    --batch_size="$BATCH_SIZE" \\
    --steps={training_steps} \\
    --num_workers="$NUM_WORKERS" \\
    --save_freq="$SAVE_FREQ" \\
    --output_dir={CHECKPOINT_MOUNT_PATH}/{exp_name} \\
    --job_name={exp_name} \\
    --wandb.enable=false
"""
    return script, effective_n_action_steps


def _evaluate_script(dataset_repo_id: str, exp_name: str, eval_episodes: list[int]) -> str:
    """Offline, self-contained evaluation -- no dependency on
    robotics-playground/Isaac Lab or any external service. Loads the
    fine-tuned checkpoint, runs it against held-out episodes from the
    staged dataset (using the checkpoint's own saved pre/post-processors,
    since pi0.5 needs its language inputs tokenized the same way it was
    trained), and reports per-episode/mean action-prediction error as a
    smoke test rather than a task-success measure.

    Also logs those metrics to MLflow (via its REST API over stdlib
    urllib, bearer-token + workspace auth -- see finetune_pipeline.py's
    MLFLOW_TRACKING_URI) so they outlive this stage pod's short lifetime,
    and are queryable later by finetune.py's get_finetune_run_status.
    Best-effort: wrapped in its own try/except so an MLflow hiccup can't
    fail the eval stage itself.
    """
    checkpoint_dir = _checkpoint_dir(exp_name)
    eval_script = f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={PREPARED_DATASET_ROOT}
cat > /tmp/run_eval.py << 'PYEOF'
import torch
import numpy as np
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CHECKPOINT_DIR = "{checkpoint_dir}"
EXP_NAME = "{exp_name}"
DATASET_REPO_ID = "{dataset_repo_id}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PI05Policy.from_pretrained(CHECKPOINT_DIR).to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CHECKPOINT_DIR,
    preprocessor_overrides={{"device_processor": {{"device": str(device)}}}},
)

dataset = LeRobotDataset(DATASET_REPO_ID, root="{PREPARED_DATASET_ROOT}")
held_out = {eval_episodes}
print(f"Evaluating against held-out episodes: {{held_out}}")

episode_frame_ranges = dataset.meta.episodes

errors = []
for ep_idx in held_out:
    policy.reset()
    from_idx = episode_frame_ranges["dataset_from_index"][ep_idx]
    to_idx = episode_frame_ranges["dataset_to_index"][ep_idx]
    frame_errors = []
    for frame_idx in range(from_idx, to_idx):
        ep = dataset[frame_idx]
        ground_truth = np.asarray(ep["action"])
        batch = {{k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in ep.items() if k != "action"}}
        batch = preprocessor(batch)
        with torch.no_grad():
            predicted = policy.select_action(batch)
        predicted = postprocessor(predicted).cpu().numpy().squeeze()
        frame_errors.append(float(np.mean((predicted - ground_truth) ** 2)))
    err = sum(frame_errors) / len(frame_errors)
    errors.append(err)
    print(f"episode {{ep_idx}}: mean action MSE over {{len(frame_errors)}} frames = {{err:.4f}}")

mean_mse = sum(errors) / len(errors)
print(f"EVAL_MEAN_ACTION_MSE={{mean_mse:.4f}}")
print("EVAL_SMOKE_TEST=PASS")
"""
    mlflow_logging = """
try:
    import json
    import os
    import ssl
    import time
    import urllib.error
    import urllib.request

    with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as _f:
        _sa_token = _f.read().strip()
    _mlflow_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_sa_token}",
        "X-MLFLOW-WORKSPACE": os.environ["MLFLOW_WORKSPACE"],
    }
    _no_verify_ctx = ssl.create_default_context()
    _no_verify_ctx.check_hostname = False
    _no_verify_ctx.verify_mode = ssl.CERT_NONE

    def _mlflow_request(method, path, payload=None):
        url = os.environ["MLFLOW_TRACKING_URI"] + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=_mlflow_headers)
        with urllib.request.urlopen(req, timeout=10, context=_no_verify_ctx) as resp:
            return json.loads(resp.read())

    try:
        experiment_id = _mlflow_request(
            "GET", "/api/2.0/mlflow/experiments/get-by-name?experiment_name=fine-tuning"
        )["experiment"]["experiment_id"]
    except urllib.error.HTTPError:
        experiment_id = _mlflow_request(
            "POST", "/api/2.0/mlflow/experiments/create", {"name": "fine-tuning"}
        )["experiment_id"]

    now_ms = int(time.time() * 1000)

    # submit_finetune_run's log_finetune_run_params already created this run
    # (run_name=EXP_NAME) at submission time to record the recipe params
    # before training even started -- find it and append metrics there
    # instead of creating a second one. Only create one here as a fallback,
    # e.g. if that submission-time logging failed.
    search_result = _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment_id],
            "filter": f"tags.\"mlflow.runName\" = '{EXP_NAME}'",
            "max_results": 1,
        },
    )
    existing_runs = search_result.get("runs", [])
    if existing_runs:
        run_id = existing_runs[0]["info"]["run_id"]
    else:
        run_id = _mlflow_request(
            "POST",
            "/api/2.0/mlflow/runs/create",
            {"experiment_id": experiment_id, "run_name": EXP_NAME, "start_time": now_ms},
        )["run"]["info"]["run_id"]

    metrics = [{"key": "mean_action_mse", "value": mean_mse, "timestamp": now_ms, "step": 0}]
    for ep_idx, err in zip(held_out, errors):
        metrics.append({"key": f"action_mse_ep{ep_idx}", "value": err, "timestamp": now_ms, "step": 0})

    _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/log-batch",
        {"run_id": run_id, "metrics": metrics},
    )
    _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/update",
        {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
    )
    print("Logged eval results to MLflow.")
except Exception as e:
    print(f"WARNING: failed to log eval results to MLflow: {e}")
PYEOF
cd /tmp && python3 run_eval.py
"""
    return eval_script + mlflow_logging


def get_recipe(
    model_name: str,
    dataset_repo_id: str,
    exp_name: str,
    dataset_subset: str | None = None,
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    empty_cameras: int | None = None,
    training_profile: str = "expert_only",
    training_steps: int = 50,
    batch_size_per_gpu: int | None = None,
    num_workers: int | None = None,
    save_freq: int | None = None,
    compile_model: bool | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Returns the ordered stage list for a model's fine-tuning recipe, plus
    the resolved recipe as a flat dict of MLflow-safe (string-valued) params
    -- submit_finetune_run logs this to MLflow at submission time so a run's
    exact provenance (dataset, step count, batch size, episode split, ...) is
    recoverable later even if these hardcoded values change in a future
    commit, or the run fails before the evaluate stage would otherwise be the
    only thing writing to MLflow at all.

    Each stage: name, image, command (list, passed to bash -c), gpu (int
    GPUs requested; 0 means no nodeSelector/GPU resource added).

    Called twice per run (submit_finetune_run for stage 0, then
    get_finetune_run_status again when advancing to stage 1) -- fetching
    total_episodes fresh each time rather than caching it is deliberate,
    since re-deriving the same split both times is what keeps
    train_episodes/eval_episodes identical across both calls without having
    to persist the split anywhere.

    dataset_subset: for a PVC pulled from a repo that bundles several
    independent LeRobot datasets as subfolders (see pull_dataset and
    split_dataset_repo_id in datasets.py) rather than one dataset per repo,
    which subfolder within that already-staged PVC to train on. Appended
    to dataset_repo_id (as "{dataset_repo_id}/{dataset_subset}") to build
    the effective identifier _fetch_lerobot_info/_train_script/
    _evaluate_script actually use for --dataset.root -- this is the one
    place that composition happens; the PVC mount path submit_finetune_run
    builds separately stays based on the plain dataset_repo_id, since the
    PVC holds the whole repo regardless of which subset a given run trains
    on.

    chunk_size/n_action_steps/empty_cameras: passed straight through to
    _train_script (see its docstring, including why rename_map was removed
    from here entirely rather than kept as an option) -- only needed for
    datasets whose fps or camera count differs from droid_100's. The eval
    stage doesn't need them separately: it loads the fine-tuned
    checkpoint's own saved config, which already has these baked in.
    """
    if model_name != "pi05":
        raise ValueError(f"No fine-tuning recipe for '{model_name}' -- only 'pi05' is defined so far.")

    effective_dataset_id = f"{dataset_repo_id}/{dataset_subset}" if dataset_subset else dataset_repo_id

    info = _fetch_lerobot_info(effective_dataset_id)
    if isinstance(info, str):
        raise ValueError(f"Could not resolve recipe for '{effective_dataset_id}': {info}")
    train_episodes, eval_episodes = split_episodes(info["total_episodes"])
    resolved_chunk_size = chunk_size or (max(1, round(float(info.get("fps") or 0) * 5)) if info.get("fps") else 50)
    resolved_n_action_steps = n_action_steps or min(resolved_chunk_size, PI05_BASE_DEFAULT_N_ACTION_STEPS)

    # Temporarily reduced from 3_000 -- at the measured ~5.4s/step pace on a
    # single L40S, 3_000 steps takes ~4.5 hours. 50 steps (~4.5 minutes) is
    # enough to validate the full pipeline (train -> checkpoint -> evaluate)
    # end to end without tying up a shared GPU for hours on every dry run.
    # Bump back up for a real training run meant to produce a usable policy.
    NUM_TRAIN_STEPS = training_steps
    BATCH_SIZE = batch_size_per_gpu or (4 if training_profile == "expert_only" else 1)

    train_script, effective_n_action_steps = _train_script(
        effective_dataset_id,
        exp_name,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        train_episodes=train_episodes,
        chunk_size=resolved_chunk_size,
        n_action_steps=resolved_n_action_steps,
        empty_cameras=empty_cameras,
        training_steps=NUM_TRAIN_STEPS,
    )

    stages = [
        {
            "name": "prepare-dataset",
            "image": LEROBOT_IMAGE,
            "gpu": 0,
            "needs_source_dataset": True,
            "command": [
                "/bin/bash",
                "-c",
                _prepare_script(
                    effective_dataset_id,
                    chunk_size=resolved_chunk_size,
                    n_action_steps=resolved_n_action_steps,
                    empty_cameras=empty_cameras,
                    training_profile=training_profile,
                    training_steps=NUM_TRAIN_STEPS,
                    batch_size_per_gpu=batch_size_per_gpu,
                    num_workers=num_workers,
                    save_freq=save_freq,
                    compile_model=compile_model,
                ),
            ],
        },
        {
            "name": "train",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": [
                "/bin/bash",
                "-c",
                train_script,
            ],
        },
        {
            "name": "evaluate",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": [
                "/bin/bash",
                "-c",
                _evaluate_script(effective_dataset_id, exp_name, eval_episodes=eval_episodes),
            ],
        },
    ]

    params = {
        "model_name": model_name,
        "dataset_repo_id": dataset_repo_id,
        "pretrained_path": PI05_PRETRAINED_PATH,
        "num_train_steps": str(NUM_TRAIN_STEPS),
        "training_profile": training_profile,
        "batch_size_per_gpu": str(BATCH_SIZE),
        "normalization_mapping": NORMALIZATION_MAPPING,
        "train_expert_only": "true" if training_profile == "expert_only" else "profile-resolved",
        "chunk_size": str(resolved_chunk_size),
        "n_action_steps": str(resolved_n_action_steps),
        "total_episodes": str(info["total_episodes"]),
        "num_train_episodes": str(len(train_episodes)),
        "num_eval_episodes": str(len(eval_episodes)),
        "eval_episodes": str(eval_episodes),
    }
    if dataset_subset:
        params["dataset_subset"] = dataset_subset
    if empty_cameras is not None:
        params["empty_cameras"] = str(empty_cameras)

    return stages, params
