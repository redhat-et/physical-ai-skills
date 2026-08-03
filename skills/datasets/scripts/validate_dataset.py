#!/usr/bin/env python3
"""Check a Hugging Face dataset's compatibility with a target model's
expected input format, without downloading it. See ../SKILL.md."""
import argparse
import json

import httpx

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"

# Tied to the pi0.5 recipe's specific training mechanism (fine-tuning skill's finetune_recipes.py),
# NOT a fixed platform-wide fact: that recipe trains via LeRobot's own native
# `lerobot-train` CLI, whose current releases default to the newer LeRobotDataset
# v3.0 format (v2.x needs an explicit conversion script or an older lerobot pin).
# An earlier version of this recipe trained via openpi's JAX scripts instead, which
# are pinned to the OLDER v2.x format -- the opposite requirement. If a future
# recipe for a different model uses a v2.x-only training mechanism again, it needs
# its own version check rather than sharing this one.
LEROBOT_COMPATIBLE_VERSION_PREFIX = "v3"


def _resolve_config_split(
    dataset_repo_id: str, config: str | None, split: str, http: httpx.Client
) -> tuple[str, str, list[str]] | str:
    """Shared config/split resolution against the HF datasets-server API.
    Returns (resolved_config, resolved_split, available_configs), or an
    error string.
    """
    splits_resp = http.get(f"{DATASETS_SERVER_URL}/splits", params={"dataset": dataset_repo_id})
    if splits_resp.status_code != 200:
        return (
            f"Could not fetch split info for '{dataset_repo_id}' "
            f"(HTTP {splits_resp.status_code}) — it may be private, "
            f"gated, or not yet processed by the datasets-server."
        )
    available = splits_resp.json().get("splits", [])
    if not available:
        return f"'{dataset_repo_id}' has no known splits."

    resolved_config = config or available[0]["config"]
    matching = [s for s in available if s["config"] == resolved_config]
    if not matching:
        configs = sorted({s["config"] for s in available})
        return f"Config '{config}' not found for '{dataset_repo_id}'. Available configs: {configs}"

    resolved_split = split if any(s["split"] == split for s in matching) else matching[0]["split"]
    return resolved_config, resolved_split, sorted({s["config"] for s in available})


def _fetch_schema_preview(dataset_repo_id: str, config: str | None, split: str) -> dict | str:
    """Returns a dict with resolved config/split/features/sample_row, or an
    error string. Uses the HF datasets-server REST API (no local `datasets`
    library dependency) to preview schema/rows without downloading anything.
    """
    try:
        with httpx.Client(timeout=15.0) as http:
            resolved = _resolve_config_split(dataset_repo_id, config, split, http)
            if isinstance(resolved, str):
                return resolved
            resolved_config, resolved_split, available_configs = resolved

            rows_resp = http.get(
                f"{DATASETS_SERVER_URL}/first-rows",
                params={"dataset": dataset_repo_id, "config": resolved_config, "split": resolved_split},
            )
            if rows_resp.status_code != 200:
                return (
                    f"Could not fetch a row preview for '{dataset_repo_id}' "
                    f"config='{resolved_config}' split='{resolved_split}' "
                    f"(HTTP {rows_resp.status_code})."
                )
            payload = rows_resp.json()
    except httpx.HTTPError as e:
        return f"Network error reaching the HF datasets-server: {e}"

    features = [
        {"name": f["name"], "dtype": f.get("type", {}).get("dtype", f.get("type", {}).get("_type", "unknown"))}
        for f in payload.get("features", [])
    ]
    sample_row = payload.get("rows", [{}])[0].get("row") if payload.get("rows") else None

    return {
        "config": resolved_config,
        "split": resolved_split,
        "available_configs": available_configs,
        "features": features,
        "sample_row": sample_row,
    }


def _validate_generic_schema(
    dataset_repo_id: str,
    expected_feature_keys: list[str] | None,
    config: str | None,
    split: str,
) -> str:
    preview = _fetch_schema_preview(dataset_repo_id, config, split)
    if isinstance(preview, str):
        return preview

    actual_columns = {f["name"] for f in preview["features"]}
    feature_lines = "\n".join(f"  - {f['name']}: {f['dtype']}" for f in preview["features"])
    header = (
        f"'{dataset_repo_id}' config='{preview['config']}' split='{preview['split']}'\n"
        f"Columns:\n{feature_lines}\n"
        f"Sample row: {preview['sample_row']}"
    )

    if not expected_feature_keys:
        return header

    missing = [c for c in expected_feature_keys if c not in actual_columns]
    extra = sorted(actual_columns - set(expected_feature_keys))
    if missing:
        return (
            f"{header}\n\n"
            f"INCOMPATIBLE: missing expected column(s) {missing}. "
            f"Present but unexpected: {extra or 'none'}."
        )
    return f"{header}\n\nCOMPATIBLE: all expected columns {expected_feature_keys} are present."


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
    """Returns the parsed meta/info.json, or an error string."""
    from huggingface_hub import hf_hub_download

    real_repo_id, subset = split_dataset_repo_id(dataset_repo_id)
    filename = f"{subset}/meta/info.json" if subset else "meta/info.json"

    try:
        info_path = hf_hub_download(repo_id=real_repo_id, repo_type="dataset", filename=filename)
    except Exception as e:
        return f"Could not fetch {filename} for '{dataset_repo_id}': {e}. Is this actually a LeRobot-format dataset?"

    with open(info_path) as f:
        return json.load(f)


def _count_camera_features(features: dict, substring: str) -> int:
    """Counts image/video-typed features whose key contains `substring`
    (case-insensitive) -- used for expected_exterior_cameras/
    expected_wrist_cameras, which check camera COUNT rather than exact key
    spelling. Exact key names vary too much between real-world DROID
    re-hosts (confirmed: 'observation.image.X' vs 'observation.images.X',
    'wrist_image_left' vs 'wrist_left' vs 'wrist') for a string-match check
    to be reliable -- and asking a model to supply an exact key it doesn't
    actually know tends to produce fabricated-but-plausible-looking keys
    (confirmed repeatedly in practice) rather than an honest "I don't know".
    """
    substring = substring.lower()
    return sum(
        1
        for name, spec in features.items()
        if substring in name.lower() and spec.get("dtype") in ("image", "video")
    )


def _validate_lerobot_dataset(
    dataset_repo_id: str,
    expected_action_dim: int | None,
    expected_exterior_cameras: int | None,
    expected_wrist_cameras: int | None,
    expected_feature_keys: list[str] | None,
) -> str:
    info = _fetch_lerobot_info(dataset_repo_id)
    if isinstance(info, str):
        return info

    codebase_version = info.get("codebase_version", "unknown")
    version_note = ""
    if not codebase_version.startswith(LEROBOT_COMPATIBLE_VERSION_PREFIX):
        version_note = (
            f"INCOMPATIBLE: codebase_version is '{codebase_version}', but this platform's current "
            f"fine-tuning recipe requires LeRobot {LEROBOT_COMPATIBLE_VERSION_PREFIX}.x -- LeRobot "
            f"dataset format versions are NOT backward/forward compatible with each other. Look for "
            f"a {LEROBOT_COMPATIBLE_VERSION_PREFIX}.x version of this dataset (or convert it), or a different one."
        )

    features = info.get("features", {})
    feature_lines = "\n".join(
        f"  - {name}: dtype={spec.get('dtype')}, shape={spec.get('shape')}" for name, spec in features.items()
    )

    result = (
        f"'{dataset_repo_id}' LeRobot metadata:\n"
        f"codebase_version: {codebase_version}\n"
        f"robot_type: {info.get('robot_type', 'unknown')}\n"
        f"fps: {info.get('fps', 'unknown')}, total_episodes: {info.get('total_episodes', 'unknown')}\n"
        f"Features:\n{feature_lines}"
    )
    if version_note:
        result += f"\n\n{version_note}"

    checks = []
    if expected_action_dim is not None:
        action_shape = features.get("action", {}).get("shape")
        actual_dim = action_shape[0] if isinstance(action_shape, list) and action_shape else None
        if actual_dim == expected_action_dim:
            checks.append(f"COMPATIBLE: raw action dim {actual_dim} matches expected {expected_action_dim}.")
        else:
            checks.append(f"INCOMPATIBLE: raw action dim is {actual_dim}, expected {expected_action_dim}.")

    if expected_exterior_cameras is not None:
        actual = _count_camera_features(features, "exterior")
        if actual == expected_exterior_cameras:
            checks.append(f"COMPATIBLE: {actual} exterior camera(s) found, matches expected {expected_exterior_cameras}.")
        else:
            checks.append(f"INCOMPATIBLE: {actual} exterior camera(s) found, expected {expected_exterior_cameras}.")

    if expected_wrist_cameras is not None:
        actual = _count_camera_features(features, "wrist")
        if actual == expected_wrist_cameras:
            checks.append(f"COMPATIBLE: {actual} wrist camera(s) found, matches expected {expected_wrist_cameras}.")
        else:
            checks.append(f"INCOMPATIBLE: {actual} wrist camera(s) found, expected {expected_wrist_cameras}.")

    if expected_feature_keys:
        missing = [k for k in expected_feature_keys if k not in features]
        if missing:
            checks.append(f"INCOMPATIBLE: missing expected feature key(s) {missing}.")
        else:
            checks.append(f"COMPATIBLE: all expected feature keys {expected_feature_keys} are present.")

    if checks:
        result += "\n\n" + "\n".join(checks)

    return result


def validate_dataset(
    dataset_repo_id: str,
    dataset_format: str = "lerobot",
    expected_feature_keys: list[str] | None = None,
    expected_action_dim: int | None = None,
    expected_exterior_cameras: int | None = None,
    expected_wrist_cameras: int | None = None,
    config: str | None = None,
    split: str = "train",
) -> str:
    """Check a Hugging Face dataset's compatibility with a target model's
    expected input format, without downloading it. Two formats:
    'lerobot' (default) reads the dataset's own meta/info.json for
    robot-policy compatibility. 'generic' checks flat column names via
    the HF datasets-server instead -- use this for non-LeRobot data.

    NEVER invent expected_action_dim/expected_exterior_cameras/
    expected_wrist_cameras/expected_feature_keys from memory or general
    knowledge of the model -- these are caller-supplied inputs this tool
    trusts verbatim and echoes back in its verdict.
    """
    if dataset_format == "lerobot":
        return _validate_lerobot_dataset(
            dataset_repo_id,
            expected_action_dim,
            expected_exterior_cameras,
            expected_wrist_cameras,
            expected_feature_keys,
        )
    if dataset_format == "generic":
        return _validate_generic_schema(dataset_repo_id, expected_feature_keys, config, split)
    return f"Unknown dataset_format '{dataset_format}'. Valid formats: 'lerobot', 'generic'."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-format", default="lerobot", choices=["lerobot", "generic"])
    parser.add_argument(
        "--expected-feature-keys", nargs="*", default=None,
        help="For 'lerobot', exact LeRobot feature keys. For 'generic', expected column names.",
    )
    parser.add_argument("--expected-action-dim", type=int, default=None, help="'lerobot' only.")
    parser.add_argument("--expected-exterior-cameras", type=int, default=None, help="'lerobot' only.")
    parser.add_argument("--expected-wrist-cameras", type=int, default=None, help="'lerobot' only.")
    parser.add_argument("--config", default=None, help="'generic' only -- defaults to the first available config.")
    parser.add_argument("--split", default="train", help="'generic' only.")
    args = parser.parse_args()

    try:
        print(
            validate_dataset(
                args.dataset_repo_id,
                args.dataset_format,
                args.expected_feature_keys,
                args.expected_action_dim,
                args.expected_exterior_cameras,
                args.expected_wrist_cameras,
                args.config,
                args.split,
            )
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
