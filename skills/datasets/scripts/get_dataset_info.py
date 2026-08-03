#!/usr/bin/env python3
"""Inspect a Hugging Face dataset without downloading it. See ../SKILL.md."""
import argparse

import httpx

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"

MAX_DATASET_ROWS_PER_CALL = 20
MAX_DATASET_ROWS_OUTPUT_CHARS = 20_000
MAX_DATASET_FILE_BYTES = 2_000_000
MAX_DATASET_FILE_CHARS = 20_000


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


def _dataset_info_rows(
    dataset_repo_id: str,
    offset: int,
    length: int,
    config: str | None,
    split: str,
) -> str:
    length = max(1, min(length, MAX_DATASET_ROWS_PER_CALL))
    try:
        with httpx.Client(timeout=15.0) as http:
            resolved = _resolve_config_split(dataset_repo_id, config, split, http)
            if isinstance(resolved, str):
                return resolved
            resolved_config, resolved_split, _ = resolved

            rows_resp = http.get(
                f"{DATASETS_SERVER_URL}/rows",
                params={
                    "dataset": dataset_repo_id,
                    "config": resolved_config,
                    "split": resolved_split,
                    "offset": offset,
                    "length": length,
                },
            )
            if rows_resp.status_code != 200:
                return (
                    f"Could not fetch rows for '{dataset_repo_id}' "
                    f"config='{resolved_config}' split='{resolved_split}' "
                    f"offset={offset} length={length} (HTTP {rows_resp.status_code})."
                )
            payload = rows_resp.json()
    except httpx.HTTPError as e:
        return f"Network error reaching the HF datasets-server: {e}"

    rows = payload.get("rows", [])
    if not rows:
        return (
            f"No rows returned for '{dataset_repo_id}' at offset={offset} "
            f"(config='{resolved_config}', split='{resolved_split}')."
        )

    lines = [
        f"'{dataset_repo_id}' config='{resolved_config}' split='{resolved_split}', "
        f"rows {offset}-{offset + len(rows) - 1}:"
    ]
    for r in rows:
        lines.append(f"  row_idx={r.get('row_idx')}: {r.get('row')}")
    out = "\n".join(lines)
    if len(out) > MAX_DATASET_ROWS_OUTPUT_CHARS:
        out = out[:MAX_DATASET_ROWS_OUTPUT_CHARS] + "\n... (truncated -- request fewer rows)"
    return out


def _dataset_info_summary(dataset_repo_id: str, config: str | None, split: str) -> str:
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(dataset_repo_id, files_metadata=True)
    except Exception as e:
        return f"Could not fetch dataset info for '{dataset_repo_id}' from Hugging Face: {e}"

    size_bytes = info.used_storage
    if size_bytes is None and info.siblings:
        size_bytes = sum((s.size or (s.lfs.size if s.lfs else 0) or 0) for s in info.siblings)
    size_str = f"{size_bytes / 1e9:.2f} GB" if size_bytes else "unknown"

    license_str = getattr(info.card_data, "license", None) or "unknown"
    task_categories = getattr(info.card_data, "task_categories", None) or []

    preview = _fetch_schema_preview(dataset_repo_id, config, split)
    if isinstance(preview, str):
        schema_section = f"Schema preview unavailable: {preview}"
    else:
        feature_lines = "\n".join(f"  - {f['name']}: {f['dtype']}" for f in preview["features"])
        schema_section = (
            f"Configs available: {preview['available_configs']}\n"
            f"Previewing config='{preview['config']}' split='{preview['split']}'\n"
            f"Columns:\n{feature_lines}\n"
            f"Sample row: {preview['sample_row']}"
        )

    return (
        f"'{dataset_repo_id}':\n"
        f"Size: {size_str}\n"
        f"License: {license_str}\n"
        f"Task categories: {task_categories}\n"
        f"Downloads: {info.downloads or 0}, Likes: {info.likes or 0}\n"
        f"Gated: {info.gated}\n"
        f"Created: {info.created_at}, Last modified: {info.last_modified}\n"
        f"Tags: {info.tags or []}\n"
        f"{schema_section}"
    )


def _dataset_info_file(dataset_repo_id: str, filename: str) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    try:
        info = HfApi().dataset_info(dataset_repo_id, files_metadata=True)
    except Exception as e:
        return f"Could not look up '{dataset_repo_id}': {e}"

    sibling = next((s for s in (info.siblings or []) if s.rfilename == filename), None)
    if sibling is None:
        available = sorted(s.rfilename for s in (info.siblings or []))[:30]
        return f"'{filename}' not found in '{dataset_repo_id}'. Some files present: {available}"

    size = sibling.size or (sibling.lfs.size if sibling.lfs else 0)
    if size and size > MAX_DATASET_FILE_BYTES:
        return (
            f"'{filename}' is {size / 1e6:.1f}MB — too large to fetch as text "
            f"(limit {MAX_DATASET_FILE_BYTES / 1e6:.0f}MB). This tool is for "
            f"docs/metadata, not data files."
        )

    try:
        path = hf_hub_download(repo_id=dataset_repo_id, repo_type="dataset", filename=filename)
    except Exception as e:
        return f"Could not download '{filename}' from '{dataset_repo_id}': {e}"

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"'{filename}' isn't text — this tool only reads text/metadata files."

    if len(content) > MAX_DATASET_FILE_CHARS:
        content = content[:MAX_DATASET_FILE_CHARS] + f"\n... (truncated, {len(content)} total characters)"
    return f"'{dataset_repo_id}/{filename}':\n{content}"


def get_dataset_info(
    dataset_repo_id: str,
    view: str = "summary",
    filename: str | None = None,
    offset: int = 0,
    length: int = 5,
    config: str | None = None,
    split: str = "train",
) -> str:
    """Inspect a Hugging Face dataset without downloading it. Three views:
    'summary' (default) for size, license, gated status, tags, dates,
    schema, and one sample row; 'rows' for a range of raw rows via
    offset/length, to compare fields across several rows instead of just
    one; 'file' for one file's raw text (e.g. 'README.md' or
    'meta/stats.json'). See the datasets skill for which view answers
    which compatibility question.

    Always call view='summary' and relay its size/license/Gated status to
    the user before ever calling pull_dataset -- pulling consumes real
    shared-cluster storage, and a gated dataset needs manual approval
    first.
    """
    if view == "summary":
        return _dataset_info_summary(dataset_repo_id, config, split)
    if view == "rows":
        return _dataset_info_rows(dataset_repo_id, offset, length, config, split)
    if view == "file":
        if not filename:
            return "view='file' requires a filename, e.g. 'README.md' or 'meta/stats.json'."
        return _dataset_info_file(dataset_repo_id, filename)
    return f"Unknown view '{view}'. Valid views: 'summary', 'rows', 'file'."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True, help="HF dataset repo id, e.g. 'GEAR-Dreams/DreamZero-DROID'.")
    parser.add_argument("--view", default="summary", choices=["summary", "rows", "file"])
    parser.add_argument("--filename", default=None, help="Required when --view=file, e.g. 'README.md'.")
    parser.add_argument("--offset", type=int, default=0, help="Row index to start from, only used when --view=rows.")
    parser.add_argument("--length", type=int, default=5, help="Number of rows to fetch, only used when --view=rows (capped at 20).")
    parser.add_argument("--config", default=None, help="Dataset config/subset name. Defaults to the first available one.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    try:
        print(get_dataset_info(args.dataset_repo_id, args.view, args.filename, args.offset, args.length, args.config, args.split))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
