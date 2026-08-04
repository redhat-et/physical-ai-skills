#!/usr/bin/env python3
# ---
# description: >
#   Start a fine-tuning run for a model against an already-staged dataset.
#   Runs as a real KFP pipeline consuming real GPU-hours on the shared
#   cluster. Only call this after discussing the recipe with the user and
#   they've explicitly said to proceed -- never call this speculatively.
#   Call get_finetune_run_status afterward to check progress.
# parameters:
#   - name: dataset-pvc-name
#     type: string
#     required: true
#     description: PVC name of an already-pull_dataset-staged dataset.
#   - name: exp-name
#     type: string
#     required: true
#     description: Short experiment name, lowercase alphanumeric and hyphens.
#   - name: model-name
#     type: string
#     required: false
#     default: pi05
#     description: Only 'pi05' exists so far.
#   - name: dataset-subset
#     type: string
#     required: false
#     description: Subfolder within a multi-dataset repo PVC, if applicable.
#   - name: chunk-size
#     type: integer
#     required: false
#   - name: n-action-steps
#     type: integer
#     required: false
#   - name: empty-cameras
#     type: integer
#     required: false
# ---
"""Start a fine-tuning run for a model against an already-staged dataset.
See ../SKILL.md."""
import argparse
import os
import sys
from pathlib import Path

from kubernetes import client

DATASETS_NAMESPACE = os.environ.get("DATASETS_NAMESPACE", "physical-ai")

# Resolved from this script's own location, not a dotted platform_agent.skills
# path -- so this still works if the fine-tuning skill folder is renamed,
# moved, or installed standalone (e.g. via `npx skills add --skill
# fine-tuning`) outside of platform_agent's package tree entirely.
def _resolve_lib_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()

from lib.finetune_pipeline import get_pipeline_run_state, log_finetune_run_params, submit_pipeline_run  # noqa: E402
from lib.finetune_recipes import CHECKPOINT_MOUNT_PATH, dataset_mount_path, get_recipe  # noqa: E402

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
FINETUNE_RUN_ID_ANNOTATION = "physical-ai.io/kfp-run-id"
DATASET_REPO_LABEL = "physical-ai.io/dataset-repo"


def dataset_repo_id_from_pvc(pvc) -> str | None:
    """Recovers the HF dataset repo id a staged PVC was pulled from, via the
    DATASET_REPO_LABEL the datasets skill's pull_dataset script sets (slashes
    get swapped for "--" since K8s label values can't contain "/"). Returns
    None if the PVC isn't labeled as a dataset cache -- e.g. it wasn't
    created by pull_dataset.
    """
    label = (pvc.metadata.labels or {}).get(DATASET_REPO_LABEL)
    return label.replace("--", "/") if label else None


def _get_core_api():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def _checkpoint_pvc_name(exp_name: str) -> str:
    return f"finetune-{exp_name}-checkpoint-pvc"


def submit_finetune_run(
    dataset_pvc_name: str,
    exp_name: str,
    model_name: str = "pi05",
    dataset_subset: str | None = None,
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    empty_cameras: int | None = None,
) -> str:
    """Start a fine-tuning run for a model against an already-staged dataset.

    This runs as a real KFP pipeline (train -> evaluate for pi05) against
    RHOAI's Data Science Pipelines, consuming real GPU-hours on the shared
    cluster for potentially hours. Only call this after you've discussed
    the recipe (which model, which dataset, roughly how long it'll take)
    with the user and they've explicitly said to proceed -- never call this
    speculatively. The pipeline advances through its own stages on its
    own (no manual "create the next stage" step needed); call
    get_finetune_run_status afterward to check progress.

    Resubmitting under an exp_name that already has a run is normally
    refused -- but if that prior run's state is FAILED (e.g. after fixing a
    dataset-format error with convert_dataset_to_v3), this proceeds and
    reuses the same checkpoint PVC, overwriting the failed attempt's output.
    """
    core_api = _get_core_api()

    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(
            name=dataset_pvc_name, namespace=DATASETS_NAMESPACE
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"Dataset PVC '{dataset_pvc_name}' not found in '{DATASETS_NAMESPACE}'. Pull it first with pull_dataset."
        return f"Could not read PVC '{dataset_pvc_name}': {e.reason}"

    dataset_repo_id = dataset_repo_id_from_pvc(pvc)
    if not dataset_repo_id:
        return f"PVC '{dataset_pvc_name}' isn't labeled as a dataset cache — was it created by pull_dataset?"

    try:
        stages, recipe_params = get_recipe(
            model_name,
            dataset_repo_id,
            exp_name,
            dataset_subset=dataset_subset,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            empty_cameras=empty_cameras,
        )
    except ValueError as e:
        return str(e)

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=DATASETS_NAMESPACE,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": checkpoint_pvc_name, "labels": {FINETUNE_EXP_LABEL: exp_name}},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "100Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create checkpoint PVC: {e.reason}"
        existing_pvc = core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=DATASETS_NAMESPACE
        )
        existing_run_id = (existing_pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
        if existing_run_id and get_pipeline_run_state(existing_run_id) != "FAILED":
            return (
                f"A fine-tuning run named '{exp_name}' already exists (pipeline run "
                f"'{existing_run_id}'). Check get_finetune_run_status('{exp_name}')."
            )
        # existing_run_id's prior run reached the terminal FAILED state (or
        # was never recorded at all) -- allow retrying under the same
        # exp_name. Reuse the checkpoint PVC as-is rather than deleting and
        # recreating it: the failed run's stage pods are deliberately left
        # running for debugging (see submit_pipeline_run's ttl_seconds
        # docstring), so they'd still be mounting this PVC and a delete would
        # just hang in Terminating behind the pvc-protection finalizer.
        # lerobot-train's own --output_dir semantics already overwrite a
        # prior run's contents on a fresh run, so this is safe.

    # dataset_repo_id here is always the plain repo id pull_dataset stored on
    # the PVC (never subset-qualified -- pull_dataset always downloads the
    # whole repo). The PVC gets mounted at the path that plain id implies;
    # get_recipe is what appends dataset_subset on top of it to build
    # --dataset.root, since that only matters inside the training/eval
    # containers reading from the already-mounted filesystem, not for where
    # the PVC itself gets mounted.
    try:
        run_id, dashboard_url = submit_pipeline_run(
            exp_name=exp_name,
            model_name=model_name,
            stages=stages,
            dataset_pvc_name=dataset_pvc_name,
            checkpoint_pvc_name=checkpoint_pvc_name,
            dataset_mount_path=dataset_mount_path(dataset_repo_id),
            checkpoint_mount_path=CHECKPOINT_MOUNT_PATH,
        )
    except Exception as e:
        return f"Failed to submit fine-tuning pipeline: {e}"

    log_finetune_run_params(exp_name, recipe_params, kfp_run_id=run_id)

    try:
        core_api.patch_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name,
            namespace=DATASETS_NAMESPACE,
            body={"metadata": {"annotations": {FINETUNE_RUN_ID_ANNOTATION: run_id}}},
        )
    except client.exceptions.ApiException:
        pass

    return (
        f"Started fine-tuning '{model_name}' as experiment '{exp_name}' — pipeline run "
        f"'{run_id}' submitted to Data Science Pipelines with {len(stages)} stage(s): "
        f"{', '.join(s['name'] for s in stages)}. Find it by name ('{exp_name}') in the "
        f"RHOAI dashboard's pipeline runs list at {dashboard_url}, or call "
        f"get_finetune_run_status('{exp_name}') to check progress."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pvc-name", required=True, help="PVC name of an already-pull_dataset-staged dataset.")
    parser.add_argument("--exp-name", required=True, help="Short experiment name, lowercase alphanumeric and hyphens.")
    parser.add_argument("--model-name", default="pi05", help="Only 'pi05' exists so far.")
    parser.add_argument("--dataset-subset", default=None, help="Subfolder within a multi-dataset repo PVC, if applicable.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--empty-cameras", type=int, default=None)
    args = parser.parse_args()

    try:
        print(
            submit_finetune_run(
                args.dataset_pvc_name,
                args.exp_name,
                args.model_name,
                args.dataset_subset,
                args.chunk_size,
                args.n_action_steps,
                args.empty_cameras,
            )
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
