#!/usr/bin/env python3
"""Start deploying a finished fine-tuning checkpoint as a live, callable
model endpoint, separate from the base model, for side-by-side testing. See
../SKILL.md."""
import argparse

from kubernetes import client

from platform_agent.config import settings

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
CHECKPOINT_DEPLOYMENT_LABEL = "physical-ai.io/checkpoint-deployment"
CHECKPOINT_EXPORT_PORT = 8080
EXPORT_JOB_TIMEOUT_SECONDS = 900  # safety net in case the import side never shows up
_SUPPORTED_MODELS = ("pi05",)

# Also defined in the fine-tuning skill's finetune_recipes.py -- duplicated
# rather than imported cross-skill, so this script has no dependency on
# another skill being installed.
CHECKPOINT_MOUNT_PATH = "/mnt/checkpoint"


def _checkpoint_dir(exp_name: str) -> str:
    """lerobot-train's own convention: {output_dir}/checkpoints/last/pretrained_model
    always points at the most recent checkpoint (a directory containing
    config.json + model.safetensors + pre/post-processor json)."""
    return f"{CHECKPOINT_MOUNT_PATH}/{exp_name}/checkpoints/last/pretrained_model"


def _checkpoint_pvc_name(exp_name: str) -> str:
    return f"finetune-{exp_name}-checkpoint-pvc"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()


def _require_pi05(model_name: str) -> str | None:
    if model_name not in _SUPPORTED_MODELS:
        return f"No checkpoint-deployment support for '{model_name}' -- only {_SUPPORTED_MODELS} so far."
    return None


def _isvc_name(model_name: str, exp_name: str) -> str:
    return f"{model_name}-ft-{exp_name}"


def _export_job_name(exp_name: str) -> str:
    return f"checkpoint-export-{exp_name}"


def _model_cache_pvc_name(isvc_name: str) -> str:
    return f"{isvc_name}-model-cache"


def deploy_checkpoint_model(exp_name: str, model_name: str = "pi05") -> str:
    """Start deploying a finished fine-tuning checkpoint as a live, callable
    model endpoint, separate from the base model, for side-by-side testing.

    This only starts the process -- it copies the checkpoint from the
    fine-tuning checkpoint PVC into a new PVC in the models namespace (a Pod
    can never mount a PVC from a different namespace, so this copy can't be
    skipped). Call get_checkpoint_deployment_status(exp_name) afterward,
    repeatedly, to advance through the copy and finish standing up the
    InferenceService -- unlike get_finetune_run_status, there's no pipeline
    driving this forward on its own.

    Only call this for a checkpoint whose fine-tuning run has actually
    succeeded (check get_finetune_run_status first) -- this only checks that
    the checkpoint PVC exists, not that training actually finished cleanly.
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    if len(isvc_name) + len("-predictor") > 63:
        return (
            f"'{exp_name}' is too long -- the resulting resource name "
            f"'{isvc_name}-predictor' would exceed Kubernetes' 63-character name limit."
        )

    core_api, batch_api, _ = _get_clients()

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=settings.datasets_namespace
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return (
                f"No fine-tuning checkpoint found for '{exp_name}' -- has submit_finetune_run "
                f"been run for it? Check get_finetune_run_status('{exp_name}')."
            )
        return f"Could not read checkpoint PVC '{checkpoint_pvc_name}': {e.reason}"

    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}
    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.models_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": model_cache_pvc, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "30Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create destination PVC '{model_cache_pvc}': {e.reason}"

    export_job_name = _export_job_name(exp_name)
    export_script = f"""\
set -e
timeout {EXPORT_JOB_TIMEOUT_SECONDS} python3 -m http.server {CHECKPOINT_EXPORT_PORT} --directory {_checkpoint_dir(exp_name)}
"""
    try:
        batch_api.create_namespaced_job(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": export_job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": EXPORT_JOB_TIMEOUT_SECONDS + 60,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "export",
                                    "image": "python:3.11-slim",
                                    "command": ["/bin/bash", "-c", export_script],
                                    "volumeMounts": [
                                        {
                                            "name": "checkpoint",
                                            "mountPath": CHECKPOINT_MOUNT_PATH,
                                            "readOnly": True,
                                        }
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "checkpoint",
                                    "persistentVolumeClaim": {"claimName": checkpoint_pvc_name},
                                }
                            ],
                        },
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to start checkpoint export Job '{export_job_name}': {e.reason}"

    return (
        f"Started copying checkpoint '{exp_name}' into the models namespace as '{isvc_name}'. "
        f"Call get_checkpoint_deployment_status('{exp_name}') to advance and check progress -- "
        f"it needs to be called repeatedly until it reports the model deployed."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, help="The exp_name passed to submit_finetune_run.")
    parser.add_argument("--model-name", default="pi05", help="Only 'pi05' is supported so far.")
    args = parser.parse_args()

    try:
        print(deploy_checkpoint_model(args.exp_name, args.model_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
