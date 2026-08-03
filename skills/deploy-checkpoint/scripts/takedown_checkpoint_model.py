#!/usr/bin/env python3
"""Tear down a checkpoint deployment created by deploy_checkpoint_model. See
../SKILL.md."""
import argparse

from kubernetes import client

from platform_agent.config import settings

_SUPPORTED_MODELS = ("pi05",)


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


def _import_job_name(isvc_name: str) -> str:
    return f"checkpoint-import-{isvc_name}"


def _model_cache_pvc_name(isvc_name: str) -> str:
    return f"{isvc_name}-model-cache"


def _triton_cache_pvc_name(isvc_name: str) -> str:
    return f"{isvc_name}-triton-cache"


def _scaler_name(isvc_name: str) -> str:
    return f"{isvc_name}-http-scaler"


def takedown_checkpoint_model(exp_name: str, model_name: str = "pi05") -> str:
    """Tear down a checkpoint deployment created by deploy_checkpoint_model:
    its InferenceService, HTTPScaledObject, and the PVCs/Jobs this feature
    created for it. Frees the GPU it was holding (if scaled up) and the
    storage the checkpoint was copied into.

    Does NOT touch the original fine-tuning checkpoint PVC
    ('finetune-<exp_name>-checkpoint-pvc') -- that's still owned by the
    fine-tuning tools (see list_finetune_runs), so the same checkpoint can be
    redeployed later with deploy_checkpoint_model.
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    core_api, batch_api, custom_api = _get_clients()
    removed = []

    for group, version, plural, name in (
        ("http.keda.sh", "v1alpha1", "httpscaledobjects", _scaler_name(isvc_name)),
        ("serving.kserve.io", "v1beta1", "inferenceservices", isvc_name),
    ):
        try:
            custom_api.delete_namespaced_custom_object(
                group=group, version=version, namespace=settings.models_namespace, plural=plural, name=name
            )
            removed.append(name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete {plural} '{name}': {e.reason}"

    for pvc_name in (_model_cache_pvc_name(isvc_name), _triton_cache_pvc_name(isvc_name)):
        try:
            core_api.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=settings.models_namespace)
            removed.append(pvc_name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete PVC '{pvc_name}': {e.reason}"

    for namespace, job_name in (
        (settings.datasets_namespace, _export_job_name(exp_name)),
        (settings.models_namespace, _import_job_name(isvc_name)),
    ):
        try:
            batch_api.delete_namespaced_job(name=job_name, namespace=namespace, propagation_policy="Background")
            removed.append(job_name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete Job '{job_name}': {e.reason}"

    if not removed:
        return f"No checkpoint deployment found for '{exp_name}' -- nothing to take down."
    return f"Took down checkpoint deployment '{isvc_name}': removed {', '.join(removed)}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, help="The exp_name passed to deploy_checkpoint_model.")
    parser.add_argument("--model-name", default="pi05", help="Only 'pi05' is supported so far.")
    args = parser.parse_args()

    try:
        print(takedown_checkpoint_model(args.exp_name, args.model_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
