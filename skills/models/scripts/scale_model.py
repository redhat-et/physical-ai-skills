#!/usr/bin/env python3
"""Scale a model by setting its minReplicas. See ../SKILL.md."""
import argparse

from kubernetes import client
from kubernetes.client import AppsV1Api

from platform_agent.config import settings


def _get_k8s_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CustomObjectsApi(), client.CoreV1Api()


def scale_model(model_name: str, min_replicas: int) -> str:
    """Scale a model by setting its minReplicas. Use 1 to bring a model up
    and keep it running. Use 0 to shut it down immediately.
    """
    custom_api, core_api = _get_k8s_client()

    try:
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=model_name,
            body={"spec": {"predictor": {"minReplicas": min_replicas}}},
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"InferenceService '{model_name}' not found."
        return f"Failed to scale '{model_name}': {e.reason}"

    scaler_name = f"{model_name}-http-scaler"
    try:
        custom_api.patch_namespaced_custom_object(
            group="http.keda.sh",
            version="v1alpha1",
            namespace=settings.models_namespace,
            plural="httpscaledobjects",
            name=scaler_name,
            body={"spec": {"replicas": {"min": min_replicas}}},
        )
    except client.exceptions.ApiException:
        pass

    # The HTTPScaledObject's own scaledownPeriod (idle cooldown, often ~1hr)
    # otherwise keeps its generated ScaledObject/HPA pinned at minReplicas=1
    # while "active", fighting any direct scale-down below. KEDA's pause
    # annotation is the documented way to force an exact replica count
    # regardless of triggers/cooldown; not all models have this scaler
    # (always-on models like mocklm/qwen25-cpu don't), so 404s are expected.
    try:
        if min_replicas == 0:
            custom_api.patch_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=settings.models_namespace,
                plural="scaledobjects",
                name=scaler_name,
                body={"metadata": {"annotations": {"autoscaling.keda.sh/paused-replicas": "0"}}},
            )
        else:
            custom_api.patch_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=settings.models_namespace,
                plural="scaledobjects",
                name=scaler_name,
                body={"metadata": {"annotations": {"autoscaling.keda.sh/paused-replicas": None}}},
            )
    except client.exceptions.ApiException:
        pass

    if min_replicas == 0:
        apps_api = AppsV1Api()
        deploy_name = f"{model_name}-predictor"
        try:
            apps_api.patch_namespaced_deployment_scale(
                name=deploy_name,
                namespace=settings.models_namespace,
                body={"spec": {"replicas": 0}},
            )
        except client.exceptions.ApiException:
            pass
        pods = core_api.list_namespaced_pod(
            namespace=settings.models_namespace,
            label_selector=f"serving.kserve.io/inferenceservice={model_name}",
        )
        deleted = 0
        for pod in pods.items:
            core_api.delete_namespaced_pod(
                name=pod.metadata.name,
                namespace=settings.models_namespace,
            )
            deleted += 1
        return (
            f"Shut down '{model_name}' — deleted {deleted} pod(s). "
            f"Model is now scaled to zero."
        )

    return f"Scaled '{model_name}' to minReplicas={min_replicas}. Model will start up shortly."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="The name of the InferenceService to scale.")
    parser.add_argument("--min-replicas", type=int, required=True, help="0 = shut down, 1+ = keep running.")
    args = parser.parse_args()

    try:
        print(scale_model(args.model_name, args.min_replicas))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
