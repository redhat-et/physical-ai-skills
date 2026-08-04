#!/usr/bin/env python3
# ---
# description: >
#   List fine-tuned checkpoints currently deployed as live comparison
#   endpoints (via deploy_checkpoint_model), with real-time status. Separate
#   from the models skill's LISTING MODELS steps (permanent catalog models)
#   and list_finetune_runs (checkpoints that exist but may not be deployed
#   anywhere).
# parameters: []
# ---
"""List fine-tuned checkpoints currently deployed as live comparison
endpoints. See ../SKILL.md."""
import argparse
import os

from kubernetes import client

MODELS_NAMESPACE = os.environ.get("MODELS_NAMESPACE", "physical-ai-models")

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
CHECKPOINT_DEPLOYMENT_LABEL = "physical-ai.io/checkpoint-deployment"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.CustomObjectsApi()


def _live_pod_status(pods: list) -> str:
    """Same pod-based ground truth as get_model_readiness() (tools/models.py
    in the platform-agent repo) -- the ISVC's own Ready condition is
    misleading for scale-to-zero models (KServe reports Ready=True even at
    zero replicas), so status must come from actual pods.
    """
    if not pods:
        return "scaled to zero (no pods running)"
    ready_count = sum(
        1
        for p in pods
        if p.status.container_statuses and all(cs.ready for cs in p.status.container_statuses)
    )
    if ready_count > 0:
        return f"running ({ready_count}/{len(pods)} pod(s) ready)"
    return f"starting ({len(pods)} pod(s) not ready yet)"


def list_checkpoint_deployments() -> str:
    """List fine-tuned checkpoints currently deployed as live comparison
    endpoints (via deploy_checkpoint_model), with real-time status. Separate
    from the models skill's LISTING MODELS steps (permanent catalog models)
    and list_finetune_runs (checkpoints that exist but may not be deployed
    anywhere)."""
    core_api, custom_api = _get_clients()
    items = custom_api.list_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=MODELS_NAMESPACE,
        plural="inferenceservices",
        label_selector=CHECKPOINT_DEPLOYMENT_LABEL,
    )

    isvcs = items.get("items", [])
    if not isvcs:
        return "No checkpoint deployments found."

    all_pods = core_api.list_namespaced_pod(namespace=MODELS_NAMESPACE)
    pods_by_isvc: dict[str, list] = {}
    for pod in all_pods.items:
        name = (pod.metadata.labels or {}).get("serving.kserve.io/inferenceservice")
        if name:
            pods_by_isvc.setdefault(name, []).append(pod)

    lines = []
    for isvc in isvcs:
        name = isvc["metadata"]["name"]
        exp_name = isvc["metadata"].get("labels", {}).get(FINETUNE_EXP_LABEL, "unknown")
        status = _live_pod_status(pods_by_isvc.get(name, []))
        lines.append(f"- {name} (exp_name={exp_name}): {status}")

    return "Checkpoint deployments:\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        print(list_checkpoint_deployments())
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
