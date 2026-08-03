#!/usr/bin/env python3
"""List GPU capacity on the cluster, grouped by GPU product type. See
../SKILL.md."""
import argparse

from kubernetes import client

from platform_agent.config import settings

# VRAM per GPU product, in GB. Deliberately small and explicit rather than
# guessed — add an entry here when a new GPU type is added to the cluster.
GPU_VRAM_GB = {
    "NVIDIA-L40S": 48,
}


def _get_core_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def list_cluster_gpus() -> str:
    """List GPU capacity on the cluster, grouped by GPU product type: total
    GPUs, how many are currently in use by running model pods, and VRAM per
    GPU where known. Use this before recommending hardware for a new model
    deployment.
    """
    core_api = _get_core_client()

    nodes = core_api.list_node()
    capacity = {}
    node_counts = {}
    for node in nodes.items:
        labels = node.metadata.labels or {}
        product = labels.get("nvidia.com/gpu.product")
        if not product:
            continue
        gpu_count = int((node.status.allocatable or {}).get("nvidia.com/gpu", "0"))
        if gpu_count == 0:
            continue
        capacity[product] = capacity.get(product, 0) + gpu_count
        node_counts[product] = node_counts.get(product, 0) + 1

    if not capacity:
        return "No GPU nodes found on the cluster (no nodes with an nvidia.com/gpu.product label)."

    pods = core_api.list_namespaced_pod(namespace=settings.models_namespace)
    in_use = {}
    for pod in pods.items:
        if pod.status.phase not in ("Running", "Pending"):
            continue
        product = (pod.spec.node_selector or {}).get("nvidia.com/gpu.product")
        if not product:
            continue
        gpu_count = 0
        for container in pod.spec.containers:
            requests = (container.resources.requests or {}) if container.resources else {}
            gpu_count += int(requests.get("nvidia.com/gpu", "0"))
        if gpu_count:
            in_use[product] = in_use.get(product, 0) + gpu_count

    lines = []
    for product, total in sorted(capacity.items()):
        vram = GPU_VRAM_GB.get(product)
        vram_str = f"{vram}GB VRAM" if vram else "VRAM unknown"
        used = in_use.get(product, 0)
        lines.append(
            f"- {product}: {vram_str}, {total} total on {node_counts[product]} "
            f"node(s), {used} in use, {total - used} free"
        )
    return "GPU capacity:\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        print(list_cluster_gpus())
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
