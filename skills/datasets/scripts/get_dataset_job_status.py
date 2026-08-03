#!/usr/bin/env python3
"""Check the status of a dataset download started by pull_dataset. See ../SKILL.md."""
import argparse

from kubernetes import client

from platform_agent.config import settings


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api()


def get_dataset_job_status(dataset_name: str) -> str:
    """Check the status of a dataset download started by pull_dataset:
    whether the Job succeeded/failed/is still running, and the backing PVC's
    bound state.
    """
    core_api, batch_api = _get_clients()
    pvc_name = f"dataset-{dataset_name}-pvc"
    job_name = f"download-{dataset_name}-dataset"

    try:
        # read_namespaced_job (not _status) -- the /status subresource needs
        # separate RBAC from the base "jobs" resource we're actually granted;
        # the plain read already returns the full object including .status.
        job = batch_api.read_namespaced_job(name=job_name, namespace=settings.datasets_namespace)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No download Job '{job_name}' found — has pull_dataset been called for '{dataset_name}'?"
        return f"Could not read Job '{job_name}': {e.reason}"

    status = job.status
    if status.succeeded:
        state = "succeeded"
    elif status.failed:
        state = "failed"
    elif status.active:
        state = "running"
    else:
        state = "pending"

    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=settings.datasets_namespace)
        pvc_phase = pvc.status.phase
    except client.exceptions.ApiException:
        pvc_phase = "unknown"

    result = f"Dataset '{dataset_name}': download Job is {state}, PVC '{pvc_name}' is {pvc_phase}."

    if state == "failed":
        pods = core_api.list_namespaced_pod(
            namespace=settings.datasets_namespace,
            label_selector=f"job-name={job_name}",
        )
        if pods.items:
            try:
                logs = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=settings.datasets_namespace,
                    tail_lines=30,
                )
                result += f"\nLast 30 log lines:\n{logs}"
            except client.exceptions.ApiException:
                pass
    elif state == "succeeded":
        result += " Ready to reference by PVC name in a fine-tuning pipeline."

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="The --dataset-name passed to pull_dataset.")
    args = parser.parse_args()

    try:
        print(get_dataset_job_status(args.dataset_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
