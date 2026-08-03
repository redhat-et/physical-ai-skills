#!/usr/bin/env python3
"""Check the status of a v2.1-to-v3.0 dataset conversion started by
convert_dataset_to_v3. See ../SKILL.md."""
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


def _conversion_job_name(dataset_pvc_name: str) -> str:
    return f"convert-{dataset_pvc_name}-v3"


def get_dataset_conversion_status(dataset_pvc_name: str) -> str:
    """Check the status of a v2.1-to-v3.0 dataset conversion started by
    convert_dataset_to_v3: whether the Job succeeded/failed/is still running.
    """
    core_api, batch_api = _get_clients()
    job_name = _conversion_job_name(dataset_pvc_name)

    try:
        job = batch_api.read_namespaced_job(name=job_name, namespace=settings.datasets_namespace)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No conversion Job '{job_name}' found — has convert_dataset_to_v3 been called for '{dataset_pvc_name}'?"
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

    result = f"Dataset conversion for '{dataset_pvc_name}': Job is {state}."

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
        result += " Dataset is now v3.0 -- retry submit_finetune_run with the same dataset_pvc_name."

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pvc-name", required=True, help="The --dataset-pvc-name passed to convert_dataset_to_v3.")
    args = parser.parse_args()

    try:
        print(get_dataset_conversion_status(args.dataset_pvc_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
