#!/usr/bin/env python3
# ---
# description: >
#   Convert an already-staged LeRobot dataset from v2.1 to v3.0 format, in
#   place on its existing PVC. Only call this after a fine-tuning run's train
#   stage has actually failed with a dataset-format error -- check
#   get_finetune_run_status's stage logs first. Runs as a Kubernetes Job, no
#   GPU needed. Check progress with get_dataset_conversion_status afterward.
# parameters:
#   - name: dataset-pvc-name
#     type: string
#     required: true
#     description: PVC name of an already-pull_dataset-staged dataset.
# ---
"""Convert an already-staged LeRobot dataset from v2.1 to v3.0 format, in
place on its existing PVC. See ../SKILL.md."""
import argparse
import os

from kubernetes import client

DATASETS_NAMESPACE = os.environ.get("DATASETS_NAMESPACE", "physical-ai")

DATASET_REPO_LABEL = "physical-ai.io/dataset-repo"

# Also defined in the fine-tuning skill's finetune_recipes.py -- duplicated
# rather than imported cross-skill, so this script has no dependency on
# another skill being installed (see the fine-tuning skill for the training
# image this same tag is used for).
LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api()


def dataset_repo_id_from_pvc(pvc) -> str | None:
    """Recovers the HF dataset repo id a staged PVC was pulled from, via the
    DATASET_REPO_LABEL pull_dataset sets (slashes get swapped for "--" since
    K8s label values can't contain "/"). Returns None if the PVC isn't
    labeled as a dataset cache -- e.g. it wasn't created by pull_dataset.
    """
    label = (pvc.metadata.labels or {}).get(DATASET_REPO_LABEL)
    return label.replace("--", "/") if label else None


def _conversion_job_name(dataset_pvc_name: str) -> str:
    return f"convert-{dataset_pvc_name}-v3"


def convert_dataset_to_v3(dataset_pvc_name: str) -> str:
    """Convert an already-staged LeRobot dataset from v2.1 to v3.0 format, in
    place on its existing PVC. Only call this after a fine-tuning run's train
    stage has actually failed with a dataset-format error (e.g.
    BackwardCompatibilityError, or log lines mentioning "v2.1"/"v3.0") --
    check get_finetune_run_status's stage logs first. lerobot-train and this
    platform's fine-tuning recipes only support v3.0 datasets.

    Runs `python -m lerobot.scripts.convert_dataset_v21_to_v30` as a
    Kubernetes Job against the dataset's existing PVC, no GPU needed. Check
    progress with get_dataset_conversion_status afterward -- this returns
    as soon as the Job is created.
    """
    core_api, batch_api = _get_clients()

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

    job_name = _conversion_job_name(dataset_pvc_name)
    convert_script = """\
set -e
export HOME=/tmp
# convert_dataset_v21_to_v30 converts --root in place, but uses a `<root>_v30`
# sibling directory as scratch space while doing so (confirmed live: pointing
# --root directly at the PVC mount gets a PermissionError trying to create
# that sibling under /mnt itself, which OpenShift's restricted SCC --
# arbitrary non-root UID -- can't write to). So nest the original content one
# level inside the PVC mount first, giving the scratch dir room to exist
# inside the still-writable PVC too. The script also leaves its own internal
# v2.1 backup as a SIBLING of --root, at <root>_old (confirmed live -- not
# nested inside --root, despite it looking that way on an earlier inspection;
# that turned out to be a retry re-processing an already-converted directory,
# whose own first move-into-place step swept the previous attempt's stray
# sibling backup inside too).
#
# Restore EVERYTHING from the scratch dir back to the PVC root on ANY exit --
# not just an allowlist of known LeRobot dirs (data/meta/videos), which used
# to silently rm -rf any other top-level file (README.md, .gitattributes,
# ...) along with the scratch dir on every successful conversion. This same
# trap also fires when the conversion itself FAILS (set -e triggers the EXIT
# trap on any nonzero exit): it falls back to lerobot's own <root>_old
# pre-conversion backup if the scratch dir ends up empty, so the PVC is left
# with a working dataset at the root path every other tool expects, instead
# of stuck nested and unusable with no repair tool available.
trap '
    set +e
    restore_from=/mnt/dataset/_v21_orig
    if [ ! -d "$restore_from" ] || [ -z "$(ls -A "$restore_from" 2>/dev/null)" ]; then
        restore_from=/mnt/dataset/_v21_orig_old
    fi
    if [ -d "$restore_from" ]; then
        find "$restore_from" -mindepth 1 -maxdepth 1 -exec mv {{}} /mnt/dataset/ \\;
    fi
    rm -rf /mnt/dataset/_v21_orig /mnt/dataset/_v21_orig_old
' EXIT
mkdir -p /mnt/dataset/_v21_orig
find /mnt/dataset -mindepth 1 -maxdepth 1 ! -name _v21_orig -exec mv {{}} /mnt/dataset/_v21_orig/ \\;
python -m lerobot.scripts.convert_dataset_v21_to_v30 \\
    --repo-id={dataset_repo_id} \\
    --root=/mnt/dataset/_v21_orig \\
    --push-to-hub=false
""".format(dataset_repo_id=dataset_repo_id)

    try:
        batch_api.create_namespaced_job(
            namespace=DATASETS_NAMESPACE,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": job_name, "labels": {DATASET_REPO_LABEL: dataset_repo_id.replace("/", "--")}},
                "spec": {
                    "backoffLimit": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "convert",
                                    "image": LEROBOT_IMAGE,
                                    "command": ["/bin/bash", "-c", convert_script],
                                    "env": [
                                        {
                                            "name": "HF_TOKEN",
                                            "valueFrom": {
                                                "secretKeyRef": {"name": "huggingface-token", "key": "HF_TOKEN"}
                                            },
                                        },
                                        # The image bakes in a non-writable default HF_HOME
                                        # (confirmed live: the conversion script's episodes-metadata
                                        # step failed with PermissionError writing to
                                        # /home/user_lerobot/.cache -- an inline `export HOME=/tmp`
                                        # in the script doesn't override it, since the `datasets`
                                        # library resolves its cache dir from HF_HOME directly, not
                                        # from $HOME).
                                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                                        {"name": "HOME", "value": "/tmp"},
                                    ],
                                    "volumeMounts": [{"name": "dataset-storage", "mountPath": "/mnt/dataset"}],
                                    "resources": {
                                        "requests": {"cpu": "2", "memory": "4Gi"},
                                        "limits": {"cpu": "4", "memory": "8Gi"},
                                    },
                                }
                            ],
                            "volumes": [
                                {"name": "dataset-storage", "persistentVolumeClaim": {"claimName": dataset_pvc_name}}
                            ],
                        }
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return (
                f"Job '{job_name}' already exists — a conversion for '{dataset_pvc_name}' "
                f"is already in progress or complete. Check get_dataset_conversion_status."
            )
        return f"Failed to create conversion Job '{job_name}': {e.reason}"

    return (
        f"Started converting '{dataset_repo_id}' (PVC '{dataset_pvc_name}') to v3.0 via "
        f"Job '{job_name}'. Call get_dataset_conversion_status('{dataset_pvc_name}') to check progress."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pvc-name", required=True, help="PVC name of an already-pull_dataset-staged dataset.")
    args = parser.parse_args()

    try:
        print(convert_dataset_to_v3(args.dataset_pvc_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
