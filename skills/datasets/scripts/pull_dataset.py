#!/usr/bin/env python3
"""Download a Hugging Face dataset onto the cluster so a fine-tuning job can
read it. See ../SKILL.md."""
import argparse

from kubernetes import client

from platform_agent.config import settings

DATASET_CACHE_LABEL = "physical-ai.io/dataset-cache"
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


def pull_dataset(dataset_repo_id: str, dataset_name: str, pvc_size_gb: int = 50) -> str:
    """Download a Hugging Face dataset onto the cluster so a fine-tuning job
    can read it. This consumes real shared-cluster storage — only call this
    after calling get_dataset_info and showing the user its size and license,
    and after the user has explicitly said to proceed. Never call this
    speculatively or as the first response to "find me a dataset for X".

    Downloads the entire dataset repo (config/subset selection happens at
    fine-tuning time via submit_finetune_run's dataset_subset, not download
    time) -- including for a repo that bundles several independent LeRobot
    datasets as subfolders rather than one dataset per repo. Pulling once
    here and picking a subset per fine-tuning run means one PVC serves every
    subset in the repo, instead of needing a separate PVC (and a separate,
    redundant download) per subset. Creates a PVC and a Kubernetes Job in
    the datasets namespace that runs huggingface_hub.snapshot_download.
    Check progress with get_dataset_job_status afterward — this returns as
    soon as the Job is created, not once the download finishes.

    Falls back to a plain `git clone` + `git lfs pull` of the same repo if
    snapshot_download keeps hitting HTTP 429 after a few retries (confirmed
    live: a repo subset with ~53k files exhausted the 1000 req/5min
    authenticated rate limit after only ~6.6k files, since snapshot_download
    makes a HEAD+GET pair per file; git-lfs instead fetches object URLs via
    the LFS batch API in ~100-object batches, cutting total requests by
    ~2 orders of magnitude). The fallback clones into a scratch dir and
    copies the checked-out tree into local_dir, so the end result is
    indistinguishable from a snapshot_download-produced PVC either way.
    """
    core_api, batch_api = _get_clients()
    pvc_name = f"dataset-{dataset_name}-pvc"
    job_name = f"download-{dataset_name}-dataset"
    labels = {DATASET_CACHE_LABEL: "true", DATASET_REPO_LABEL: dataset_repo_id.replace("/", "--")}

    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": pvc_name, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": f"{pvc_size_gb}Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create PVC '{pvc_name}': {e.reason}"

    # Not an f-string / .format() call: the git fallback below is full of
    # literal ${VAR} bash syntax that would otherwise collide with brace
    # interpolation. dataset_repo_id is spliced in via .replace() instead.
    download_script = """\
set -uo pipefail

python3 - <<'PYEOF'
import os, re, sys, time
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

token = os.getenv("HF_TOKEN")
repo_id = "__DATASET_REPO_ID__"

# snapshot_download can swallow a rate-limit error itself: if it can't reach
# the repo but local_dir already has *something* in it (e.g. a partial file
# from a prior attempt sharing this PVC), it logs a warning and returns the
# existing directory as-is instead of raising -- so a bare try/except around
# it can't tell "downloaded everything" from "downloaded nothing, gave up
# quietly". Compare against the repo's real file count instead of trusting
# a clean return.
try:
    expected_files = len(HfApi().list_repo_files(repo_id, repo_type="dataset", token=token))
except Exception as e:
    print(f"could not list repo files up front ({e}) -- skipping completeness check", flush=True)
    expected_files = None


def local_file_count():
    total = 0
    for root, _dirs, files in os.walk("/mnt/dataset"):
        if root == "/mnt/dataset/.cache" or root.startswith("/mnt/dataset/.cache/"):
            continue
        total += len(files)
    return total


max_attempts = 3
for attempt in range(1, max_attempts + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir="/mnt/dataset",
            token=token,
            max_workers=4,
        )
        actual_files = local_file_count()
        if expected_files is not None and actual_files < expected_files * 0.95:
            print(
                f"snapshot_download returned without error but only "
                f"{actual_files}/{expected_files} files are present -- treating as "
                f"incomplete (likely its own silent existing-local-dir fallback, not "
                f"a real success)",
                flush=True,
            )
        else:
            print(f"SNAPSHOT_DOWNLOAD_OK ({actual_files} files)", flush=True)
            sys.exit(0)
    except HfHubHTTPError as e:
        wait = 90
        m = re.search(r"Retry after (\\d+) seconds", str(e))
        if m:
            wait = int(m.group(1)) + 10
        print(f"snapshot_download attempt {attempt}/{max_attempts} failed: {e}", flush=True)
        if attempt < max_attempts:
            print(f"sleeping {wait}s before retry", flush=True)
            time.sleep(wait)
            continue
    if attempt < max_attempts:
        print(f"sleeping 90s before retry", flush=True)
        time.sleep(90)
print("SNAPSHOT_DOWNLOAD_EXHAUSTED -- falling back to git+lfs", flush=True)
sys.exit(1)
PYEOF
snapshot_rc=$?

if [ "$snapshot_rc" -ne 0 ]; then
    # snapshot_download does a HEAD+GET per file, so it burns through HF's
    # per-token rate limit fast on repos with tens of thousands of files.
    # git-lfs instead resolves object URLs via the LFS batch API in ~100-
    # object batches -- ~2 orders of magnitude fewer requests for the same
    # content. git itself is preinstalled on this image; git-lfs isn't, so
    # fetch its static binary into a user-writable dir (this image's
    # restricted-SCC UID can't apt-get install into /usr).
    set -e
    mkdir -p /tmp/bin
    export PATH="/tmp/bin:$PATH"
    curl -sL -m 60 -o /tmp/git-lfs.tar.gz \\
        https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
    tar -xzf /tmp/git-lfs.tar.gz -C /tmp
    cp /tmp/git-lfs-*/git-lfs /tmp/bin/git-lfs
    chmod +x /tmp/bin/git-lfs
    git lfs install --skip-smudge
    git config --global --add safe.directory '*'
    rm -rf /tmp/repo
    # Auth via an explicit header (not a token-in-URL) so it never shows up
    # in `git remote -v` or error output. -c only applies to this one clone;
    # persist the same header into the new repo's own config afterward so
    # `git lfs pull` (a separate process) picks it up too.
    GIT_LFS_SKIP_SMUDGE=1 git -c http.extraHeader="Authorization: Bearer ${HF_TOKEN}" \\
        clone --depth 1 "https://huggingface.co/datasets/__DATASET_REPO_ID__" /tmp/repo
    git -C /tmp/repo config http.extraHeader "Authorization: Bearer ${HF_TOKEN}"
    git -C /tmp/repo lfs pull
    find /tmp/repo -mindepth 1 -maxdepth 1 ! -name .git -exec cp -a {} /mnt/dataset/ \\;
    rm -rf /tmp/repo /mnt/dataset/.cache
    echo "DOWNLOAD_COMPLETE (git+lfs fallback)"
else
    echo "DOWNLOAD_COMPLETE (snapshot_download)"
fi
""".replace("__DATASET_REPO_ID__", dataset_repo_id)

    try:
        batch_api.create_namespaced_job(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 3,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "downloader",
                                    # Also used by convert_dataset_to_v3 -- already has git +
                                    # python3 + huggingface_hub preinstalled, so the fallback
                                    # below only needs to fetch git-lfs itself, and the primary
                                    # snapshot_download path needs no pip install step at all.
                                    "image": LEROBOT_IMAGE,
                                    "command": ["/bin/bash", "-c", download_script],
                                    "env": [
                                        {
                                            "name": "HF_TOKEN",
                                            "valueFrom": {
                                                "secretKeyRef": {"name": "huggingface-token", "key": "HF_TOKEN"}
                                            },
                                        },
                                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                                        # OpenShift's restricted SCC runs this container as an
                                        # arbitrary non-root UID with no /etc/passwd entry, so
                                        # $HOME resolves to something unwritable (e.g. "/") --
                                        # both huggingface_hub and git need a writable HOME for
                                        # their caches/config.
                                        {"name": "HOME", "value": "/tmp"},
                                        # Confirmed live: a repo with 100k+ small files (one per
                                        # episode/camera) hit HTTP 429 from HF's xet-read-token
                                        # endpoint within ~2 minutes at snapshot_download's default
                                        # concurrency, twice in a row -- disabling xet falls back
                                        # to plain HTTP/LFS downloads, which don't hit that
                                        # specific rate limit.
                                        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
                                    ],
                                    "volumeMounts": [{"name": "dataset-storage", "mountPath": "/mnt/dataset"}],
                                    "resources": {
                                        "requests": {"cpu": "2", "memory": "4Gi"},
                                        "limits": {"cpu": "4", "memory": "8Gi"},
                                    },
                                }
                            ],
                            "volumes": [{"name": "dataset-storage", "persistentVolumeClaim": {"claimName": pvc_name}}],
                        }
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return (
                f"Job '{job_name}' already exists — a pull for '{dataset_name}' "
                f"is already in progress or complete. Check get_dataset_job_status."
            )
        return f"Failed to create download Job '{job_name}': {e.reason}"

    return (
        f"Started downloading '{dataset_repo_id}' into PVC '{pvc_name}' via "
        f"Job '{job_name}' in namespace '{settings.datasets_namespace}'. "
        f"Call get_dataset_job_status('{dataset_name}') to check progress."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True, help="HF dataset repo id, e.g. 'GEAR-Dreams/DreamZero-DROID'.")
    parser.add_argument("--dataset-name", required=True, help="Short name for this staged dataset (K8s resource name prefix).")
    parser.add_argument("--pvc-size-gb", type=int, default=50)
    args = parser.parse_args()

    try:
        print(pull_dataset(args.dataset_repo_id, args.dataset_name, args.pvc_size_gb))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
