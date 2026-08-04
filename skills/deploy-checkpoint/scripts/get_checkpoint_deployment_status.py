#!/usr/bin/env python3
# ---
# description: >
#   Check progress of a checkpoint deployment started by
#   deploy_checkpoint_model, and advance it to the next stage if the current
#   one has finished. Call this repeatedly until it reports the model
#   deployed -- unlike get_finetune_run_status, nothing else drives this
#   forward automatically.
# parameters:
#   - name: exp-name
#     type: string
#     required: true
#     description: The exp-name passed to deploy_checkpoint_model.
#   - name: model-name
#     type: string
#     required: false
#     default: pi05
#     description: Only 'pi05' is supported so far.
# ---
"""Check progress of a checkpoint deployment started by deploy_checkpoint_model,
and advance it to the next stage if the current one has finished. See
../SKILL.md."""
import argparse
import os

from kubernetes import client

DATASETS_NAMESPACE = os.environ.get("DATASETS_NAMESPACE", "physical-ai")
MODELS_NAMESPACE = os.environ.get("MODELS_NAMESPACE", "physical-ai-models")

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
CHECKPOINT_DEPLOYMENT_LABEL = "physical-ai.io/checkpoint-deployment"
CHECKPOINT_EXPORT_PORT = 8080
_SUPPORTED_MODELS = ("pi05",)

# Also defined in the fine-tuning skill's finetune_pipeline.py -- duplicated
# rather than imported cross-skill, so this script has no dependency on
# another skill being installed.
GPU_NODE_SELECTOR_KEY = "nvidia.com/gpu.product"
GPU_NODE_SELECTOR_VALUE = "NVIDIA-L40S"

# lerobot-train's PreTrainedPolicy.save_pretrained() serializes the whole
# PI05Policy wrapper (self.model = PI05Pytorch(...)), so every weight key
# comes out prefixed "model." -- openpi-runtime's native loader needs the
# bare, unprefixed keys instead. Both the prefix and this one dropped key
# were confirmed against a real checkpoint's actual
# safetensors.torch.load_model error.
CHECKPOINT_KEY_PREFIX = "model."
CHECKPOINT_TIED_KEYS_TO_DROP = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
)

# Same URL platform/base/models/pi05/model-download-job.yaml fetches for the
# base model -- reused here since that's openpi's own documented mechanism
# for exactly this case (fine-tuning on the same base robot/dataset family).
NORM_STATS_URL = "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json"
NORM_STATS_ASSET_PATH = "assets/droid/norm_stats.json"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()


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


def _start_import_job(batch_api, exp_name: str, isvc_name: str, export_pod_ip: str) -> str:
    import_job_name = _import_job_name(isvc_name)
    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}

    base_url = f"http://{export_pod_ip}:{CHECKPOINT_EXPORT_PORT}/"
    import_script = f"""\
set -e
python3 << 'PYEOF'
import glob
import json
import os
import re
import struct
import urllib.parse
import urllib.request

BASE_URL = "{base_url}"
DEST_DIR = "/mnt/models"
KEY_PREFIX = {CHECKPOINT_KEY_PREFIX!r}
TIED_KEYS_TO_DROP = {CHECKPOINT_TIED_KEYS_TO_DROP!r}
NORM_STATS_URL = {NORM_STATS_URL!r}
NORM_STATS_ASSET_PATH = {NORM_STATS_ASSET_PATH!r}


def crawl(url, dest):
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode()
    for href in re.findall(r'href="([^"]+)"', html):
        if href in ("../", "./"):
            continue
        name = urllib.parse.unquote(href)
        child_url = url + href
        child_dest = os.path.join(dest, name.rstrip("/"))
        if href.endswith("/"):
            os.makedirs(child_dest, exist_ok=True)
            crawl(child_url, child_dest)
        else:
            os.makedirs(os.path.dirname(child_dest) or ".", exist_ok=True)
            urllib.request.urlretrieve(child_url, child_dest)


def rewrite_checkpoint_keys(path):
    # Strips lerobot-train's "model." wrapper prefix and drops one tied
    # embedding weight that isn't a separate parameter in openpi-runtime's
    # bare PI0Pytorch module. Dropping a tensor entry from the header alone
    # isn't enough -- safetensors' own deserializer validates that
    # data_offsets tile the data section contiguously with no gaps, so this
    # rebuilds the whole file: a fresh compact header with recomputed
    # offsets, and the data section rewritten by streaming only the KEPT
    # tensors' byte ranges across in their original order, skipping the
    # dropped one entirely.
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len

    metadata = header.pop("__metadata__", None)
    changed = False
    entries = []
    for key, meta in header.items():
        new_key = key[len(KEY_PREFIX):] if key.startswith(KEY_PREFIX) else key
        if new_key != key:
            changed = True
        if new_key in TIED_KEYS_TO_DROP:
            changed = True
            continue
        entries.append((new_key, meta))

    if not changed:
        return

    entries.sort(key=lambda item: item[1]["data_offsets"][0])

    new_header = {{}}
    if metadata is not None:
        new_header["__metadata__"] = metadata
    cursor = 0
    for new_key, meta in entries:
        start, end = meta["data_offsets"]
        length = end - start
        new_header[new_key] = {{"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [cursor, cursor + length]}}
        cursor += length

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")

    tmp_path = path + ".rewrite.tmp"
    chunk_size = 64 * 1024 * 1024
    with open(path, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(struct.pack("<Q", len(new_header_bytes)))
        dst.write(new_header_bytes)
        for new_key, meta in entries:
            start, end = meta["data_offsets"]
            src.seek(data_start + start)
            remaining = end - start
            while remaining > 0:
                chunk = src.read(min(chunk_size, remaining))
                if not chunk:
                    raise RuntimeError(f"unexpected EOF copying tensor {{new_key}} from {{path}}")
                dst.write(chunk)
                remaining -= len(chunk)

    os.replace(tmp_path, path)
    print(f"Rewrote checkpoint keys in {{path}} (stripped '{{KEY_PREFIX}}' prefix, dropped tied keys).")


def fetch_norm_stats():
    # lerobot-train checkpoints don't produce this file at all -- openpi-
    # runtime's server needs it separately. Reusing the base checkpoint's own
    # norm_stats.json, per openpi's own documented mechanism for this case.
    dest = os.path.join(DEST_DIR, NORM_STATS_ASSET_PATH)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(NORM_STATS_URL, dest)
    print(f"Fetched norm stats to {{dest}}.")


crawl(BASE_URL, DEST_DIR)
for safetensors_path in glob.glob(os.path.join(DEST_DIR, "*.safetensors")):
    rewrite_checkpoint_keys(safetensors_path)
fetch_norm_stats()
print("Checkpoint copy complete.")
PYEOF
"""
    try:
        batch_api.create_namespaced_job(
            namespace=MODELS_NAMESPACE,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": import_job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 2,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "import",
                                    "image": "python:3.11-slim",
                                    "command": ["/bin/bash", "-c", import_script],
                                    "volumeMounts": [{"name": "model-cache", "mountPath": "/mnt/models"}],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "model-cache",
                                    "persistentVolumeClaim": {"claimName": model_cache_pvc},
                                }
                            ],
                        },
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to start checkpoint import Job '{import_job_name}': {e.reason}"

    return f"Export reachable -- copying checkpoint '{exp_name}' into the models namespace now ('{import_job_name}')."


def _create_checkpoint_inference_service(custom_api, core_api, exp_name: str, isvc_name: str) -> str:
    triton_cache_pvc = _triton_cache_pvc_name(isvc_name)
    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}

    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=MODELS_NAMESPACE,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": triton_cache_pvc, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "1Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create triton-cache PVC '{triton_cache_pvc}': {e.reason}"

    isvc_body = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": MODELS_NAMESPACE,
            "labels": {
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/genai-asset": "true",
                **labels,
            },
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "serving.kserve.io/autoscalerClass": "external",
                "sidecar.istio.io/inject": "false",
                "physical-ai.io/output-kind": "unsupported",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": 0,
                "deploymentStrategy": {"type": "Recreate"},
                "nodeSelector": {GPU_NODE_SELECTOR_KEY: GPU_NODE_SELECTOR_VALUE},
                "volumes": [
                    {"name": "triton-cache", "persistentVolumeClaim": {"claimName": triton_cache_pvc}}
                ],
                "model": {
                    "modelFormat": {"name": "pytorch"},
                    "runtime": "openpi-runtime",
                    "storageUri": f"pvc://{model_cache_pvc}",
                    "resources": {
                        "requests": {"cpu": "2", "memory": "24Gi", "nvidia.com/gpu": "1"},
                        "limits": {"cpu": "4", "memory": "48Gi", "nvidia.com/gpu": "1"},
                    },
                },
            },
        },
    }
    try:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=MODELS_NAMESPACE,
            plural="inferenceservices",
            body=isvc_body,
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create InferenceService '{isvc_name}': {e.reason}"

    scaler_name = _scaler_name(isvc_name)
    scaler_body = {
        "apiVersion": "http.keda.sh/v1alpha1",
        "kind": "HTTPScaledObject",
        "metadata": {"name": scaler_name, "namespace": MODELS_NAMESPACE, "labels": labels},
        "spec": {
            "hosts": [f"{isvc_name}-predictor.{MODELS_NAMESPACE}.svc.cluster.local"],
            "targetPendingRequests": 1,
            "scaleTargetRef": {
                "name": f"{isvc_name}-predictor",
                "kind": "Deployment",
                "apiVersion": "apps/v1",
                "service": f"{isvc_name}-predictor",
                "port": 80,
            },
            "replicas": {"min": 0, "max": 1},
            "scaledownPeriod": 3600,
        },
    }
    try:
        custom_api.create_namespaced_custom_object(
            group="http.keda.sh",
            version="v1alpha1",
            namespace=MODELS_NAMESPACE,
            plural="httpscaledobjects",
            body=scaler_body,
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"InferenceService '{isvc_name}' created, but failed to create its HTTPScaledObject: {e.reason}"

    return (
        f"Deployed '{isvc_name}' -- scale-to-zero, currently at 0 replicas. "
        f"Predictor: {isvc_name}-predictor.{MODELS_NAMESPACE}.svc.cluster.local. "
        f"Call scale_model('{isvc_name}', 1) to warm it up for testing, and "
        f"takedown_checkpoint_model('{exp_name}') when you're done comparing."
    )


def get_checkpoint_deployment_status(exp_name: str, model_name: str = "pi05") -> str:
    """Check progress of a checkpoint deployment started by
    deploy_checkpoint_model, and advance it to the next stage if the current
    one has finished.

    Call this repeatedly until it reports the model deployed -- unlike
    get_finetune_run_status, nothing else drives this forward automatically.
    Each call both checks status and, if ready, starts the next stage
    (creating the import Job once the export Job's pod is reachable, then
    the InferenceService once the import Job succeeds).
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    core_api, batch_api, custom_api = _get_clients()

    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=MODELS_NAMESPACE,
            plural="inferenceservices",
            name=isvc_name,
        )
        already_deployed = True
    except client.exceptions.ApiException as e:
        if e.status != 404:
            return f"Could not read InferenceService '{isvc_name}': {e.reason}"
        already_deployed = False

    if already_deployed:
        pods = core_api.list_namespaced_pod(
            namespace=MODELS_NAMESPACE,
            label_selector=f"serving.kserve.io/inferenceservice={isvc_name}",
        )
        status = _live_pod_status(pods.items)
        return (
            f"'{isvc_name}' is deployed (scale-to-zero) at "
            f"{isvc_name}-predictor.{MODELS_NAMESPACE}.svc.cluster.local -- {status}. "
            f"Call scale_model('{isvc_name}', 1) to warm it up for testing."
        )

    import_job_name = _import_job_name(isvc_name)
    try:
        import_job = batch_api.read_namespaced_job(name=import_job_name, namespace=MODELS_NAMESPACE)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            return f"Could not read import Job '{import_job_name}': {e.reason}"
        import_job = None

    if import_job is not None:
        if import_job.status.succeeded:
            try:
                batch_api.delete_namespaced_job(
                    name=_export_job_name(exp_name),
                    namespace=DATASETS_NAMESPACE,
                    propagation_policy="Background",
                )
            except client.exceptions.ApiException:
                pass
            return _create_checkpoint_inference_service(custom_api, core_api, exp_name, isvc_name)
        if import_job.status.failed:
            return (
                f"Checkpoint copy for '{exp_name}' failed (import Job '{import_job_name}'). "
                f"Check its pod logs (see the models skill's GETTING LOGS steps), then call "
                f"deploy_checkpoint_model again after fixing the issue."
            )
        return f"Copying checkpoint '{exp_name}' into the models namespace ('{import_job_name}' still running)."

    export_job_name = _export_job_name(exp_name)
    export_pods = core_api.list_namespaced_pod(
        namespace=DATASETS_NAMESPACE, label_selector=f"job-name={export_job_name}"
    )
    export_pod = next(
        (p for p in export_pods.items if p.status.phase == "Running" and p.status.pod_ip), None
    )
    if export_pod is None:
        return f"Export Job '{export_job_name}' hasn't started serving yet -- try again shortly."

    return _start_import_job(batch_api, exp_name, isvc_name, export_pod.status.pod_ip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, help="The exp_name passed to deploy_checkpoint_model.")
    parser.add_argument("--model-name", default="pi05", help="Only 'pi05' is supported so far.")
    args = parser.parse_args()

    try:
        print(get_checkpoint_deployment_status(args.exp_name, args.model_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
