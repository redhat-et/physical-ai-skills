import math
import re

from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings

# VRAM per GPU product, in GB. Deliberately small and explicit rather than
# guessed — add an entry here when a new GPU type is added to the cluster.
GPU_VRAM_GB = {
    "NVIDIA-L40S": 48,
}

BYTES_PER_PARAM = {
    "F32": 4, "FP32": 4,
    "F16": 2, "FP16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "FP8": 1,
    "I4": 0.5, "INT4": 0.5,
}

# Rough overhead multiplier for activations/KV-cache on top of raw weight
# size. A heuristic, not a guarantee — always leave headroom beyond this.
FOOTPRINT_OVERHEAD_FACTOR = 1.2

# Fraction of a GPU's VRAM assumed usable once framework/runtime overhead is
# accounted for, when sizing tensor-parallel-size.
GPU_UTILIZATION_HEADROOM = 0.85


def _get_core_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


@tool
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


@tool
def estimate_model_footprint(
    hf_repo_id: str,
    dtype: str = "auto",
    gpu_product: str = "NVIDIA-L40S",
) -> str:
    """Estimate the GPU memory footprint of a Hugging Face model and how many
    of a given GPU type it would need. Reads parameter count from the
    model's safetensors metadata — no weights are downloaded. Call this
    before generate_model_manifests to pick a tensor_parallel_size, and call
    list_cluster_gpus first to know what GPU types/capacity are actually
    available.

    Args:
        hf_repo_id: Hugging Face repo id, e.g. 'Qwen/Qwen3-8B'.
        dtype: Weight dtype to size for, e.g. 'BF16', 'FP8', 'INT4'. Defaults
            to 'auto', which uses whichever dtype the model's safetensors
            metadata reports the most parameters in.
        gpu_product: GPU type to size against — see list_cluster_gpus for
            what's actually available on this cluster. Defaults to
            'NVIDIA-L40S', the only GPU type currently on this cluster.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(hf_repo_id)
    except Exception as e:
        return f"Could not fetch model info for '{hf_repo_id}' from Hugging Face: {e}"

    if not info.safetensors or not info.safetensors.parameters:
        return (
            f"'{hf_repo_id}' has no safetensors metadata to size from — it "
            f"may not be in safetensors format, or may be gated/private."
        )

    param_map = info.safetensors.parameters
    if dtype == "auto":
        dtype_used, total_params = max(param_map.items(), key=lambda kv: kv[1])
    else:
        dtype_used = dtype.upper()
        total_params = param_map.get(dtype_used) or sum(param_map.values())

    bytes_per_param = BYTES_PER_PARAM.get(dtype_used.upper())
    if bytes_per_param is None:
        return (
            f"Unrecognized dtype '{dtype_used}' for '{hf_repo_id}' — known "
            f"dtypes are {sorted(BYTES_PER_PARAM)}. Pass an explicit `dtype`."
        )

    estimated_vram_gb = total_params * bytes_per_param * FOOTPRINT_OVERHEAD_FACTOR / 1e9

    gpu_vram_gb = GPU_VRAM_GB.get(gpu_product)
    if gpu_vram_gb is None:
        return (
            f"~{total_params / 1e9:.1f}B params ({dtype_used}), estimated "
            f"{estimated_vram_gb:.1f}GB VRAM needed. VRAM for '{gpu_product}' "
            f"isn't in the known GPU table — add it to GPU_VRAM_GB to get a "
            f"tensor_parallel_size recommendation, or pass a known gpu_product."
        )

    usable_vram_gb = gpu_vram_gb * GPU_UTILIZATION_HEADROOM
    recommended_tp = max(1, math.ceil(estimated_vram_gb / usable_vram_gb))

    tier_note = ""
    if len(GPU_VRAM_GB) <= 1:
        tier_note = (
            f" Note: this cluster only has one known GPU type "
            f"({gpu_product}), so there's no cost/latency tier tradeoff to "
            f"weigh yet — tensor_parallel_size is the main lever."
        )

    return (
        f"'{hf_repo_id}': ~{total_params / 1e9:.1f}B params ({dtype_used}), "
        f"estimated {estimated_vram_gb:.1f}GB VRAM (includes a ~20% overhead "
        f"margin for activations/KV-cache — a rough estimate, not exact). "
        f"Recommended tensor_parallel_size={recommended_tp} on {gpu_product} "
        f"({gpu_vram_gb}GB VRAM each).{tier_note}"
    )


_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_VALID_OUTPUT_KINDS = ("chat", "image", "video")


@tool
def generate_model_manifests(
    model_name: str,
    hf_repo_id: str,
    tensor_parallel_size: int,
    output_kind: str,
    gpu_product: str = "NVIDIA-L40S",
    context_len: int = 8192,
    quantization: str = "none",
    pvc_size_gb: int = 100,
) -> str:
    """Generate the Kustomize/KServe manifests to add a new vLLM model to
    the platform's model catalog, following this repo's existing patterns.
    Plain text models (output_kind='chat') get the stock vLLM image +
    hermes tool-call-parser, matching the hand-built qwen25-gpu model —
    see platform/base/models/qwen25-gpu. Multimodal models (output_kind
    'image'/'video') get vLLM-Omni instead, matching platform/base/models/
    qwen3-omni — do NOT use vLLM-Omni for a plain chat model, it doesn't
    need that orchestration and hasn't been validated for it.

    Call list_cluster_gpus and estimate_model_footprint first to choose
    gpu_product and tensor_parallel_size — don't guess these.

    IMPORTANT: this tool only generates text. It does not deploy anything —
    it cannot write files, run kubectl/oc, or touch git. This platform uses
    GitOps (ArgoCD auto-syncs 'main' with self-heal + prune), so a model
    only becomes real once a human saves these files under
    platform/base/models/<model_name>/, adds that path to the relevant
    overlay's kustomization.yaml (e.g. platform/overlays/dev-gpu and
    platform/overlays/demo), and commits/merges it. Return the generated
    YAML to the user verbatim (in fenced code blocks per file) rather than
    summarizing or paraphrasing it — it needs to be copy-pasteable.

    Args:
        model_name: Catalog name for the model, lowercase alphanumeric and
            hyphens only (e.g. 'my-new-model'). Used as the K8s resource
            name prefix throughout.
        hf_repo_id: Hugging Face repo id to download and serve, e.g.
            'Qwen/Qwen3-8B'.
        tensor_parallel_size: Number of GPUs to shard the model across —
            get this from estimate_model_footprint, don't guess.
        output_kind: One of 'chat', 'image', or 'video' — must match what
            the model actually serves; this is what call_model uses to pick
            the right API shape, and determines stock-vLLM vs vLLM-Omni here.
        gpu_product: GPU type to target via nodeSelector — see
            list_cluster_gpus for what's available on this cluster.
        context_len: Max model/sequence length in tokens.
        quantization: vLLM quantization scheme (e.g. 'fp8'), or 'none' to
            omit and let vLLM use the model's native dtype.
        pvc_size_gb: Size of the PersistentVolumeClaim for the model cache,
            in GB. Should comfortably exceed the model's on-disk weight
            size.
    """
    if not _NAME_RE.match(model_name):
        return (
            f"'{model_name}' isn't a valid model_name — use lowercase "
            f"alphanumeric characters and hyphens only (e.g. 'my-new-model')."
        )
    if output_kind not in _VALID_OUTPUT_KINDS:
        return f"output_kind must be one of {_VALID_OUTPUT_KINDS}, got '{output_kind}'."
    if tensor_parallel_size < 1:
        return "tensor_parallel_size must be at least 1."

    use_omni = output_kind != "chat"
    devices = ",".join(str(i) for i in range(tensor_parallel_size))
    quant_line = f"\n    quantization: {quantization}" if quantization != "none" else ""

    files = {}

    files["pvc.yaml"] = f"""\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {model_name}-model-cache
  namespace: physical-ai-models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {pvc_size_gb}Gi
  storageClassName: gp3-csi
"""

    files["model-download-job.yaml"] = f"""\
apiVersion: batch/v1
kind: Job
metadata:
  name: download-{model_name}-model
  namespace: physical-ai-models
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: downloader
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          set -e
          pip install -q huggingface_hub
          python3 << 'PYEOF'
          from huggingface_hub import snapshot_download
          import os
          token = os.getenv('HF_TOKEN')
          snapshot_download(
              repo_id='{hf_repo_id}',
              local_dir='/mnt/models',
              token=token,
          )
          PYEOF
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-token
              key: HF_TOKEN
        volumeMounts:
        - name: model-storage
          mountPath: /mnt/models
        resources:
          requests:
            cpu: "2"
            memory: 8Gi
          limits:
            cpu: "4"
            memory: 16Gi
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: {model_name}-model-cache
"""

    # Conservative fixed CPU/memory defaults — not derived from a formula,
    # tune based on observed usage. GPU count is the one value that maps
    # directly to tensor_parallel_size.
    if use_omni:
        files["deploy-config.yaml"] = f"""\
stages:
  - stage_id: 0
    max_num_batched_tokens: {context_len}
    max_model_len: {context_len}
    max_num_seqs: 1
    gpu_memory_utilization: 0.95
    trust_remote_code: true
    enable_prefix_caching: false
    enforce_eager: true
    tensor_parallel_size: {tensor_parallel_size}
    devices: "{devices}"{quant_line}
    default_sampling_params:
      temperature: 0.0
      max_tokens: 2048
"""

        files["servingruntime.yaml"] = f"""\
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-omni-{model_name}-runtime
  namespace: physical-ai-models
  labels:
    opendatahub.io/dashboard: "true"
spec:
  annotations:
    opendatahub.io/kserve-runtime: vllm
    prometheus.kserve.io/path: /metrics
    prometheus.kserve.io/port: "8080"
  containers:
  - name: kserve-container
    image: quay.io/vllm/automation-vllm-omni:cuda-27760112356
    ports:
    - containerPort: 8080
      protocol: TCP
    command:
    - /bin/bash
    - -c
    args:
    - |
      unset HF_HUB_OFFLINE

      vllm serve /mnt/models \\
        --omni \\
        --port=8080 \\
        --served-model-name={{{{.Name}}}} \\
        --trust-remote-code \\
        --deploy-config /etc/vllm-omni/deploy-config.yaml \\
        --stage-init-timeout 900
    env:
    - name: HF_HOME
      value: /tmp/hf_home
    - name: TORCHINDUCTOR_CACHE_DIR
      value: /tmp/torch_cache
    volumeMounts:
    - name: deploy-config
      mountPath: /etc/vllm-omni/
      readOnly: true
    resources:
      requests:
        cpu: "4"
        memory: 64Gi
        nvidia.com/gpu: "{tensor_parallel_size}"
      limits:
        cpu: "8"
        memory: 128Gi
        nvidia.com/gpu: "{tensor_parallel_size}"
  volumes:
  - name: deploy-config
    configMap:
      name: {model_name}-deploy-config
  supportedModelFormats:
  - name: pytorch
    version: "2"
    autoSelect: true
  - name: vllm
    autoSelect: true
"""
        model_format = "pytorch"
        runtime_name = f"vllm-omni-{model_name}-runtime"
        req_mem, lim_mem = "64Gi", "128Gi"
    else:
        tp_line = (
            f" \\\n        --tensor-parallel-size={tensor_parallel_size}"
            if tensor_parallel_size > 1
            else ""
        )
        quant_arg = f" \\\n        --quantization={quantization}" if quantization != "none" else ""
        files["servingruntime.yaml"] = f"""\
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-{model_name}-runtime
  namespace: physical-ai-models
  labels:
    opendatahub.io/dashboard: "true"
spec:
  annotations:
    opendatahub.io/kserve-runtime: vllm
    prometheus.kserve.io/path: /metrics
    prometheus.kserve.io/port: "8080"
  containers:
  - name: kserve-container
    image: docker.io/vllm/vllm-openai:v0.24.0
    ports:
    - containerPort: 8080
      protocol: TCP
    command:
    - /bin/bash
    - -c
    args:
    - |
      vllm serve /mnt/models \\
        --port=8080 \\
        --served-model-name={{{{.Name}}}} \\
        --trust-remote-code \\
        --enable-auto-tool-choice \\
        --tool-call-parser hermes \\
        --max-model-len {context_len}{tp_line}{quant_arg}
    env:
    - name: HF_HOME
      value: /tmp/hf_home
    resources:
      requests:
        cpu: "4"
        memory: 32Gi
        nvidia.com/gpu: "{tensor_parallel_size}"
      limits:
        cpu: "8"
        memory: 48Gi
        nvidia.com/gpu: "{tensor_parallel_size}"
  supportedModelFormats:
  - name: vllm
    autoSelect: true
"""
        model_format = "vllm"
        runtime_name = f"vllm-{model_name}-runtime"
        req_mem, lim_mem = "32Gi", "48Gi"

    files["inferenceservice.yaml"] = f"""\
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {model_name}
  namespace: physical-ai-models
  labels:
    opendatahub.io/dashboard: "true"
    opendatahub.io/genai-asset: "true"
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/autoscalerClass: external
    sidecar.istio.io/inject: "false"
    physical-ai.io/output-kind: {output_kind}
spec:
  predictor:
    minReplicas: 0
    nodeSelector:
      nvidia.com/gpu.product: {gpu_product}
    volumes:
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 16Gi
    model:
      modelFormat:
        name: {model_format}
      runtime: {runtime_name}
      storageUri: pvc://{model_name}-model-cache
      volumeMounts:
      - name: dshm
        mountPath: /dev/shm
      resources:
        requests:
          cpu: "4"
          memory: {req_mem}
          nvidia.com/gpu: "{tensor_parallel_size}"
        limits:
          cpu: "8"
          memory: {lim_mem}
          nvidia.com/gpu: "{tensor_parallel_size}"
"""

    files["external-model.yaml"] = f"""\
apiVersion: maas.opendatahub.io/v1alpha1
kind: ExternalModel
metadata:
  name: {model_name}
  namespace: physical-ai-models
spec:
  provider: openai
  endpoint: maas-proxy.physical-ai-models.svc.cluster.local
  targetModel: {model_name}
  credentialRef:
    name: {model_name}-credentials
---
apiVersion: v1
kind: Secret
metadata:
  name: {model_name}-credentials
  namespace: physical-ai-models
  labels:
    inference.networking.k8s.io/bbr-managed: "true"
type: Opaque
stringData:
  api-key: not-used
"""

    files["model-ref.yaml"] = f"""\
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSModelRef
metadata:
  name: {model_name}
  namespace: physical-ai-models
spec:
  modelRef:
    kind: ExternalModel
    name: {model_name}
"""

    files["subscription.yaml"] = f"""\
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSSubscription
metadata:
  name: {model_name}-subscription
  namespace: models-as-a-service
spec:
  owner:
    groups:
      - name: system:authenticated
  modelRefs:
    - name: {model_name}
      namespace: physical-ai-models
      tokenRateLimits:
        - limit: 100000
          window: 1h
"""

    files["auth-policy.yaml"] = f"""\
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSAuthPolicy
metadata:
  name: {model_name}-auth
  namespace: models-as-a-service
spec:
  modelRefs:
    - name: {model_name}
      namespace: physical-ai-models
  subjects:
    groups:
      - name: system:authenticated
"""

    files["httpscaledobject.yaml"] = f"""\
apiVersion: http.keda.sh/v1alpha1
kind: HTTPScaledObject
metadata:
  name: {model_name}-http-scaler
  namespace: physical-ai-models
spec:
  hosts:
  - {model_name}-predictor.physical-ai-models.svc.cluster.local
  targetPendingRequests: 1
  scaleTargetRef:
    name: {model_name}-predictor
    kind: Deployment
    apiVersion: apps/v1
    service: {model_name}-predictor
    port: 80
  replicas:
    min: 0
    max: 3
  scaledownPeriod: 3600
"""

    if use_omni:
        files["kustomization.yaml"] = f"""\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - pvc.yaml
  - model-download-job.yaml
  - servingruntime.yaml
  - inferenceservice.yaml
  - external-model.yaml
  - model-ref.yaml
  - subscription.yaml
  - auth-policy.yaml
  - httpscaledobject.yaml

configMapGenerator:
- name: {model_name}-deploy-config
  namespace: physical-ai-models
  files:
  - deploy-config.yaml
  options:
    disableNameSuffixHash: true
"""
    else:
        files["kustomization.yaml"] = f"""\
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - pvc.yaml
  - model-download-job.yaml
  - servingruntime.yaml
  - inferenceservice.yaml
  - external-model.yaml
  - model-ref.yaml
  - subscription.yaml
  - auth-policy.yaml
  - httpscaledobject.yaml
"""

    blocks = "\n".join(
        f"### platform/base/models/{model_name}/{path}\n```yaml\n{content}```"
        for path, content in files.items()
    )

    instructions = f"""\
This is a draft only — nothing has been deployed. To actually add this
model to the catalog:

1. Save each file above under `platform/base/models/{model_name}/`.
2. Add `../../base/models/{model_name}/` to the `resources:` list in
   whichever overlay(s) should serve it — typically
   `platform/overlays/dev-gpu/kustomization.yaml` and/or
   `platform/overlays/demo/kustomization.yaml`.
3. Run `make validate` to check it builds cleanly.
4. Commit on a feature branch and open a PR — main is GitOps-synced by
   ArgoCD with self-heal and prune, so nothing applied outside of git will
   stick.

Not included here (optional, copy from an existing model like qwen3-omni
if wanted): metrics-service.yaml and prometheus-rule.yaml for
Prometheus/Grafana observability — not required for the model to function.
"""

    return f"{blocks}\n\n{instructions}"
