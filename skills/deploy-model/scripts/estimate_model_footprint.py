#!/usr/bin/env python3
"""Estimate the GPU memory footprint of a Hugging Face model and how many of
a given GPU type it would need. See ../SKILL.md."""
import argparse
import math

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
            f"dtypes are {sorted(BYTES_PER_PARAM)}. Pass an explicit --dtype."
        )

    estimated_vram_gb = total_params * bytes_per_param * FOOTPRINT_OVERHEAD_FACTOR / 1e9

    gpu_vram_gb = GPU_VRAM_GB.get(gpu_product)
    if gpu_vram_gb is None:
        return (
            f"~{total_params / 1e9:.1f}B params ({dtype_used}), estimated "
            f"{estimated_vram_gb:.1f}GB VRAM needed. VRAM for '{gpu_product}' "
            f"isn't in the known GPU table — add it to GPU_VRAM_GB to get a "
            f"tensor_parallel_size recommendation, or pass a known --gpu-product."
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", required=True, help="Hugging Face repo id, e.g. 'Qwen/Qwen3-8B'.")
    parser.add_argument("--dtype", default="auto", help="Weight dtype to size for, e.g. 'BF16', 'FP8', 'INT4'.")
    parser.add_argument("--gpu-product", default="NVIDIA-L40S", help="See list_cluster_gpus.py for what's available.")
    args = parser.parse_args()

    try:
        print(estimate_model_footprint(args.hf_repo_id, args.dtype, args.gpu_product))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
