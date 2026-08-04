#!/usr/bin/env python3
"""Check the status of a fine-tuning run started by submit_finetune_run. See
../SKILL.md."""
import argparse
import sys
from pathlib import Path

from kubernetes import client

from platform_agent.config import settings

# Resolved from this script's own location, not a dotted platform_agent.skills
# path -- see submit_finetune_run.py for why.
def _resolve_lib_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_resolve_lib_path()

from lib.finetune_pipeline import get_finetune_eval_metrics, get_pipeline_run_status  # noqa: E402

FINETUNE_RUN_ID_ANNOTATION = "physical-ai.io/kfp-run-id"


def _get_core_api():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def _checkpoint_pvc_name(exp_name: str) -> str:
    return f"finetune-{exp_name}-checkpoint-pvc"


def get_finetune_run_status(exp_name: str) -> str:
    """Check the status of a fine-tuning run started by submit_finetune_run.

    Reports the pipeline run's overall state and per-stage state. The
    pipeline advances through its own stages on its own -- this is a
    read-only status check, not something that needs to be called
    repeatedly to make progress happen. Also includes the resolved recipe
    (dataset, step count, batch size, episode split, ...) logged to MLflow
    at submission time, plus mean and per-episode action-MSE eval results
    once the evaluate stage has logged them there too -- eval results are
    omitted if that stage hasn't run yet or hasn't finished; the recipe
    itself is omitted only if the submission-time MLflow logging failed.
    """
    core_api = _get_core_api()

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=settings.datasets_namespace
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No fine-tuning run found for exp_name '{exp_name}' — has submit_finetune_run been called?"
        return f"Could not read checkpoint PVC '{checkpoint_pvc_name}': {e.reason}"

    run_id = (pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
    if not run_id:
        return (
            f"Checkpoint PVC for '{exp_name}' exists but has no pipeline run recorded — "
            f"submit_finetune_run may have failed partway through."
        )

    status = get_pipeline_run_status(run_id)
    result = f"{status}\nCheckpoint PVC: '{checkpoint_pvc_name}'."

    eval_results = get_finetune_eval_metrics(exp_name)
    if eval_results:
        recipe_params = eval_results.get("params")
        if recipe_params:
            result += "\nRecipe: " + ", ".join(f"{k}={v}" for k, v in recipe_params.items())

        metrics = eval_results["metrics"]
        per_episode = sorted(
            ((k[len("action_mse_ep") :], v) for k, v in metrics.items() if k.startswith("action_mse_ep")),
            key=lambda kv: int(kv[0]),
        )
        result += f"\nEval results (MLflow run '{exp_name}', {eval_results['status']}):"
        if "mean_action_mse" in metrics:
            result += f"\n  Mean action MSE: {float(metrics['mean_action_mse']):.4f}"
        if per_episode:
            per_episode_str = ", ".join(f"{ep}={float(mse):.4f}" for ep, mse in per_episode)
            result += f"\n  Per-episode MSE: {per_episode_str}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, help="The exp_name passed to submit_finetune_run.")
    args = parser.parse_args()

    try:
        print(get_finetune_run_status(args.exp_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
