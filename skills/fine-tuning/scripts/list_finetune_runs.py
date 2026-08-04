#!/usr/bin/env python3
"""List fine-tuning experiments started on this cluster. See ../SKILL.md."""
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

from lib.finetune_pipeline import get_pipeline_run_status  # noqa: E402

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
FINETUNE_RUN_ID_ANNOTATION = "physical-ai.io/kfp-run-id"


def _get_core_api():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def list_finetune_runs() -> str:
    """List fine-tuning experiments started on this cluster. Call this when
    asked about fine-tuning runs in general (e.g. "what's running", "any
    fine-tunes in progress") or when the exact exp_name isn't known --
    get_finetune_run_status requires the exact exp_name and has no other
    way to look one up, so without this, a forgotten exp_name is
    unrecoverable.

    Shows each experiment's exp_name and pipeline run state (best-effort --
    "no pipeline run recorded" if submit_finetune_run failed partway
    through, "status unavailable" if the pipeline run can't be reached).
    """
    core_api = _get_core_api()
    pvcs = core_api.list_namespaced_persistent_volume_claim(
        namespace=settings.datasets_namespace,
        label_selector=FINETUNE_EXP_LABEL,
    )

    if not pvcs.items:
        return "No fine-tuning runs found."

    lines = []
    for pvc in pvcs.items:
        exp_name = (pvc.metadata.labels or {}).get(FINETUNE_EXP_LABEL, "unknown")
        run_id = (pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
        if not run_id:
            state = "no pipeline run recorded"
        else:
            try:
                state = get_pipeline_run_status(run_id).splitlines()[0]
            except Exception:
                state = "status unavailable"
        lines.append(f"- {exp_name}: {state}")

    return "Fine-tuning runs:\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    try:
        print(list_finetune_runs())
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
