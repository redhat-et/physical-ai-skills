#!/usr/bin/env python3
"""Fetch a catalog model's static spec reference. See ../SKILL.md."""
import argparse
from pathlib import Path

# References live as sibling files under ../references/ -- resolved from this
# script's own location (not importlib.resources' package-relative lookup),
# since this runs standalone, not as an importable package submodule.
REFERENCES_DIR = Path(__file__).resolve().parent.parent / "references"


def get_model_reference(model_name: str) -> str:
    """Fetch a catalog model's static spec reference -- dataset/training
    compatibility, architecture notes -- bundled with this skill. This is
    not live cluster state (see the models skill for a deployed instance's
    status). model_name is the catalog directory name (e.g. 'pi05',
    'dreamzero'), not a Hugging Face repo id.
    """
    ref_file = REFERENCES_DIR / f"{model_name}.md"
    if not ref_file.is_file():
        available = sorted(p.stem for p in REFERENCES_DIR.glob("*.md"))
        return (
            f"No model-specs reference for '{model_name}'. Available: {available}. "
            f"A missing model either has no fine-tuning recipe on this platform or "
            f"hasn't been documented yet -- don't guess its specifics from general "
            f"knowledge of the base model."
        )
    return f"model-specs/references/{model_name}.md:\n{ref_file.read_text(encoding='utf-8')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Catalog model directory name, e.g. 'pi05' or 'dreamzero'.")
    args = parser.parse_args()

    try:
        print(get_model_reference(args.model_name))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
