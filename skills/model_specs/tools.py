from importlib import resources

from langchain_core.tools import tool


@tool
def get_model_reference(model_name: str) -> str:
    """Fetch a catalog model's static spec reference -- dataset/training
    compatibility, architecture notes -- bundled with this skill. This is
    not live cluster state (see the models skill for a deployed instance's
    status). model_name is the catalog directory name (e.g. 'pi05',
    'dreamzero'), not a Hugging Face repo id.

    Args:
        model_name: Catalog model directory name, e.g. 'pi05' or 'dreamzero'.
    """
    references_dir = resources.files(__package__) / "references"
    ref_file = references_dir / f"{model_name}.md"
    if not ref_file.is_file():
        available = sorted(
            p.name.removesuffix(".md") for p in references_dir.iterdir() if p.name.endswith(".md")
        )
        return (
            f"No model-specs reference for '{model_name}'. Available: {available}. "
            f"A missing model either has no fine-tuning recipe on this platform or "
            f"hasn't been documented yet -- don't guess its specifics from general "
            f"knowledge of the base model."
        )
    return f"model-specs/references/{model_name}.md:\n{ref_file.read_text(encoding='utf-8')}"
