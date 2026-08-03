---
name: model-specs
description: Use when asked about a catalog model's own static characteristics — architecture, dataset/training compatibility, serving runtime requirements — e.g. "what was DreamZero trained on" or "does pi0.5 need a fixed camera count." Not for live status, scaling, or calling a deployed model (see the models skill for that).
---
MODEL SPECS — this skill is knowledge-only: no cluster calls, no
scaling/inference actions. It's the reference for what a catalog model
*is*, as opposed to the models skill (what a deployed instance is
currently doing).

## Scripts

Run via the shell tool: `python3 "$SKILLS_ROOT/model-specs/scripts/get_model_reference.py" --model-name <name>`.

Each catalog model with a documented dataset/training profile has a file
under `references/<model_name>.md` (e.g. `references/pi05.md`,
`references/dreamzero.md`), keyed by the same catalog directory name used
elsewhere (platform/base/models/<model_name>/). A model with no file here
either has no fine-tuning recipe on this platform or hasn't been
documented yet — don't guess its specifics from general knowledge of the
base model.

pi0.5's reference is a Dimension/Priority checklist table matching the
datasets skill's own DATASET COMPATIBILITY CHECKLIST row for row — read
them side by side by row number when validating a candidate dataset.
DreamZero's is plain training-data provenance prose instead of a table,
since it's inference-only on this platform (no recipe to check a dataset
against).
