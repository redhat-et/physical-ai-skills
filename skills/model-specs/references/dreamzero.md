# DreamZero — Dataset Compatibility

DreamZero is inference-only on this platform — no fine-tuning recipe exists
here, so don't offer to fine-tune it. Because there's no recipe to check a
candidate dataset against, this section isn't formatted as the
Dimension/Priority checklist table used in pi05's reference (see
`references/pi05.md`) — it's purely training-data provenance, for
answering "what was this trained on" questions.

Trained on `GEAR-Dreams/DreamZero-DROID-Data`: 57,774 episodes (~131-145GB,
14.7M frames), a filtered DROID derivative with idle frames, non-annotated
episodes, and unsuccessful episodes removed. This is not the "~75k" figure
sometimes quoted for DROID — that figure is DROID's overall annotation
coverage, not this repo's episode count.

Uses relative joint positions as its action space, unlike the cartesian
encoding most raw LeRobot DROID ports expose.

`DreamZero-AgiBot` is a separate checkpoint trained on different data
(AgiBot G1 teleop) — don't conflate it with DreamZero-DROID when discussing
training data.
