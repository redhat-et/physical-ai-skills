# Adding a skill or script

This repo follows the [Agent Skills](https://agentskills.io) open standard.
Every skill is a plain directory under `skills/` -- no framework-specific
tool-calling code, so it stays installable via `npx skills add` in any
agent, not just this platform's own agent.

## 1. Directory structure

```
skills/<skill-name>/
  SKILL.md            # required -- workflow instructions
  scripts/            # optional -- standalone CLI scripts
    lib/               # optional -- shared helper modules, NOT directly
                       # invokable (imported by other scripts, no header)
  references/         # optional -- static reference docs an agent should
                       # read but that don't belong in SKILL.md itself
```

Not every skill needs `scripts/` -- `new-model-runtime` and `model-specs`
are (mostly) instructions-only. Copy
[`templates/SKILL.md.template`](templates/SKILL.md.template) to
`skills/<skill-name>/SKILL.md` to start.

## 2. Writing `SKILL.md`

Required frontmatter is `name` and `description` (the
[agentskills.io spec](https://agentskills.io/specification) also defines
optional `license`, `compatibility`, `metadata`, and `allowed-tools`
fields, but no skill in this repo needs them yet):

- `name` must exactly match the skill's directory name, 1-64 characters,
  lowercase letters/numbers/hyphens only, no leading/trailing or
  consecutive hyphens (`skill-check` and CI reject anything else).
- `description` is the *only* thing an agent sees to decide whether this
  skill applies to a given request (max 1024 characters), so:
  - Lead with concrete trigger phrases/tasks, not an abstract summary.
  - If another skill covers a similar-sounding request, say so explicitly
    (e.g. models/SKILL.md's description distinguishes itself from
    deploy-model and deploy-checkpoint). This is how an agent picks the
    right skill instead of guessing.

In the body, state invariants the agent must not violate -- a namespace it
must use, a value it must echo back verbatim instead of inventing, a
resource kind it must not confuse with another. See `models/SKILL.md`'s
"Never conclude the model's current scale from a previous turn's message"
for the pattern.

If the skill has scripts, document the exact invocation (`python3
"$SKILLS_ROOT/<skill>/scripts/<script>.py" --example-arg value`, replacing
`--example-arg` with the script's actual declared option name(s)) and, if a
task needs more than one script call or a before/after check, spell out the
order -- don't assume the agent will infer sequencing on its own.

## 3. Writing a script

Copy [`templates/script.py.template`](templates/script.py.template) to
`skills/<skill-name>/scripts/<script>.py`. Rules, enforced by CI (see
below):

- **Self-contained.** Each script does its whole job end-to-end (including
  submitting to the cluster, where relevant). No shared helper module
  between scripts at the top level -- if two scripts in the same skill
  genuinely need to share code, put it under `scripts/lib/` and import
  from there; `lib/` is excluded from the header check since it's not
  directly invokable via `run_script`.
- **Config via `os.environ`, not an import.** Scripts run standalone in a
  subprocess, not inside any particular Python package's process --
  `os.environ.get("MODELS_NAMESPACE", "physical-ai-models")`, not
  `from platform_agent.config import settings`.
- **Declare a `# ---`-delimited YAML header** at the top of the file (see
  the template) with:
  - `description` (required) -- what the script does and when to use it.
  - `parameters` (optional list) -- each needs `name` and `type`
    (`string`, `integer`, `number`, `boolean`, or `array`); `required` and
    `default`/`description` are optional per-parameter.
  - This header is how the platform's MCP server (`get_script`) tells an
    agent the script's contract without exposing its source, and how it
    validates arguments before running it via `run_script`.
- **Plain stdout, no structured output channel.** Whatever `print()`s in
  `main()` is the entire result an agent sees -- make it self-explanatory
  on its own (see `scale_model.py`'s return strings for the pattern).
  There is currently no supported way for a script to return an image/video
  artifact; that's why `call_model` was dropped rather than migrated (see
  `models/SKILL.md`'s NOTE) -- don't try to route media output through a
  script until that's redesigned.

## 4. Validate locally before opening a PR

```bash
npx skill-check . --no-security-scan      # agentskills.io compliance
python scripts/validate_script_headers.py # script header contract
```

Both run in CI on every PR (`.github/workflows/skill-check.yml`). A script
missing a valid header, or a `SKILL.md` that fails `skill-check`, blocks
the merge.
