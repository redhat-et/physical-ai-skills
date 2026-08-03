# Physical AI Platform Agent Skills

Agentic workflow skills for the platform agent in
[physical-ai-platform-demo](https://github.com/redhat-et/physical-ai-platform-demo).
Fetched into the agent's container image at build time.

## Structure

```
skills/
  <skill-name>/
    SKILL.md          # workflow instructions
    scripts/          # standalone CLI scripts, run via a shell/exec tool
    tools.py          # LangChain tools, for the rare skill that still needs
                       # one (e.g. structured image/video output no plain
                       # CLI script can produce)
```

Each script under `scripts/` is self-contained (no shared helper module
between scripts) and does its whole job end-to-end, including submitting to
the cluster where relevant — see a skill's `SKILL.md` for the exact
invocation. Not every skill has scripts or a `tools.py` — some are
instructions-only.

## Using these skills

This repo follows the [Agent Skills](https://agentskills.io) open standard
(`skills/<name>/SKILL.md`), the same layout used by
[`nvidia/skills`](https://github.com/NVIDIA/skills). That means these skills
are installable with the standard [`npx skills`](https://github.com/vercel-labs/skills)
CLI, in any project, for any supported agent (Claude Code, Cursor, Codex,
and others) — no cloning this repo required:

```bash
# install everything
npx skills add redhat-et/physical-ai-skills -a claude-code -y

# or just the skills you want
npx skills add redhat-et/physical-ai-skills --skill datasets --skill fine-tuning -a claude-code -y

# install globally, available in every project
npx skills add redhat-et/physical-ai-skills -a claude-code -g -y
```

If you're actively developing skills in a local checkout of this repo,
point `npx skills add` at the local path instead of the GitHub repo — it
symlinks by default, so edits show up immediately with no reinstall step:

```bash
npx skills add /path/to/physical-ai-platform-agent-skills -a claude-code -y
```
