# Physical AI Platform Agent Skills

Agentic workflow skills for the platform agent in
[physical-ai-platform-demo](https://github.com/redhat-et/physical-ai-platform-demo).
Fetched into the agent's container image at build time.

## Structure

```
skills/
  <skill-name>/
    SKILL.md    # workflow instructions the agent loads via get_skill()
    tools.py    # the skill's LangChain tools, auto-discovered by the agent
```

Not every skill has a `tools.py` — some are instructions-only.
