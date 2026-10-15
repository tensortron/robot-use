# agent_ws: the agent's working directory (L2)

Start a coding agent here (Claude Code: `cd agent_ws && claude`). It connects to the MCP server through
`.mcp.json` (the server must be running: `python -m stack.robotd` and `python -m stack.mcp_server` from the
repository root), reads `AGENTS.md` (which imports the robot description and the library API), has no
shell, and cannot edit the stack or the library. It acts only by running programs, which a human
approves on robotd's operator page.

- `AGENTS.md` - the agent's brief
- `tasks/` - task briefs (start with `tasks/pick_place.md`)
- `.agents/skills/` - the agent's own skills: drafts from discovery, validated after clean live runs and the user's
  approval (lifecycle in `AGENTS.md`, "Skills") (`.claude` is a symlink to
  `.agents`, for Claude Code; `.agents/settings.json` is Claude Code's settings for this workspace)
- Other agents: `.codex/config.toml` (Codex's MCP server; omp reads it too); `.pi/extensions/pi-permission-system/config.json`
  (omp's permission policy, needs the `@gotgenes/pi-permission-system` extension installed in omp) and
  `.omp/config.yml` (omp's own tool denials)
- Episodes are recorded by the stack outside this folder, under `$ROBOT_USE_HOME/episodes/`
