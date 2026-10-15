# Starter skills

Skills shipped with robot-use for the operating agent's first sessions. `tools/setup_agent_ws.py` copies each folder
here into `agent_ws/.agents/skills/` when the agent has no skill of that name (it never replaces the agent's own),
as a **draft** with one history line saying where it came from. From then on the skill is the agent's: it uses,
revises and validates it on its robot under the skill lifecycle in `agent_ws/AGENTS.md`.
`setup_agent_ws.py --no-starter-skills` leaves them out.

| Skill | What it does |
|---|---|
| `pick-and-place` | One arm picks a box-like object (jaws square to two flat faces, any yaw) and puts it in a cup or on a surface. Validated on a reference rig for an upright ~3.5 cm block into a cup; runs in the simulated rig's start layout. |

A change here is code that agents run: it gets the independent review, and a run in the sim's start layout
(`tools/run_sim.py`), before it is committed.
