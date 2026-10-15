# robot-use: controlling robots with AI agents

robot-use is a stack that lets an AI agent drive a robot by writing Python programs. A safety daemon (robotd)
plans and checks every motion before any servo moves. The first robot it fully supports is a bimanual SO-101
(two follower arms, an ego camera between them, a wrist camera on each); [docs/EMBODIMENTS.md](docs/EMBODIMENTS.md)
says how it would extend to other bodies.

This file is for agents working **on** the repo. The agent that **operates** the robot has its own brief in
`agent_ws/AGENTS.md`, and it must not edit the stack.

**If `docs/RIG.md` exists, read it before you act.** It describes this checkout's own robot, machine and records,
and its rules apply in addition to these.

## Layout

| Path | What it is |
|---|---|
| `robot.yaml` | **The single measured description** of the robot: arm mounts, joint signs and offsets, board, obstacles, cameras, gripper, TCP. Each section lists its `sources`. |
| `calib/` | The ego camera's calibration files, named in `robot.yaml` |
| `stack/rig/` | Loads the description (`rig.py`), FK and IK (`kinematics.py`), the camera model and overlays (`camera_model.py`), and `embodiment.md` (the robot described in words, for agents) |
| `stack/robotd/` | L0, the safety daemon. It owns the serial ports, plans and checks every motion (`safety.py`, `planner.py`), and handles the lease, approvals and contact monitoring. It also serves the operator page (`http.py`, `operator.html`). |
| `stack/robotd.yaml` | robotd's non-geometry settings. It rejects geometry keys, because geometry lives in robot.yaml. |
| `stack/runner/` | Runs agent programs: a static check, an audit-hook sandbox, a kernel box, a child process over JSON pipes, budgets, checkpoints, traces and episodes |
| `stack/twin_mujoco/` | The headless MuJoCo twin: the simulated rig (`robotd --sim`) and program previews. Its scene is built from robot.yaml. |
| `stack/mcp_server/` | The MCP surface the agent sees (`--surface agent`, the default) or a debug surface (`--surface debug`) |
| `stack/tests/` | The test suite (about 330 tests, ~8 min) |
| `lib/robot_use/` | L1, the library agent programs import. It provides primitives only. Its API is in `lib/robot_use/README.md`. |
| `agent_ws/` | L2, the operating agent's working directory: its brief, `.mcp.json`, settings (Bash and stack edits denied; `settings.json` and the pi permissions are rendered from their `*.template.json` by `tools/setup_agent_ws.py`: edit the template, have it reviewed, re-pin it, re-run the tool), skills and task briefs |
| `tools/` | Launchers (`run_sim.py`, `run_rig.py`), `doctor.py`, `setup_agent_ws.py`, the calibration page (`calibrate.py`) and its pieces, device identification, bench checks, `torque_off.py`, `fetch_so101_assets.py` |
| `udev/` | Rules for durable `/dev/so101_*` and `/dev/cam_*` device names |
| `docs/` | Design and guides (see below) |

## Environment

- Install with `uv sync --extra dev` (add `--extra hardware` for the real arms: lerobot, pinned), then use the
  project's Python, `.venv/bin/python` (or `uv run python`). A new dependency goes into `pyproject.toml`, then
  `uv lock`. If this machine's environment was built by hand, `docs/RIG.md` says so and how to install instead.
- What the stack records lives outside the repo, beneath `$ROBOT_USE_HOME` (default `~/.tensortron/robot-use`):
  `agent_logs/` (robotd), `episodes/` (programs, traces, keyframes), `sim/` (the simulated rig's), `calibration/`.
- The SO-101 MuJoCo model is fetched with `tools/fetch_so101_assets.py`. The sim's cameras need OpenGL:
  `tools/doctor.py` probes which `MUJOCO_GL` setting works here.

## Commands

```bash
.venv/bin/python -m pytest stack/tests -q                 # the whole suite; keep it green
.venv/bin/python tools/doctor.py [--hardware]             # what this machine is missing, and the fix
.venv/bin/python tools/run_sim.py [--port-base 8790]      # robotd --sim + the MCP server, Ctrl-C stops both
.venv/bin/python tools/setup_agent_ws.py [--check]        # the agent's permissions, rendered for this machine
.venv/bin/python tools/platform_check.py                  # agent settings, doctor, escape probe, suite: one report
.venv/bin/python tools/run_rig.py [--enable-motion] [--approval none]
                                                          # the real arms: robotd + the MCP server (Hardware safety)
.venv/bin/python tools/calibrate.py [--trial] [--camera-only]
                                                          # the guided calibration page (docs/CALIBRATION.md)
.venv/bin/python -m stack.robotd --mock                   # robotd alone, with no hardware
.venv/bin/python -m stack.mcp_server [--surface debug]    # MCP over HTTP at 127.0.0.1:8771/mcp
```

## Hardware safety (these rules are not optional)

- **Only move the real arms with a person at the arms' power supply.** Cutting the power (12 V for the SO-101) is
  the e-stop. Never send motion to hardware without the user knowing and present.
- Keep `motion_enabled: false` in `stack/robotd.yaml`, and use the `--enable-motion` flag for a session.
  (Tests rely on the file default.)
- robotd opens the ports exclusively (TIOCEXCL). Stop it before running `bench_check`, `torque_off`,
  teleoperation or calibration.
- **Never use `pkill -f` / `pgrep -f` with a bare pattern:** it matches its own shell and other sessions.
  Kill by PID, or bracket the pattern: `pgrep -f "[s]tack.robotd"`.
- Power the arms before plugging in or powering the USB hub.
- Torque must never come on with stale goals. Write Present_Position into Goal_Position, then enable torque
  (see `hold_here` in robotd's hardware layer). lerobot's `torque_disabled()` context manager breaks this rule.
- Safety checks refuse a motion; they never silently clip it. Keep it that way.

## Conventions

- **Geometry comes from robot.yaml only.** Code reads it through `stack.rig.load_rig()`. Never hard-code a
  measurement elsewhere.
- **The rig frame:** the origin is on the work surface, midway between the shoulder-pan axes.
  +x is forward, +y is the robot's left, +z is up, and units are metres and degrees.
- **Don't hand the operating agent facts it is denied:** the robot's measured geometry and the camera's pose
  (`robot.yaml`, `calib/`), and anything else an experiment withholds. Its permission settings deny them; keep
  them out of the documents and tool texts it reads too.
- **Keep `docs/ARCHITECTURE.md` current:** a change to a process, to how they talk, or to a switch or its default
  updates it in the same commit.
- Write each finding down where the next person will look: failure modes in `docs/troubleshooting.md`, design in
  `docs/ARCHITECTURE.md`, calibration in `docs/CALIBRATION.md`.
- Commit messages start with the component (`robotd:`, `safety:`, `docs:`, ...) and say why, not just what.
  A hardware bug fix comes with a regression test (e.g. the `_FakeBus` tests).

## How work is done (sprints)

1. **Plan first.** Before changing anything, write the work up as a sprint plan in
   `docs/sprints/YYYY-MM-DD-<topic>.md` (template: `docs/sprints/README.md`): context, goal, tasks, the
   decisions it needs (with a recommendation each), acceptance criteria, risks. Ask the user to review it.
2. **Execute only after approval.** Work through the approved plan; if the plan has to change, update the file
   and ask again before going on.
3. **Keep the plan as the work log.** Update the sprint file as you go: progress per task, what was found,
   review findings and how they were handled, test results, what is left. It is the record of that sprint.
4. **Hand-offs to other agents** are written in the repo, in `docs/handoffs/YYYY-MM-DD-<agent>-<topic>.md`: what
   is needed, why, the exact format, acceptance, and where the reply goes. Link it from the sprint file.
5. **Consolidate.** When a sprint ends, fold its lasting results into the design docs (`docs/ARCHITECTURE.md`,
   and `docs/RIG.md`'s consolidated record if this checkout has one), and link the sprint file from there.

## Commits

**No commit (and no push) without the user's explicit approval** - each time, for that change. Prepare the
change, show what it contains (files, summary, test and review results) and ask; do not take an earlier
approval as covering later commits.

## Independent review (required for every code change)

Every code change goes through an independent review before it is committed, and before robotd is
restarted with it at the rig. "Code" means anything that runs or configures what runs: `stack/`, `lib/`,
`tools/`, tests, `robot.yaml`, `stack/robotd.yaml` and `agent_ws/` settings. Docs-only changes are exempt.

- The reviewer is a separate agent (a fresh subagent) that did not write the change. Give it the diff, the
  purpose of the change and the files it touches, and ask for findings, not edits: concrete defects with
  `file:line` and a failure scenario, most severe first. Safety-related code (robotd, safety, planner,
  motion, hardware I/O, the runner sandbox, the agent's permissions) gets the closest look.
- The reviewer reads and runs tests only. It never talks to a running robotd (no API calls), never opens
  the serial ports or cameras, and never stops or signals processes.
- The author fixes each finding or records why it is not one, re-runs the tests, and asks for a
  re-review if the fixes are substantial.
- Say in the commit message that the change was reviewed, with any finding deliberately left open, and
  record the review in the sprint file.

## Docs

| Doc | Contents |
|---|---|
| `README.md` | What robot-use is, getting started in the sim and on the real arms |
| `docs/ARCHITECTURE.md` | The processes, how a program becomes motion, the safety layers, and every switch with its default (keep current) |
| `docs/CALIBRATION.md` | Calibrating a rig: each step, what it produces and where it goes (`tools/calibrate.py`) |
| `docs/AGENT_SANDBOX.md` | How the agent's programs are boxed in, per platform |
| `docs/EMBODIMENTS.md` | Beyond the SO-101: a parts-based design for other bodies (a proposal, not built) |
| `docs/troubleshooting.md` | Failure modes seen in practice and their fixes |
| `docs/usb-bandwidth.md`, `docs/camera-calibration.md` | Camera bandwidth limits; how the lenses were measured |
| `docs/environment.md` | A Jetson Orin environment: JetPack, CUDA, torch pins |
| `docs/sprints/` | One plan and work log per sprint (`README.md` has the template) |
| `docs/RIG.md` | This checkout's own robot, machine and records, if present |
