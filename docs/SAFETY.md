# Safety

robot-use moves real arms on the instructions of an AI agent. This page says what you must do, and what the stack
does to keep a motion from doing harm. The design behind it is in [ARCHITECTURE.md](ARCHITECTURE.md).

## What you must do

- **A person at the power supply whenever the arms can move.** Cutting the arms' 12 V is the e-stop: it works
  whatever the software is doing. Keep it within reach.
- **Start read-only.** Motion is off unless you start the stack with `--enable-motion`; `stack/robotd.yaml` keeps
  `motion_enabled: false`.
- **Approve the agent's programs** until you trust the setup: `run_rig.py` asks a person to approve each program on
  the operator page by default (`--approval first_motion`). `--approval none` runs programs unattended; it still
  needs a person at the 12 V.
- **Keep the operator page open.** It shows the cameras, what is running, and a big STOP (Escape). It listens on
  the LAN (port 8772) with no login, so anyone on your network can reach its STOP, approvals and torque-off: run it
  on a network you trust.
- **Calibrate and check before moving.** A wrong `robot.yaml` puts the arms somewhere other than where robotd
  believes them to be. The overlay on the operator page shows it: the drawn arms must lie on the real ones.
- **Power the arms before the USB**, and stop the stack before running other tools on the arms (teleoperation,
  calibration, `bench_check`, `torque_off`): robotd holds the serial ports exclusively.

## What the stack does

| Layer | What it does |
|---|---|
| **The agent cannot reach the hardware** | The agent only writes programs. Its permission settings deny it a shell and the stack, and its programs run in a sandbox (a static check, a Python audit hook, and a kernel box: Landlock on Linux, Seatbelt on macOS) that lets them do nothing but ask for steps ([AGENT_SANDBOX.md](AGENT_SANDBOX.md)). |
| **The twin preview** | By default each program runs in a MuJoCo twin first, and one whose steps fail there is rejected before the real arms see it. The agent may skip the preview for a program (the approver is shown that it did), and a program runs without it when the SO-101 model is not installed. With `--approval none`, nobody sees that before the arms move: robotd's checks below still apply to every step. |
| **robotd checks every step, along the whole path** | Joint limits, the table and the board, the obstacles, the other arm, and the gripper's own shape (its camera, jaws and body) are checked along the planned trajectory, not just at its end. |
| **Refuse, never clip** | A step that fails a check is refused, with the reason. robotd never quietly changes a motion into a different one the agent did not ask for. |
| **The contact monitor** | While a motion runs, a joint that lags its command while drawing high current is taken as a push against something: the motion stops and the arm backs off. |
| **Budgets** | A program has limits on steps, queries, motion time and wall-clock time; one that exceeds them is stopped. |
| **A lease** | Only the program holding the lease can move the arms, and a program that stops renewing it (a crash) loses it: its motion stops. |
| **Torque never comes on with stale goals** | Before torque is enabled, each servo's goal is set to where it is now, so an arm never jumps toward an old target. |
| **Idle rest** | An arm held without a command for 10 minutes goes slowly to its rest pose. |
| **STOP** | The operator page's STOP halts every motion at once. Stopping the stack (Ctrl-C) also halts a running motion first, then leaves the arms holding where they are. |
| **Torque off** | The operator page can turn an arm's torque off (with a confirmation: the arm falls under gravity). |

None of these replaces the person at the 12 V. Software can fail; the power switch cannot.
