# Hardware setup: the bimanual SO-101

The real-arms path, from parts to a first read-only start. Try the simulated rig first (README, "Getting started:
the simulated rig"): it needs none of this.

**Only move the real arms with a person at the arms' power supply.** Cutting the 12 V is the e-stop
([SAFETY.md](SAFETY.md)).

## What you need

| Part | Notes |
|---|---|
| **Two SO-101 follower arms** | Built as the [SO-ARM100 project](https://github.com/TheRobotStudio/SO-ARM100) describes, each with its servo driver board (a USB serial adapter). Leader arms are not needed. |
| **A 12 V supply per arm** (or one that can feed both) | The arms' power, and the e-stop: keep it within reach. |
| **Three USB cameras (UVC)** | One ego camera between the arms, looking down at the work area, and one on each wrist. This rig uses Innomaker U20CAM 1080p modules: the ego at 1280x720, the wrists at 1024x576 (`robot.yaml`, `cameras`). |
| **Wrist camera mounts** | The SO-ARM100 project's 32x32 mm UVC wrist mount. Its pose comes from the arm's URDF, so a different mount needs a different URDF link. |
| **A board and a camera stand** | The arms bolt to a board; the ego camera stands between and behind them. Both go into `robot.yaml` as measurements, and robotd keeps the arms clear of them. |
| **A powered USB hub** | A bus-powered hub can starve the adapters ([troubleshooting.md](troubleshooting.md)). The three cameras share one USB 2.0 bus: see [usb-bandwidth.md](usb-bandwidth.md) for the modes that fit together. |
| **A Linux machine** | x86_64 or aarch64 with **Linux 6.7 or newer**: the agent's programs run in a Landlock box that the real arms require ([AGENT_SANDBOX.md](AGENT_SANDBOX.md)). This rig runs on a Jetson Orin ([environment.md](environment.md)). |
| **The printed checkerboard** | `assets/checkerboard_9x6_25mm.pdf`, for the camera calibration ([CALIBRATION.md](CALIBRATION.md)). |

## 1. Install with the hardware extra

```bash
uv sync --extra dev --extra hardware      # lerobot, pinned, with its Feetech servo support
uv run python tools/fetch_so101_assets.py
uv run python tools/setup_agent_ws.py
```

On a Jetson, don't use `uv sync`: it would replace the CUDA build of torch with PyPI's. Build the environment as
[environment.md](environment.md) says, and use `.venv/bin/python` where these steps say `uv run python`.

Your user needs the serial and camera groups: `sudo usermod -aG dialout,video $USER`, then log in again.

## 2. Power and plug in, in this order

1. **12 V on** to both arms.
2. **Then** the USB hub and the arms' adapters. (Powering the USB side first can leave an adapter in a bad state.)
3. The cameras on the hub.

New servos may need their IDs set once (1 to 6 per arm): lerobot's `lerobot-setup-motors` does it. Arms bought
assembled usually come with them set.

## 3. Give the devices stable names (udev)

The stack opens `/dev/so101_follower_left`, `/dev/so101_follower_right`, `/dev/cam_ego`, `/dev/cam_wrist_left` and
`/dev/cam_wrist_right` (`stack/robotd.yaml`). The kernel's own names (`/dev/ttyACM0`, `/dev/video2`) change between
boots and replugs, so udev rules make the stable ones:

```bash
cp udev/99-so101-arms.rules.template udev/99-so101-arms.rules
cp udev/99-so101-cameras.rules.template udev/99-so101-cameras.rules
uv run python tools/identify_arms.py       # move ONE arm by hand: its serial's column jumps
uv run python tools/identify_cameras.py    # move a wrist or cover a lens: that camera's picture changes
```

Fill each arm's serial and each camera's USB port path into the two `.rules` files (the templates say where), then:

```bash
sudo cp udev/99-so101-*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/so101_* /dev/cam_*
```

The arms are named by their adapter's serial, so they can move between USB ports. The cameras are named by USB
port, so each must stay in its port: label the cables. (Your filled-in `.rules` files are not committed: they name
your devices.)

## 4. Check the machine

```bash
uv run python tools/doctor.py --hardware   # what is missing, and the fix
uv run python tools/bench_check.py         # the arms answer, the cameras stream at their modes
```

## 5. Calibrate, then start read-only

Calibrate with the guided page ([CALIBRATION.md](CALIBRATION.md)), then start the stack without motion and check the
overlay on the operator page: the drawn arms must lie on the real ones.

```bash
uv run python tools/calibrate.py
uv run python tools/run_rig.py             # read-only: robotd reads and plans, moves nothing
```

Only then `--enable-motion`, with a person at the 12 V (README, "Getting started: the real arms").
