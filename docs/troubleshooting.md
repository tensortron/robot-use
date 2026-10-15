# Troubleshooting

Failure modes actually hit during setup, with what fixed them. Most of these fail
**silently** — that's what makes them worth writing down.

## Cameras

### A camera returns 0 frames / `actual 0x0` at every resolution

The camera is **wedged**: enumerated on USB, dead at the V4L2 level. Distinguish
from bandwidth starvation by testing it **alone** — a starved camera works alone,
a wedged one does not.

Escalate in this order:

1. `sudo modprobe -r uvcvideo && sudo modprobe uvcvideo quirks=128` (clears driver state)
2. **Unplug and replug that camera's own cable** ← this is what actually worked
3. Reboot

> Observed twice on the ego camera. A **hub replug was not sufficient**; the
> camera's own connector had to be reseated. If it recurs, suspect that cable or
> hub port: swap the ego camera into port `4.1.2` and a working wrist cam into
> `4.1.3`. If the fault follows the camera it's the camera; if it stays with the
> port it's the port or cable.

### `/dev/videoN` numbering changed

Expected, and harmless if you use the udev symlinks. When a camera drops off and
re-enumerates it takes the next free node — observed going
`video2` → `video3` → `video2` within one session, while the USB path stayed
`1-4.1.3` throughout.

**Never hardcode `/dev/videoN`.** Use `/dev/cam_*` (or by-path).

### The third camera returns 0 frames but the other two are fine

Bandwidth starvation, not a fault. See [usb-bandwidth.md](usb-bandwidth.md).
Confirm by running the camera alone — it will work.

### Cameras record at 10 fps despite `fps: 30`

Missing `fourcc: MJPG`. lerobot defaults to `None`, OpenCV negotiates uncompressed
YUYV, and the camera throttles to 10 fps. **Silent** — the dataset is simply wrong.

### `/dev/v4l/by-id/` shows only one camera

Correct and expected: all three report serial `SN0001`. by-id is unusable here and
its symlinks even straddle different physical devices. Use by-path or `/dev/cam_*`.

### First frames are black

Auto-exposure needs ~2–3 seconds of warm-up. Grab frames for a couple of seconds
before trusting an image. (Also: check the lens caps.)

### Opening several cameras back-to-back starves one

Stagger the opens (~0.7 s apart) so each isochronous reservation settles before
the next claims bandwidth. `tools/identify_cameras.py` does this.

## USB hub power — marginal and transient-sensitive, NOT a hard limit

> **Corrected 2026-09-27.** This section used to say CONFIRMED: that the
> bus-powered hub could not carry four arms, and that three cameras would break it
> even with two. Both claims are contradicted by later measurements (below). What
> the evidence supports is weaker and more useful: the hub is **marginal**, and a
> **transient** can wedge adapters until the hub is power-cycled.
>
> **Updated 2026-09-27 evening:** adapters also wedged **mid-session, three times
> in one evening, with nothing being powered up**. So "a hub that is healthy once
> up" is too optimistic. The leaders have been moved off this hub (see
> "Current wiring" below).
>
> **Updated again 2026-09-27 19:30: the evening's main culprit was the leader
> right's USB cable, not the hub.** Leader right was involved in every incident,
> and it dropped off USB entirely while on a direct Jetson port. After a cable
> swap, a 5-minute soak ran clean. The hub is still marginal, but check the arm
> that keeps appearing before blaming the hub.

**The hub is bus-powered (no power adapter).** A bus-powered hub may supply only
100 mA per downstream port (USB spec; both Genesys hubs report `bMaxPower=100mA`),
while the CH343 adapters request 138 mA and the cameras 500 mA each. Every device
is nominally over budget -- which is why it is marginal. But nominal budget is not
what fails: the full rig runs clean in steady state.

### What was measured, in order

| date | load | result |
|---|---|---|
| 09-04 | 1 arm | 30/30 pings; pose read 0.01 s |
| 09-04 | 2 arms, 24 Hz, 30 s | 1446 reads, 0 errors |
| 09-04 | 4 arms | intermittent; **which** arm failed changed between runs |
| 09-04 | 2 arms + 3 cameras streaming, 25 s | **0 errors** |
| 09-27 | left arms' 12 V switched on | follower left **dropped off USB**; leader left silent |
| 09-27 | after a **hub power-cycle**: 4 arms, 30 s | ~720 reads each, **0 errors** |
| 09-27 | **4 arms + 3 cameras, 5 minutes** | **50,546 reads/frames, 0 errors**, USB 4/4 + 3/3 after |
| 09-27 ~16:55 | idle, first touch after a policy-client run | follower right + leader right **wedged** (`open()` 5.1 s) |
| 09-27 18:49 | after a hub power-cycle: 4 arms + 3 cameras, 30 s | 30/30 PASS, 0 errors -- but cameras only **16-20 fps**, even one at a time |
| 09-27 19:09:55 | **teleop, 98 s in**, healthy loop (24.9 Hz, 6.4 ms work/tick) | leader right: `There is no status packet!` x3, teleop crashed |
| 09-27 ~19:14 | bench_check pinged all four fine; **~1 min later, idle** | follower left + leader right **wedged** (`open()` 10.5 s) |
| 09-27 19:17 | **leaders moved to the Jetson's own ports**, followers alone on the hub | 30/30 PASS, 0 errors; cameras back to **28.5-29.0 fps** |
| 09-27 19:20:05 | teleop, 40 s in, leader right on **direct port 1-4.3** | leader right **dropped off USB** (`USB disconnect`), then `error -32`/`-71`, `unable to enumerate` even after the kernel power-cycled the port; lerobot `Input/output error` |
| 09-27 19:24 | **leader right USB cable swapped**; 4 arms + 3 cameras, **5 minutes** | 29,080 arm reads + 26,486 frames, **0 errors**, no kernel USB events; cameras 29.4 fps |

The 5-minute soak ran every arm at 24.1 Hz and every camera at 29.4 fps
concurrently -- more load than the 09-04 four-arm run that failed. So four arms
is not a ceiling.

### The working hypothesis

A transient pushes a marginal hub over, and the adapters **stay wedged until the
hub is power-cycled** -- re-reading, reseating one arm, or waiting does not clear
it. The clearest instance: follower left dropped off USB moments after its 12 V
was switched on, and only replugging the hub brought the arms back. That also fits
the 09-04 shifting failures, which followed a lot of cable handling.

It is a hypothesis. Servo power-up was the leading candidate for the transient,
but the evening of 09-27 rules out power-up as the **only** trigger: three wedges
came mid-session, one while the arms were idle and nothing was plugged, powered or
opened. Three of the four adapters were hit at different moments, which looked
like the shared hub. **But leader right was in every one of them**, and at 19:20
it failed alone, off the hub, as a physical USB disconnect that would not
re-enumerate. That is a cable or connector fault, not hub power. A new cable fixed
it (5-minute soak clean). The likely mechanism: the leader is moved by hand, which
tugs its USB cable. Whether the other adapters' wedges were collateral from this
arm's faults on the shared hub, or independent hub marginality, is not
established. The kernel logged nothing,
and the wedged adapters stayed enumerated and `active` in sysfs.

The camera rate on 09-27 (16-20 fps before the leaders moved, 28.5-29.0 fps
after, same room and similar image brightness) suggests the arm hub was also
disturbing the cameras. That rests on one run each way, so it is not established.

### Current wiring (since 2026-09-27 19:15)

| device | port | on the bus-powered Genesys hub? |
|---|---|---|
| leader right | `1-4.3` (Realtek hub, Jetson port) | no |
| leader left | `1-4.4` (Realtek hub, Jetson port) | no |
| follower right | `1-4.2.1` | **yes** |
| follower left | `1-4.2.4` | **yes** |

udev keys the arms by USB serial, so `/dev/so101_*` did not change. The
followers are still exposed to the hub. If one goes silent mid-run, move it to a
direct port too, or use a **powered** hub.

### What to do

0. **If one arm keeps appearing in failures, suspect its USB cable first.**
   `journalctl -k | grep -E "disconnect|error -(32|71)"` showing that port is
   conclusive. Swap the cable and soak with `tools/bench_check.py --soak` while
   moving the arm. Leaders are hand-moved, so route their cables with slack.
1. **Power the arms' 12 V first, then connect USB** -- or replug the hub after
   powering them. This is the habit the hypothesis predicts will matter.
2. **If an arm goes silent or vanishes, power-cycle the hub before chasing
   cables.** On 09-27 that was the fix; the cable was not. Signatures: lerobot
   `There is no status packet!` mid-run, or `tools/torque_off.py --status` /
   `bench_check` reporting "opening took 5.1 s" (a USB control timeout).
   Checking also clears nothing: an adapter that pinged fine can wedge a minute
   later.
3. **Verify with load, not pings.** Pings passed in states that later failed.
   A 30 s concurrent read across all arms is the minimum; the soak above is the
   reference.
4. A **powered** hub removes the marginal budget and should make all of this
   moot. Still worth doing.

### How to recognise a hub problem vs a device problem

The tell is that **failures move between devices**. A broken device fails
consistently; contention picks a different loser each run. But note the
09-27 case failed *consistently* on both left arms -- and was still the hub
(wedged, cleared by a power-cycle). So consistent failure does not rule the hub
out; a hub power-cycle is the cheap test that does.

Two symptoms:

- *adapter enumerates, servos silent* -- the CH343's USB chip enumerates at idle
  current but its serial transceiver is not driving the servo bus.
- *device vanishes from USB* -- the whole board has dropped.

An arm that failed five times on 09-04 then ran ~390 flawless transactions once
isolated -- nothing was wrong with the arm itself.

**This does NOT explain the camera dropouts.** See the next section.

### Checks

```bash
# per-device advertised draw
for d in /sys/bus/usb/devices/1-4.*/; do
  echo "$(basename $d) $(cat $d/bMaxPower 2>/dev/null) $(cat $d/product 2>/dev/null)"; done

sudo dmesg -w      # over-current change, or error -71, is conclusive
```

## Camera dropping off USB mid-session — UNEXPLAINED

Symptom: a camera stops delivering, lerobot logs `read failed (status=False)`
repeatedly, then `exceeded maximum consecutive read failures`, and the process
dies on the next `get_observation()`. Teardown shows
`VIDIOC_REQBUFS: errno=19 (No such device)` -- the device left the USB bus, so
this is a disconnect, not a stream error.

Observed on the ego camera (`1-4.1.3`) a handful of times. Recovery needs a
physical replug or `modprobe -r uvcvideo && modprobe uvcvideo`.

**Cause not established.** Two hypotheses were tested and neither held:

| Hypothesis | Test | Result |
|---|---|---|
| Hub current budget | 3 cameras streaming 300 s, arms idle | **No failure** -- 8800 frames each. If current were the cause this should have failed. |
| Servo-motion EMI / ground bounce | teleop with arms driven hard | **No failure** -- did not reproduce |

So it is rare and not reproducible on demand. Do not assume the powered hub
fixes it; that was an unfounded claim, disproved by the soak above.

Remaining candidates, untested: a marginal cable or connector on that camera
(it did need a physical reseat once), thermal shutdown under sustained encoding,
or camera firmware. It is also the camera requesting the most bandwidth, so it
is the most stressed by any mechanism.

**Practical stance:** it is infrequent enough to work through, but it kills the
process when it happens. That matters most during recording: plan for it
before a long data-collection session. If it becomes frequent, swap that
camera's cable first, then the camera.

## Arms

### An arm is intermittently missing or silent

Two distinguishable failure modes, with different fixes:

| Symptom | Meaning | Check |
|---|---|---|
| `/dev/so101_*` missing, `lsusb` shows < 4 CH343 | the adapter fell off USB | hub power, USB cable |
| symlink present, motors do not answer | adapter on USB, servo bus silent | the arm's 12 V supply, servo cable into servo #1 |

Localise it per servo -- the *pattern* tells you where the chain breaks:

```bash
sg dialout -c "./.venv/bin/python tools/diagnose_arm.py /dev/so101_follower_left --rounds 3"
```

- all six answer: bus healthy
- none answer: no 12 V, or the first servo cable is unseated
- `1..N` answer and the rest do not: the chain breaks after servo N

If several devices on the same hub misbehave, read the hub-power section above
before chasing individual cables.

### `Permission denied` on `/dev/ttyACM*`

Not in the `dialout` group **in this session**. `usermod -aG dialout` writes to
`/etc/group` immediately, but existing shells keep their old credentials.

```bash
getent group dialout            # is the user listed? (persistent state)
id -nG | grep dialout           # does THIS shell have it? (session state)
```

Fix: log out and back in. Workaround without re-login: `sg dialout -c "<command>"`.

### `Incorrect status packet!` during teleop

Feetech buses occasionally return corrupted packets under load. Raise
`num_read_retries` (default 2) on the affected arm config.

### Two processes on one bus

The Feetech buses and the cameras tolerate only one reader. Stop
`identify_arms.py` / `identify_cameras.py` before teleop or record.

### Servo transaction hangs (bench_check false failures, torque_off never returns)

**Root cause, 2026-09-27: a wedged CH343 adapter plus an unbounded write in
`scservo_sdk`.** Two faults stack:

1. **The hang: an infinite write timeout.** `scservo_sdk.PortHandler.setupPort`
   opens pyserial with `timeout=0` (reads never block; `rxPacket`'s loop is
   bounded by the packet timeout) but leaves `write_timeout=None`. When an
   adapter stops accepting USB OUT data, `ser.write()` then waits in `select()`
   for the tty to become writable, forever. `clearPort()` is `ser.flush()` =
   `tcdrain()`, which is just as unbounded. Every lerobot/SDK caller inherits this.
2. **The false failures: `PortHandler.is_using`.** `txPacket` sets it and only
   `rxPacket` clears it. A transaction abandoned mid-write leaves it `True`, so
   every later transaction on that PortHandler returns `COMM_PORT_BUSY` in ~1 ms.
   So the later pings were not getting "their reply eaten". They were refused.
   Closing and reopening the port, as the old watchdog did, does not reset the flag.

**The trigger is the adapter, not the code.** On 09-27 around 16:55, both right-hand
adapters (follower right and leader right, hub ports 1-4.2.1/.2) were wedged, while the
left pair on the same hub was healthy.

```text
# read-only probe, tools-style ping loop (scratch script), faulthandler dump at 3 s:
DONE /dev/so101_follower_left fails=0 slow=[]          # pings 0.5-0.8 ms each
Timeout (0:00:03)!                                     # follower_right, then leader_right:
  File ".../serial/serialposix.py", line 640 in write  #   select([...], [fd], [], None)
  File ".../scservo_sdk/port_handler.py", line 64 in writePort
  File ".../scservo_sdk/protocol_packet_handler.py", line 96 in txPacket
  ...  line 221 in ping

# open() only (O_NONBLOCK, no data written), repeated minutes apart:
/dev/so101_follower_right: open 5.104s   # = 5 s USB control-transfer timeout
/dev/so101_leader_right:   open 5.117s   #   (SET_CONTROL_LINE_STATE never ACKed)
/dev/so101_leader_left:    open 0.001s
```

`journalctl -k` showed nothing for those adapters after the 15:44 hub replug,
apart from one `tegra-xusb ... Transfer event TRB DMA ptr not part of current TD
ep_index 4 comp_code 6` at 16:00:17. The adapters stayed enumerated, and sysfs
reported them `active`. The software half reproduces with no hardware, using a pty
whose output buffer is full in place of the wedged adapter, and bench_check's old
`bounded()` plus close/reopen:

```text
ping 1: res=None timed_out=True 3.001s is_using=True
  after close/reopen: thread alive=False is_using=True
ping 2: res=None timed_out=False 0.001s is_using=True
ping 3: res=None timed_out=False 0.000s is_using=True
```

What the brief's hypotheses came to:

- *Serial read with no timeout*: **ruled out for reads**. `timeout=0`, and the
  packet-timeout loop always ends, unless a continuous stream of header-less garbage
  arrives, which was never seen. **It holds for writes** (above).
- *Hub/CH343 wedge*: **this is the trigger**. The adapter wedge is described
  under "USB hub power". Why the adapters wedge is still open.
- *Torque write under load*: not the cause. The hang is in the host's `write()`,
  before any byte reaches a servo.
- *Lock register (55)*: ruled out on the same grounds: the hang is below the
  protocol level. Not separately tested.
- The old torque_off printed only after all six IDs of a port, so a hang on the
  first port looks like "printed nothing".
- "The first arm opened fails most often" is **not explained**. The first arm
  (follower left) was healthy in every run on 09-27.

**Fix (tools/bench_check.py, tools/torque_off.py).** Each port is opened with a 0.3 s
`write_timeout` and a bounded `clearPort`. After an I/O error, `is_using` is reset.
bench_check runs each arm's ping check and load reader in its own subprocess,
killed at a hard deadline (8 s for pings, all arms in parallel), and the whole run
has a watchdog cap of 90 s plus the load test. torque_off has a SIGALRM deadline of
8 s per port and exits 1 with "the USB adapter is not responding". A wedged
adapter is now reported as `adapter not accepting data` / `opening took 5.1 s` with
the `arm_wedged` fix: **power-cycle the arm hub**.

**Diagnose it by hand:** a port whose `open()` takes ~5 s is wedged at the USB
level. Nothing in software will clear that. Power-cycle the hub.

### An arm moved to a different USB port

Fine — arm rules are keyed on serial, not port. Nothing to update.

## Python environment

### `RuntimeError: operator torchvision::nms does not exist`

torch/torchvision CUDA-build mismatch (`+cu130` vs `+cu128`), caused by lerobot's
`[tool.uv.sources]` pinning the cu128 index. Reinstall both from PyPI with
`--index-url https://pypi.org/simple`. See [environment.md](environment.md).

### `OSError: libnvrtc.so.13: cannot open shared object file`

The `nvidia/cu13/lib` directory isn't on the loader path. Fixed by
`_cu13_preload.pth` in site-packages. If it reappears, confirm both
`_cu13_preload.py` and `_cu13_preload.pth` still exist there — a `uv pip install`
that recreates the venv will drop them.

> A `sitecustomize.py` will **not** work: Debian's
> `/usr/lib/python3.12/sitecustomize.py` shadows it.

### `OSError: Could not load this library: .../libtorchcodec_core4.so`

Two possible causes:
- missing system FFmpeg → `sudo apt install ffmpeg` (provides `libavcodec.so.62`,
  matching torchcodec's `core8` probe)
- missing NPP → `uv pip install nvidia-npp`

The traceback names `core4` even when a different probe is the real failure;
diagnose with `ldd .../libtorchcodec_core8.so | grep "not found"`.

### `[NVBLAS] CPU Blas library need to be provided` then `Segmentation fault`

The venv's `_cu13_preload.pth` hook loaded `libnvblas.so.13` along with the
other CUDA 13 libraries. NVBLAS interposes BLAS process-wide and, with no CPU
fallback, crashes the first LAPACK call -- `cv2.calibrateCamera`, and
potentially any numpy linear algebra. The hook now skips `nvblas`; if the venv
is rebuilt, make sure it still does (see environment.md, fix 4).

### `UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7`

Harmless, prints on every CUDA import. The wheel has no `sm_87`; it runs via
sm_80 binary compatibility. See [environment.md](environment.md).

### numpy conflicts with opencv

Reinstalling torch from PyPI drags numpy past lerobot's `<2.3` bound. Pin it back:
`uv pip install "numpy<2.3"`.

## Diagnostic one-liners

```bash
# arms: which serial is on which node
for d in /dev/ttyACM*; do echo -n "$d: "; udevadm info -q property -n $d | grep ID_SERIAL_SHORT; done

# cameras: capture nodes and their USB paths
for d in /dev/video*; do
  udevadm info -q property -n $d | grep -q "ID_V4L_CAPABILITIES=:capture:" && {
    echo -n "$(basename $d): "; udevadm info -a -n $d | grep -m1 'KERNELS=="[0-9]'; }
done

# USB topology and negotiated speeds
lsusb -t

# is a camera alive at the driver level?
v4l2-ctl --device=/dev/cam_ego --all 2>&1 | head    # (needs v4l2-utils; not installed)

# full python stack check
cd <repo> && ./.venv/bin/python -c "
import torch, torchvision, torchcodec
from torchvision.ops import nms
from torchcodec.decoders import VideoDecoder
print(torch.__version__, torchvision.__version__, torchcodec.__version__, torch.cuda.is_available())"
```

## Visualisation over SSH

### Teleop: `Control loop is running slower (23.2 Hz) than the target FPS (25 Hz)`

Seen 2026-09-27: teleop starts, the arms move a little, then this warning repeats.
Cause: `--display_data=true` with no `--display_mode`, `--display_ip` or
`--display_port`. That defaults to rerun, and lerobot calls `rr.spawn()` to open
a local viewer. On the headless Jetson the viewer exits immediately (no
`DISPLAY`/`WAYLAND_DISPLAY`), but `rr.spawn()` does not raise. So teleop carries
on JPEG-encoding every camera frame (~10.7 ms per camera, ~32 ms per tick for
three) into a dead gRPC sink. That leaves no room in the 40 ms budget for four
serial buses.

Fix: drop `--display_data` (and `--display_compressed_images`) from the command.
Leave them out of teleop and record commands. To watch the cameras,
run foxglove mode through an SSH tunnel, and only while checking the setup.

### Foxglove: `Dropping client 127.0.0.1:NNNNN: handshake failed`

The TCP connection arrives (so the tunnel works) but the WebSocket upgrade is
rejected. Measured against a live server:

| Subprotocol offered | Server response |
|---|---|
| `foxglove.sdk.v1` | **101 Switching Protocols** |
| `foxglove.websocket.v1` (legacy) | 400 — "Missing expected sec-websocket-protocol header" |
| none (plain browser navigation) | 400 — same |

`foxglove-sdk` speaks **only** `foxglove.sdk.v1`. Checked across releases
0.19.0 / 0.22.0 / 0.25.3 — all of them — so **there is no legacy-protocol version
to downgrade to**, and the Foxglove app must therefore support it. Downgrading the
SDK is not a fix.

So the cause is the client:
- navigating a browser to `http://<host>:8765` (sends no subprotocol header),
- or something between laptop and Jetson stripping `Sec-WebSocket-Protocol`.

Connect via <https://app.foxglove.dev> → **Open connection** → **Foxglove
WebSocket** → `ws://localhost:8765`, in Chrome/Edge.

To tell "my tunnel/server is broken" from "the Foxglove UI is misconfigured", run
this **on the laptop** (heredoc so nothing wraps-breaks):

```bash
cat > /tmp/fgtest.py <<'PY'
import socket, base64, os

key = base64.b64encode(os.urandom(16)).decode()
req = (
    "GET / HTTP/1.1\r\n"
    "Host: localhost:8765\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "Sec-WebSocket-Protocol: foxglove.sdk.v1\r\n"
    "\r\n"
)
s = socket.create_connection(("127.0.0.1", 8765), timeout=5)
s.sendall(req.encode())
print(s.recv(300).decode(errors="replace").splitlines()[0])
s.close()
PY
python3 /tmp/fgtest.py
```

- `101 Switching Protocols` → tunnel and server fine; the problem is in the Foxglove UI.
- `ConnectionRefusedError` → tunnel not up, or wrong port.
- `400 Bad Request` → a proxy is stripping the subprotocol header; no Foxglove
  setting can fix that. Use the MJPEG bridge instead.

**When in doubt, skip Foxglove**: `tools/mjpeg_bridge.py` serves the same frames
over plain HTTP.

### Rerun hangs or fails over SSH

With no `--display_ip`/`--display_port`, `init_rerun` calls `rr.spawn()`, which
tries to open a viewer window on the Jetson — there is no display. Either use
foxglove, or run the viewer on your laptop and point the Jetson at it with
`--display_mode=rerun --display_ip=<laptop> --display_port=9876` (needs the Jetson
to reach *back* to your machine; reverse-tunnel with `ssh -R` if it can't).

### `failed to set fps=25 (actual_fps=30.0)`

`_validate_fps` requires the camera's driver-reported rate to match exactly, and
these cameras advertise only one MJPG interval (30 fps). Set **camera** `fps: 30`
and control the loop rate separately with `--fps=25`. The two are independent.

### Control loop slower than target, `telemetry` dominating the cadence summary

`display_compressed_images` defaults to `False`, sending raw frames every tick.
Measured at 25.30 ms mean — **80.8% of loop work** — against ~6 ms for all robot
I/O combined. Add `--display_compressed_images=true`, or `--display_data=false`
while recording.

### `exceeded maximum consecutive read failures` mid-run

A camera dropped off the bus. If it is always the *same* camera while others
survive identical treatment, suspect hardware rather than bandwidth — see the
wedged-camera entry above and run the port-swap isolation test.
