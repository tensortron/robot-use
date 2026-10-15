#!/usr/bin/env python
"""Pre-flight check for the bimanual SO-101 bench. Run before every session.

Answers one question -- is the bench fit to use? -- with checks chosen from
what actually went wrong during bring-up, not from what is easy to test:

  1. nothing else is holding the arms or cameras
  2. the user can open the serial ports (dialout)
  3. every expected arm adapter is on USB, identified by SERIAL
  4. every expected camera is on its expected USB PORT (cameras that share a
     serial are told apart by port, so a moved camera or hub breaks this)
  5. every /dev/so101_* and /dev/cam_* symlink resolves
  6. every servo on every arm answers a ping
  7. one frame from each camera at its deployment resolution: saved, checked
     for a black frame (lens cap) and scored for sharpness (focus)
  8. a CONCURRENT LOAD TEST: all arms read at 25 Hz and all cameras streaming,
     at once. This is the check that matters most. Pings passed repeatedly in
     states that then failed under load; the bus-powered hub is marginal and
     only fails when everything is busy.

Expected devices are read from udev/99-so101-*.rules, the same files that
create the symlinks, so re-mapping a device needs no change here.

Every failure carries the fix that has worked before. Results go to a
timestamped directory: report.md, report.json, one JPEG per camera, and a
contact sheet of all three.

    ./.venv/bin/python tools/bench_check.py                 # ~45 s
    ./.venv/bin/python tools/bench_check.py --soak          # 5-minute load test
    ./.venv/bin/python tools/bench_check.py --skip-load     # connectivity only
    ./.venv/bin/python tools/bench_check.py --arms so101_follower_left,so101_leader_left

Every arm is opened in its own subprocess with hard deadlines (8 s for the
ping check), so a wedged adapter is reported as failed instead of hanging the
run or poisoning later checks; the whole run is also capped (90 s + load test).

Exit status: 0 all checks passed, 1 something failed, 2 could not run.
"""

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARM_RULES = REPO / "udev" / "99-so101-arms.rules"
CAM_RULES = REPO / "udev" / "99-so101-cameras.rules"

# Deployment resolutions: the combination that fits the shared USB 2.0 budget
# with all three streaming (docs/usb-bandwidth.md). Checking at any other mode
# would test something the robot never runs.
CAM_MODE = {"cam_ego": (1280, 720), "cam_wrist_left": (1024, 576), "cam_wrist_right": (1024, 576)}

ARM_VID_PID = ("1a86", "55d3")
CAM_VID_PID = ("0c45", "6366")

BLACK_FRAME_MEAN = 12.0     # mean brightness below this: lens cap or dead sensor
BLURRY_SHARPNESS = 60.0     # Laplacian variance below this: out of focus or motion

# Remediation, from what actually worked on this bench (docs/troubleshooting.md).
FIX = {
    "holder": "Stop it first (Ctrl-C the teleop/record/live_view/policy client). Only one process may "
              "open a camera or a servo bus.",
    "dialout": "Log out and back in (usermod -aG dialout takes effect on new sessions), or prefix the "
               "command with: sg dialout -c \"...\"",
    "arm_missing": "Adapter not on USB. Reseat its USB cable at both ends -- that has fixed it "
                   "before. Arms are serial-keyed, so any free port on the arm hub works.",
    "arm_silent": "Adapter is on USB but the servos do not answer: the 12 V side. Check the arm's power "
                  "supply is on. If SEVERAL arms are silent, POWER-CYCLE THE ARM HUB before chasing "
                  "cables -- arms have stayed wedged after their 12 V came on until the hub "
                  "was replugged. Habit: power the arms first, then connect USB.",
    "arm_wedged": "The USB adapter itself stopped responding: opening the port hit the 5 s USB control "
                  "timeout and/or the instruction packet could not be written. This is the hub wedge "
                  "(docs/troubleshooting.md, 'USB hub power'): POWER-CYCLE THE ARM HUB (unplug its upstream "
                  "cable, replug). Reseating one arm or waiting does not clear it.",
    "arm_partial": "Some servos answer and some do not: the daisy chain breaks after the last one that "
                   "answered. Check the cable leaving that servo. tools/diagnose_arm.py localises it.",
    "cam_missing": "Camera not on USB. Check its cable and the hub it is plugged into.",
    "cam_port": "Camera is on USB but not on its expected port, so its /dev/cam_* symlink points "
                "nowhere. A camera (or its hub) moved to another USB port: put it back, or run "
                "tools/identify_cameras.py and update KERNELS in the camera rules. Camera identity IS its "
                "port when the cameras share a serial (cheap UVC cameras often do).",
    "symlink": "The udev rules are not installed or were not triggered: sudo cp udev/*.rules "
               "/etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger",
    "cam_black": "Frame is black: lens cap on, or the sensor is dead.",
    "cam_blurry": "Frame is soft. Check focus -- these are manual-focus lenses, and the intrinsics are "
                  "only valid for the focus they were calibrated at. If the ring moved, recalibrate.",
    "cam_noframe": "Camera opened but delivered no frames: wedged. Replug the camera hub, or "
                   "sudo modprobe -r uvcvideo && sudo modprobe uvcvideo quirks=128",
    "load": "Errors only under concurrent load: the bus-powered hub is marginal. Power-cycle the hub "
            "and re-run. If it recurs, run with fewer arms until the powered hub is in use.",
}


# --------------------------------------------------------------------------- servo I/O
#
# Why arm checks run in subprocesses (docs/troubleshooting.md, "Servo transaction
# hangs"): scservo_sdk opens the port with pyserial's default write_timeout=None,
# so when a CH343 adapter stops accepting USB OUT data (the hub wedge) the
# instruction packet's write() blocks forever in select(). The SDK also marks
# the port busy (PortHandler.is_using) at the start of txPacket and clears it
# only at the end of rxPacket, so once one transaction is abandoned every later
# one on that PortHandler returns COMM_PORT_BUSY at once -- the "false failures".
# A thread cannot be killed; a process can, and its port closes with it.
# Inside the worker, every write is also bounded (write_timeout) so a wedged
# adapter is diagnosed as such instead of just timing out.

SERVO_IDS = range(1, 7)
BAUD = 1_000_000
WRITE_TIMEOUT_S = 0.3        # a 14-byte packet takes ~0.1 ms at 1 Mbps; 0.3 s means wedged
SLOW_OPEN_S = 1.0            # open() on a healthy port takes ~1 ms; ~5 s = USB control timeout
PING_DEADLINE_S = 8.0        # hard kill for one arm's whole ping check (all arms run in parallel)
LOAD_MARGIN_S = 15.0         # hard kill for a load worker: its duration + this
ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION = 56, 2

_workers = []                # live worker Popen objects, so the run watchdog can kill them


def _packet_timeout(self, packet_length):
    """lerobot's patch_setPacketTimeout (feetech.py): the SDK's own formula is wrong."""
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


def _setup_port(self, cflag_baud):
    """PortHandler.setupPort with a finite write_timeout (the SDK leaves it infinite)."""
    import serial
    if self.is_open:
        self.closePort()
    self.ser = serial.Serial(port=self.port_name, baudrate=self.baudrate, bytesize=serial.EIGHTBITS,
                             timeout=0, write_timeout=WRITE_TIMEOUT_S)
    self.is_open = True
    self.ser.reset_input_buffer()
    self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
    return True


def _bounded_clear(self):
    """PortHandler.clearPort is ser.flush() == tcdrain(), which blocks forever if the
    adapter stopped taking data. Wait briefly for the output queue, then drop it."""
    deadline = time.monotonic() + WRITE_TIMEOUT_S
    while self.ser.out_waiting and time.monotonic() < deadline:
        time.sleep(0.001)
    if self.ser.out_waiting:
        self.ser.reset_output_buffer()


def open_servo_port(path, baud=BAUD):
    """(PortHandler, PacketHandler) for a Feetech bus, with every blocking call bounded."""
    import types

    import scservo_sdk as scs
    ph = scs.PortHandler(path)
    ph.setupPort = types.MethodType(_setup_port, ph)
    ph.setPacketTimeout = types.MethodType(_packet_timeout, ph)
    ph.clearPort = types.MethodType(_bounded_clear, ph)
    if not ph.setBaudRate(baud):
        raise OSError(f"unsupported baud rate {baud}")
    return ph, scs.PacketHandler(0)


def close_servo_port(ph):
    try:
        ph.ser.reset_output_buffer()   # nothing queued, so close() cannot wait out closing_wait
        ph.closePort()
    except Exception:
        pass


def servo_txn(ph, fn, *args):
    """Run one SDK transaction. Returns (result, exc). On an I/O error (a write that
    timed out) release the SDK's busy flag so the next transaction is not refused."""
    import serial
    try:
        return fn(*args), None
    except (serial.SerialException, OSError) as e:   # SerialTimeoutException included
        ph.is_using = False
        try:
            ph.ser.reset_output_buffer()
        except Exception:
            pass
        return None, e


def _emit(**kw):
    print(json.dumps(kw), flush=True)


def worker_ping(path):
    """Subprocess body: ping ids 1-6 on one port, one JSON line per event."""
    import scservo_sdk as scs
    t = time.monotonic()
    try:
        ph, pk = open_servo_port(path)
    except Exception as e:
        _emit(event="open_failed", error=f"{type(e).__name__}: {e}")
        return 1
    _emit(event="open", open_s=round(time.monotonic() - t, 3))
    for sid in SERVO_IDS:
        t = time.monotonic()
        for _ in range(2):                              # one retry, as before
            res, exc = servo_txn(ph, pk.ping, ph, sid)
            if exc is not None:
                _emit(event="wedged", id=sid, error=f"{type(exc).__name__}: {exc}")
                close_servo_port(ph)
                return 1
            model, comm, err = res
            if comm == scs.COMM_SUCCESS:
                break
        _emit(event="ping", id=sid, ok=comm == scs.COMM_SUCCESS, model=model, status_error=err,
              comm=pk.getTxRxResult(comm), ms=round((time.monotonic() - t) * 1000, 1))
    close_servo_port(ph)
    _emit(event="done")
    return 0


def worker_load(path, seconds, start_at):
    """Subprocess body: sync-read Present_Position of ids 1-6 at 25 Hz from start_at."""
    import scservo_sdk as scs
    try:
        ph, pk = open_servo_port(path)
    except Exception as e:
        _emit(event="open_failed", error=f"{type(e).__name__}: {e}")
        return 1
    gsr = scs.GroupSyncRead(ph, pk, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for sid in SERVO_IDS:
        gsr.addParam(sid)
    time.sleep(max(0.0, start_at - time.time()))
    ok = bad = 0
    first_fail, last_emit, err_text = None, time.time(), None
    while time.time() - start_at < seconds:
        res, exc = servo_txn(ph, gsr.txRxPacket)
        if exc is None and res == scs.COMM_SUCCESS:
            ok += 1
        else:
            bad += 1
            if first_fail is None:
                first_fail = round(time.time() - start_at, 1)
                err_text = f"{type(exc).__name__}: {exc}" if exc else pk.getTxRxResult(res)
        if time.time() - last_emit >= 1.0:
            _emit(event="progress", ok=ok, bad=bad, first_fail=first_fail, error=err_text)
            last_emit = time.time()
        time.sleep(0.04)                                # 25 Hz, the control-loop rate
    close_servo_port(ph)
    _emit(event="done", ok=ok, bad=bad, first_fail=first_fail, error=err_text)
    return 0


def worker_main(argv):
    mode, path = argv[0], argv[1]
    if mode == "ping":
        return worker_ping(path)
    if mode == "load":
        return worker_load(path, float(argv[2]), float(argv[3]))
    return 2


def run_arm_workers(jobs, deadline_s):
    """Run {name: worker argv} in parallel subprocesses, each killed at deadline_s.

    Returns {name: (events, status, stderr_tail)}; status is "done", "killed"
    (hit the deadline) or "stuck" (did not even die: blocked in the kernel, which
    only happens on a wedged adapter)."""
    import subprocess
    procs = {}
    for name, argv in jobs.items():
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--arm-worker", *argv],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs[name] = p
        _workers.append(p)
    end = time.monotonic() + deadline_s
    results = {}
    for name, p in procs.items():
        try:
            out, err = p.communicate(timeout=max(0.0, end - time.monotonic()))
            status = "done"
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                out, err = p.communicate(timeout=3.0)
                status = "killed"
            except subprocess.TimeoutExpired:
                out, err, status = "", "", "stuck"
        events = []
        for line in (out or "").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        tail = (err or "").strip().splitlines()[-1:] or [""]
        results[name] = (events, status, tail[0][:160])
    return results


# --------------------------------------------------------------------------- utils


def sysfs_usb_devices(vid_pid):
    """{usb_path: serial} for every USB device with this vendor:product."""
    found = {}
    for d in glob.glob("/sys/bus/usb/devices/*/"):
        try:
            vid = Path(d, "idVendor").read_text().strip()
            pid = Path(d, "idProduct").read_text().strip()
        except OSError:
            continue
        if (vid, pid) == vid_pid:
            serial = Path(d, "serial").read_text().strip() if Path(d, "serial").exists() else ""
            found[Path(d).name] = serial
    return found


def parse_rules(path, key):
    """[(match_value, symlink)] from a udev rules file."""
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("SUBSYSTEM"):
            continue
        m = re.search(rf'{key}=="([^"]+)"', line)
        s = re.search(r'SYMLINK\+="([^"]+)"', line)
        if m and s:
            out.append((m.group(1), s.group(1)))
    return out


class Report:
    def __init__(self):
        self.checks = []

    def add(self, section, name, ok, detail, fix=None, data=None):
        self.checks.append(dict(section=section, name=name, ok=bool(ok), detail=detail,
                                fix=None if ok else fix, data=data or {}))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<22} {detail}", flush=True)

    @property
    def passed(self):
        return all(c["ok"] for c in self.checks)


# --------------------------------------------------------------------------- checks

def check_holders(rep):
    """Anything already holding an arm port or a camera node."""
    targets = set()
    for pattern in ("/dev/so101_*", "/dev/cam_*"):
        for link in glob.glob(pattern):
            targets.add(os.path.realpath(link))
    me = os.getpid()
    holders = []
    for pid_dir in glob.glob("/proc/[0-9]*"):
        pid = int(Path(pid_dir).name)
        if pid == me:
            continue
        try:
            fds = os.listdir(f"{pid_dir}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                tgt = os.readlink(f"{pid_dir}/fd/{fd}")
            except OSError:
                continue
            if tgt in targets:
                try:
                    cmd = Path(pid_dir, "cmdline").read_text().replace("\0", " ").strip()[:70]
                except OSError:
                    cmd = "?"
                holders.append(f"PID {pid} ({cmd}) -> {tgt}")
                break
    rep.add("preflight", "no other holder", not holders,
            "nothing holding the devices" if not holders else "; ".join(holders), FIX["holder"])
    return not holders


def check_dialout(rep):
    ports = [os.path.realpath(p) for p in glob.glob("/dev/so101_*")]
    denied = [p for p in ports if not os.access(p, os.R_OK | os.W_OK)]
    ok = bool(ports) and not denied
    detail = ("can open all arm ports" if ok else
              f"no read/write access to {denied}" if denied else "no arm ports to test")
    rep.add("preflight", "dialout permission", ok, detail, FIX["dialout"])
    return ok


def check_arms_usb(rep, arm_rules):
    present = sysfs_usb_devices(ARM_VID_PID)
    by_serial = {s: p for p, s in present.items()}
    arms = {}
    for serial, name in arm_rules:
        path = by_serial.get(serial)
        rep.add("usb", name, path is not None,
                f"serial {serial} at {path}" if path else f"serial {serial} NOT on USB",
                FIX["arm_missing"], {"serial": serial, "usb_path": path})
        if path:
            arms[name] = serial
    extra = set(by_serial) - {s for s, _ in arm_rules}
    if extra:
        rep.add("usb", "unexpected adapters", False, f"adapters not in the udev rules: {sorted(extra)}",
                "A new or swapped adapter: add its serial to udev/99-so101-arms.rules.")
    return arms


def check_cameras_usb(rep, cam_rules):
    present = set(sysfs_usb_devices(CAM_VID_PID))
    cams = {}
    for kernels, name in cam_rules:
        port = kernels.split(":")[0]          # "1-4.1.3:1.0" -> "1-4.1.3"
        ok = port in present
        rep.add("usb", name, ok, f"on port {port}" if ok else f"nothing on expected port {port}",
                FIX["cam_port"] if present else FIX["cam_missing"], {"expected_port": port})
        if ok:
            cams[name] = port
    stray = present - {k.split(":")[0] for k, _ in cam_rules}
    if stray:
        rep.add("usb", "cameras on wrong port", False,
                f"camera(s) on {sorted(stray)}, which no rule expects", FIX["cam_port"])
    return cams


def check_symlinks(rep, names):
    for name in names:
        link = f"/dev/{name}"
        ok = os.path.exists(link)
        rep.add("symlinks", link, ok, f"-> {os.path.realpath(link)}" if ok else "does not resolve",
                FIX["symlink"])


def check_arm_pings(rep, arm_ports, deadline_s=PING_DEADLINE_S):
    """Ping ids 1-6 on every arm, each arm in its own killable subprocess, all in
    parallel. Read-only: nothing is written to any servo register. Bounded by
    deadline_s (+3 s to reap) however the adapters behave."""
    healthy = []
    jobs = {}
    for name, port in arm_ports.items():
        if os.path.exists(port):
            jobs[name] = ["ping", port]
        else:
            rep.add("servos", name, False, f"{port} does not exist (see the USB and symlink checks)",
                    FIX["arm_missing"])
    t0 = time.monotonic()
    results = run_arm_workers(jobs, deadline_s)
    for name, (events, status, err_tail) in results.items():
        port = arm_ports[name]
        ev = {e["event"]: e for e in events}
        pings = [e for e in events if e["event"] == "ping"]
        answered = [e["id"] for e in pings if e["ok"]]
        open_s = ev.get("open", {}).get("open_s")
        slow_open = f"; opening the port took {open_s:.1f} s" if open_s and open_s > SLOW_OPEN_S else ""
        data = {"answered": answered, "worker": status, "open_s": open_s,
                "ping_ms": {e["id"]: e["ms"] for e in pings}}
        if "open_failed" in ev:
            rep.add("servos", name, False, f"could not open {port} ({ev['open_failed']['error']})",
                    FIX["arm_missing"], data)
        elif status != "done" and "open" not in ev:
            rep.add("servos", name, False,
                    f"opening {port} did not return within {deadline_s:.0f} s (worker {status}): "
                    f"adapter wedged", FIX["arm_wedged"], data)
        elif "wedged" in ev:
            rep.add("servos", name, False,
                    f"adapter not accepting data: write to id {ev['wedged']['id']} timed out after "
                    f"{WRITE_TIMEOUT_S} s{slow_open}", FIX["arm_wedged"], data)
        elif status != "done":
            rep.add("servos", name, False,
                    f"{len(answered)}/6 answered before the {deadline_s:.0f} s deadline (worker {status})"
                    f"{slow_open}", FIX["arm_wedged"], data)
        elif "done" not in ev:
            rep.add("servos", name, False, f"check crashed: {err_tail or 'no output'}", None, data)
        else:
            n = len(answered)
            missing = [i for i in SERVO_IDS if i not in answered]
            faults = {e["id"]: e["status_error"] for e in pings if e["ok"] and e["status_error"]}
            detail = f"{n}/6 servos answer" + (f" (missing ids {missing})" if missing else "")
            if faults:
                detail += f"; status error bits {faults} (bit 0 voltage, 2 overheat, 5 overload)"
            detail += slow_open
            ok = n == 6 and not faults and not slow_open
            fix = (None if ok else FIX["arm_wedged"] if slow_open else FIX["arm_silent"] if n == 0
                   else FIX["arm_partial"] if missing else
                   "A servo reports a fault. Power-cycle the arm's 12 V; if it recurs, run "
                   "tools/diagnose_arm.py on it.")
            rep.add("servos", name, ok, detail, fix, data)
            if ok:
                healthy.append(name)
    if jobs:
        print(f"      (servo checks took {time.monotonic() - t0:.1f} s)", flush=True)
    return healthy


def capture_frames(rep, cam_names, out_dir):
    import cv2

    shots = {}
    for name in cam_names:
        w, h = CAM_MODE.get(name, (1280, 720))
        cap = cv2.VideoCapture(f"/dev/{name}", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # YUYV would be 10 fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        frame, t0 = None, time.time()
        while time.time() - t0 < 2.5:            # let auto-exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        time.sleep(0.5)
        if frame is None:
            rep.add("frames", name, False, "opened but delivered no frames", FIX["cam_noframe"])
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        path = out_dir / f"{name}.jpg"
        cv2.imwrite(str(path), frame)
        shots[name] = frame
        data = {"file": path.name, "resolution": f"{aw}x{ah}", "mean_brightness": round(mean, 1),
                "sharpness": round(sharp, 1)}
        if (aw, ah) != (w, h):
            rep.add("frames", name, False, f"got {aw}x{ah}, expected {w}x{h}",
                    "The camera did not accept its deployment mode; check nothing else changed it.", data)
        elif mean < BLACK_FRAME_MEAN:
            rep.add("frames", name, False, f"BLACK frame (mean {mean:.0f}) -> {path.name}",
                    FIX["cam_black"], data)
        elif sharp < BLURRY_SHARPNESS:
            rep.add("frames", name, False, f"soft frame (sharpness {sharp:.0f}) -> {path.name}",
                    FIX["cam_blurry"], data)
        else:
            rep.add("frames", name, True,
                    f"{aw}x{ah}, brightness {mean:.0f}, sharpness {sharp:.0f} -> {path.name}", None, data)

    if shots:                                   # one image to glance at before a session
        tiles, H = [], 360
        for name in ("cam_wrist_left", "cam_ego", "cam_wrist_right"):
            if name not in shots:
                continue
            f = shots[name]
            t = cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H))
            cv2.rectangle(t, (0, 0), (t.shape[1], 30), (20, 24, 30), -1)
            cv2.putText(t, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
            tiles.append(t)
        import numpy as np
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), np.hstack(tiles))
    return shots


def load_test(rep, arm_ports, cam_names, seconds):
    """Everything busy at once -- the condition the marginal hub fails under.

    Arms are read by killable subprocesses (see "servo I/O" above), cameras by
    threads here; both start on the same wall-clock instant."""
    import cv2

    arm_names = list(arm_ports)
    res, first_fail, arm_err = {}, {}, {}
    caps = {}
    for name in cam_names:                    # staggered: each isochronous reservation settles
        w, h = CAM_MODE.get(name, (1280, 720))
        c = cv2.VideoCapture(f"/dev/{name}", cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        caps[name] = c
        time.sleep(0.7)

    start = time.time() + 1.0                 # workers need ~0.3 s to import and open

    def arms():
        jobs = {n: ["load", p, str(seconds), repr(start)] for n, p in arm_ports.items()}
        for name, (events, status, err_tail) in run_arm_workers(jobs, seconds + LOAD_MARGIN_S).items():
            last = ([e for e in events if e["event"] in ("progress", "done")] or [{}])[-1]
            ok, bad = last.get("ok", 0), last.get("bad", 0)
            if last.get("first_fail") is not None:
                first_fail[name] = last["first_fail"]
            if last.get("error"):
                arm_err[name] = last["error"]
            if status != "done":                  # killed mid-run: the port blocked
                bad += 1
                first_fail.setdefault(name, round(time.time() - start, 1))
                arm_err[name] = f"worker {status} after {seconds + LOAD_MARGIN_S:.0f} s"
            elif not events or events[-1]["event"] != "done":
                bad += 1
                arm_err[name] = next((e["error"] for e in events if e["event"] == "open_failed"),
                                     err_tail or "worker crashed")
            res[name] = (ok, bad)

    def cam(name, c):
        time.sleep(max(0.0, start - time.time()))
        ok = bad = 0
        while time.time() - start < seconds:
            if c.read()[0]:
                ok += 1
            else:
                bad += 1
                first_fail.setdefault(name, time.time() - start)
        res[name] = (ok, bad)

    threads = [threading.Thread(target=arms, daemon=True)] if arm_ports else []
    threads += [threading.Thread(target=cam, args=(n, c), daemon=True) for n, c in caps.items()]
    [t.start() for t in threads]
    next_tick = 30
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        if seconds >= 60 and time.time() - start >= next_tick:
            print(f"      ... {int(time.time() - start)}s, camera failures so far: "
                  f"{sorted(first_fail) or 'none'}", flush=True)
            next_tick += 30
    for c in caps.values():
        c.release()

    for name in list(arm_names) + list(cam_names):
        ok, bad = res.get(name, (0, 0))
        rate = ok / seconds if seconds else 0
        unit = "Hz" if name.startswith("so101") else "fps"
        detail = f"{ok} ok, {bad} errors ({rate:.1f} {unit})"
        if name in first_fail:
            detail += f", first failure at {first_fail[name]:.0f}s"
        if name in arm_err:
            detail += f" ({arm_err[name]})"
        rep.add("load", name, bad == 0 and ok > 0, detail, FIX["load"],
                {"ok": ok, "errors": bad, "rate": round(rate, 1), "first_failure_s": first_fail.get(name),
                 "error": arm_err.get(name)})


# --------------------------------------------------------------------------- output

def write_report(rep, out_dir, args, started, elapsed):
    verdict = "READY" if rep.passed else "NOT READY"
    failed = [c for c in rep.checks if not c["ok"]]
    lines = [f"# Bench check -- {verdict}", "",
             f"{started:%Y-%m-%d %H:%M:%S} | {elapsed:.0f} s | load test "
             f"{'skipped' if args.skip_load else f'{args.load_seconds:.0f} s'} | arms opened: "
             f"{', '.join(args.arms_checked) or 'none'}", ""]
    if failed:
        lines += ["## What to fix", ""]
        seen = set()
        for c in failed:
            lines.append(f"- **{c['name']}** -- {c['detail']}")
            if c["fix"] and c["fix"] not in seen:
                lines.append(f"  - {c['fix']}")
                seen.add(c["fix"])
        lines.append("")
    section_title = {"preflight": "Pre-flight", "usb": "USB enumeration", "symlinks": "Device symlinks",
                     "servos": "Servo pings", "frames": "Camera frames", "load": "Concurrent load test",
                     "run": "Run"}
    for sec, title in section_title.items():
        rows = [c for c in rep.checks if c["section"] == sec]
        if not rows:
            continue
        lines += [f"## {title}", "", "| | check | result |", "|---|---|---|"]
        lines += [f"| {'PASS' if c['ok'] else '**FAIL**'} | `{c['name']}` | {c['detail']} |" for c in rows]
        lines.append("")
    if (out_dir / "contact_sheet.jpg").exists():
        lines += ["## Frames", "", "![contact sheet](contact_sheet.jpg)", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "report.json").write_text(json.dumps(
        {"verdict": verdict, "started": started.isoformat(), "elapsed_s": round(elapsed, 1),
         "load_seconds": 0 if args.skip_load else args.load_seconds,
         "arms_checked": args.arms_checked, "checks": rep.checks}, indent=2))
    return verdict


def main():
    if sys.argv[1:2] == ["--arm-worker"]:        # internal: one arm's check, run by run_arm_workers
        return worker_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="results directory (default $ROBOT_USE_HOME/bench_checks/<timestamp>)")
    ap.add_argument("--load-seconds", type=float, default=30.0)
    ap.add_argument("--soak", action="store_true", help="5-minute load test instead of 30 s")
    ap.add_argument("--skip-load", action="store_true", help="connectivity and frames only")
    ap.add_argument("--arms", help="comma-separated arm names to open (default: every arm in the udev "
                                   "rules), e.g. so101_follower_left,so101_leader_left")
    ap.add_argument("--port", action="append", default=[], metavar="NAME=PATH",
                    help="testing: check arm NAME at PATH instead of /dev/NAME (repeatable)")
    args = ap.parse_args()
    if args.soak:
        args.load_seconds = 300.0
    # Hard bound on the whole run: every arm step already has its own deadline, but a
    # camera read or anything else unforeseen must not be able to hang a pre-flight check.
    budget_s = 90.0 + (0 if args.skip_load else args.load_seconds)

    started = datetime.now()
    sys.path.insert(0, str(REPO))
    from stack import paths
    out_dir = Path(args.out or paths.base() / "bench_checks" / started.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ARM_RULES.exists() or not CAM_RULES.exists():
        print(f"[FATAL] udev rules not found under {REPO / 'udev'}", file=sys.stderr)
        return 2
    arm_rules = parse_rules(ARM_RULES, r"ATTRS\{serial\}")
    cam_rules = parse_rules(CAM_RULES, "KERNELS")
    arm_names = [n for _, n in arm_rules]
    cam_names = [n for _, n in cam_rules]
    overrides = dict(o.split("=", 1) for o in args.port)
    checked = args.arms.split(",") if args.arms else arm_names
    unknown = [n for n in checked + list(overrides) if n not in arm_names]
    if unknown:
        print(f"[FATAL] not arms in the udev rules: {unknown} (known: {arm_names})", file=sys.stderr)
        return 2
    arm_ports = {n: overrides.get(n, f"/dev/{n}") for n in arm_names if n in checked}
    args.arms_checked = list(arm_ports)

    rep = Report()
    t0 = time.time()

    def overrun():
        print(f"\n[bench] ABORT: run exceeded its {budget_s:.0f} s bound", file=sys.stderr, flush=True)
        for p in _workers:
            try:
                p.kill()
            except Exception:
                pass
        rep.add("run", "bounded run", False, f"aborted after {budget_s:.0f} s: a device call blocked",
                "Something blocked past every per-step deadline. Note which step was running, "
                "power-cycle the hubs and re-run.")
        try:
            write_report(rep, out_dir, args, started, time.time() - t0)
        finally:
            os._exit(1)

    watchdog = threading.Timer(budget_s, overrun)
    watchdog.daemon = True
    watchdog.start()
    print(f"[bench] {len(arm_names)} arms, {len(cam_names)} cameras expected -> {out_dir}\n", flush=True)

    print("pre-flight")
    free = check_holders(rep)
    check_dialout(rep)
    print("\nUSB enumeration")
    check_arms_usb(rep, arm_rules)
    check_cameras_usb(rep, cam_rules)
    print("\nsymlinks")
    check_symlinks(rep, arm_names + cam_names)

    if not free:
        print("\n[bench] devices are held by another process; skipping the checks that open them.")
    else:
        print("\nservo pings")
        healthy = check_arm_pings(rep, arm_ports)
        print("\ncamera frames")
        shots = capture_frames(rep, [n for n in cam_names if os.path.exists(f"/dev/{n}")], out_dir)
        if not args.skip_load:
            print(f"\nconcurrent load test ({args.load_seconds:.0f} s, everything at once)")
            load_test(rep, {n: arm_ports[n] for n in healthy}, list(shots), args.load_seconds)

    watchdog.cancel()
    verdict = write_report(rep, out_dir, args, started, time.time() - t0)
    n_fail = sum(1 for c in rep.checks if not c["ok"])
    print(f"\n[bench] {verdict}: {len(rep.checks) - n_fail}/{len(rep.checks)} checks passed"
          f" -> {out_dir}/report.md")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
