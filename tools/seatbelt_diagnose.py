#!/usr/bin/env python3
"""macOS only: find which part of the program sandbox's Seatbelt profile stops Python from starting.

    uv run python tools/seatbelt_diagnose.py        # writes seatbelt_diagnose-<host>-<date>.txt here
    uv run python tools/seatbelt_diagnose.py --all  # also round 1's cases (each group left out, each alone)

It starts Python under the profile the runner uses (stack/runner/sandbox.py) and records, for each case, whether
Python started, its exit and output, the kernel's Seatbelt denials for that process (`log show`) and the abort
reason from the crash report macOS writes for it (matched by pid). Round 2 (default; the first Mac run showed the
read group alone stops dyld finding its shared cache): the full profile plus one extra readable place per case, a
deny-list variant, `(subpath "/")` and unfiltered-read controls, and Apple's dyld-support.sb. With --all also round
1's cases: only `(allow default)`, each rule group left out in turn, and each group alone. Three targets: a bare
interpreter, a program's imports, and the runner's own child start-up (its imports and the sandbox_check it makes).
Controls first: sandbox-exec itself, /bin/echo under the read group, and whether this account can read the sandbox
log at all. Changes nothing; touches no hardware. Send the report back.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TARGETS = {
    "bare": "print('STARTED')",
    "imports": ("import sys; sys.path[:0] = [{lib!r}, {repo!r}]\n"
                "import numpy, cv2, robot_use\nfrom stack.rig import camera_model, rig\nprint('STARTED')"),
    "child": ("import sys; sys.path[:0] = [{lib!r}, {repo!r}]\n"
              "import stack.runner.child\nfrom stack.runner import sandbox\n"
              "print(sandbox.apply_in_child('seatbelt', 'x')); print('STARTED')"),
}
ENV = {"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}           # as the runner gives a program
REPORTS = os.path.expanduser("~/Library/Logs/DiagnosticReports")
# Round 2 (the first Mac report: the read group alone stops dyld finding its shared cache, and this account cannot
# read the sandbox log): the full profile plus ONE extra readable place each, to find the one dyld needs.
READ_CANDIDATES = ("/private/preboot", "/System/Cryptexes", "/System/Volumes/Preboot", "/private/var/db",
                   "/private/var", "/private", "/usr", "/Library", "/Volumes", "/dev", "/.vol", "/Applications", "/")


def run_one(cmd: list[str]) -> dict:
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(REPO), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        err += "\nTIMED OUT after 60 s"
    return {"pid": p.pid, "exit": p.returncode, "started": "STARTED" in out, "s": round(time.time() - t0, 2),
            "stdout": out[-800:], "stderr": err[-1500:], "t0": t0}


def denials(pid: int, since: float) -> list[str]:
    """The kernel's Seatbelt denials for this pid, from the unified log."""
    start = datetime.datetime.fromtimestamp(since - 1).strftime("%Y-%m-%d %H:%M:%S")
    try:
        r = subprocess.run(["/usr/bin/log", "show", "--start", start, "--style", "compact", "--predicate",
                            'sender == "Sandbox" OR eventMessage CONTAINS "Sandbox:"'],
                           capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [f"(log show failed: {e})"]
    hits = [ln.strip() for ln in r.stdout.splitlines() if f"({pid})" in ln or f"[{pid}]" in ln]
    return hits[:60] + ([f"... {len(hits) - 60} more"] if len(hits) > 60 else [])


def crash_reports(pid: int, since: float, wait_s: float = 6.0) -> list[str]:
    """Termination details from the crash report macOS writes for this pid (polled: ReportCrash is slow)."""
    deadline = time.time() + wait_s
    while True:
        for f in glob.glob(os.path.join(REPORTS, "*.ips")):
            try:
                if os.path.getmtime(f) < since - 1:
                    continue
                text = Path(f).read_text(errors="replace")
            except OSError:
                continue
            header, _, body = text.partition("\n")
            try:
                d = json.loads(body)
            except ValueError:
                continue
            if d.get("pid") != pid:
                continue
            out = [f"{os.path.basename(f)}: termination={json.dumps(d.get('termination', {}))[:700]}"]
            if d.get("asi"):
                out.append(f"  asi={json.dumps(d['asi'])[:900]}")
            thread = next((t for t in d.get("threads", []) if t.get("triggered")), None)
            if thread:
                images = d.get("usedImages", [])
                frames = []
                for fr in thread.get("frames", [])[:16]:
                    img = images[fr["imageIndex"]].get("name", "?") if fr.get("imageIndex", -1) < len(images) else "?"
                    frames.append(f"{fr.get('symbol', '?')}({img})")
                out.append("  crashed thread: " + " <- ".join(frames))
            return out
        if time.time() > deadline:
            return ["(no crash report for this pid)"]
        time.sleep(0.5)


def profile_text(sandbox, paths, exe, skip=(), read_lines=None) -> str:
    """The runner's profile, optionally with the read group's lines replaced."""
    rules = sandbox.seatbelt_rules(paths, exe)
    if read_lines is not None:
        rules["read"] = read_lines
    return "\n".join(["(version 1)", "(allow default)"]
                     + [ln for g in sandbox.SEATBELT_GROUPS if g not in skip for ln in rules[g]]) + "\n"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"({e})"


def main() -> int:
    if sys.platform != "darwin":
        print("macOS only")
        return 2
    from stack.runner import sandbox
    paths = sandbox.read_paths(extra=())
    groups = sandbox.SEATBELT_GROUPS
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    report = Path.cwd() / f"seatbelt_diagnose-{socket.gethostname().split('.')[0]}-{stamp}.txt"
    home = os.path.expanduser("~")
    q = sandbox._sb_quote
    allowed = " ".join(f"(subpath {q(p)})" if os.path.isdir(p) else f"(literal {q(p)})"
                       for p in paths if os.path.exists(p))
    # case: (name, skip groups or None for unboxed, read lines replacing the read group or None)
    cases = [("unboxed", None, None), ("full profile", (), None)]
    for c in READ_CANDIDATES:
        extra = f'(subpath "{c}")' if c != "/" else '(literal "/")'
        cases.append((f"full + read {c}", (), ["(deny file-read-data)", f"(allow file-read-data {allowed} {extra})"]))
    # controls: is the file dyld needs reachable by any path filter at all?
    cases.append(("full + read subpath /", (),
                  ["(deny file-read-data)", f'(allow file-read-data {allowed} (subpath "/"))']))
    cases.append(("full + read unfiltered", (), ["(deny file-read-data)", "(allow file-read-data)"]))
    # Apple's own rules for a process's dyld cache, imported after the deny so its allows win
    cases.append(("full + import dyld-support", (),
                  ["(deny file-read-data)", f"(allow file-read-data {allowed})", '(import "dyld-support.sb")']))
    cases.append(("full + read deny-list", (),                 # the other design: deny the user's places only
                  [f'(deny file-read-data (subpath "{home}") (subpath "/Users") (subpath "/private/var/folders") '
                   '(subpath "/private/tmp") (subpath "/Volumes"))', f"(allow file-read-data {allowed})"]))
    if "--all" in sys.argv:                                     # round 1's cases
        cases += [("only (allow default)", groups, None)]
        cases += [(f"without {g}", (g,), None) for g in groups]
        cases += [(f"only {g}", tuple(x for x in groups if x != g), None) for g in groups]
    summary = []
    with open(report, "w") as out:
        out.write(f"seatbelt diagnose {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        sw = sh(["sw_vers"]).replace("\n", "; ")
        out.write(f"macOS {platform.mac_ver()[0]} {platform.machine()}; sw_vers: {sw}\n")
        out.write(f"SIP: {sh(['csrutil', 'status'])}\n")
        exe = sys.executable
        out.write(f"python {sys.version.split()[0]} {exe} (symlink: {os.path.islink(exe)})"
                  f" -> {os.path.realpath(exe)}\n")
        out.write(f"base_prefix {sys.base_prefix}\nexecs allowed {sandbox._pythons(exe)}\n")
        out.write("read paths:\n" + "".join(f"  {p}\n" for p in paths))
        out.write("\nfull profile:\n" + sandbox.seatbelt_profile(paths, exe) + "\n")
        out.write("\n----- /System/Library/Sandbox/Profiles/dyld-support.sb\n"
                  + sh(["cat", "/System/Library/Sandbox/Profiles/dyld-support.sb"])[:4000] + "\n")
        out.write("----- profiles mentioning cryptex: "
                  + sh(["sh", "-c", "grep -l -i cryptex /System/Library/Sandbox/Profiles/*.sb | head -20"]) + "\n")
        out.write("----- mount\n" + sh(["mount"]) + "\n")
        out.write("----- ls -la /System/Cryptexes /private/preboot\n"
                  + sh(["ls", "-la", "/System/Cryptexes", "/private/preboot"]) + "\n")

        # controls: sandbox-exec itself, and whether this account can read Seatbelt denials from the log
        r = run_one(["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/bin/echo", "STARTED"])
        line = f"{'control: sandbox-exec /bin/echo':40s} started={r['started']!s:5s} exit={r['exit']}"
        summary.append(line)
        out.write(f"\n===== {line}\n{r['stderr']}\n")
        r = run_one(["/usr/bin/sandbox-exec", "-p", profile_text(sandbox, paths, exe,
                                                                 tuple(g for g in groups if g != "read")),
                     "/bin/echo", "STARTED"])
        line = f"{'control: /bin/echo under the read group':40s} started={r['started']!s:5s} exit={r['exit']}"
        summary.append(line)
        out.write(f"\n===== {line}\n{r['stderr']}\n")
        deny_hosts = '(version 1)(allow default)(deny file-read-data (literal "/etc/hosts"))'
        r = run_one(["/usr/bin/sandbox-exec", "-p", deny_hosts, "/bin/cat", "/etc/hosts"])
        time.sleep(1.5)
        d = denials(r["pid"], r["t0"])
        ok = any("deny" in x for x in d)
        line = f"{'control: denial log readable':40s} {'yes' if ok else 'NO (deny lists below may be empty)'}"
        summary.append(line)
        out.write(f"\n===== {line}\n" + "".join(f"  deny: {x}\n" for x in d))
        out.flush()

        for name, skip, read_lines in cases:
            for tname, tcode in TARGETS.items():
                print(f"-- {name}: {tname} ...", flush=True)
                try:
                    code = tcode.format(lib=str(REPO / "lib"), repo=str(REPO))
                    cmd = [exe, "-c", code]
                    if skip is not None:
                        cmd = ["/usr/bin/sandbox-exec", "-p", profile_text(sandbox, paths, exe, skip, read_lines), *cmd]
                    r = run_one(cmd)
                    line = f"{name:36s} {tname:8s} started={r['started']!s:5s} exit={r['exit']} {r['s']} s"
                    summary.append(line)
                    out.write(f"\n===== {line}\n")
                    if r["stderr"].strip():
                        out.write("stderr: " + r["stderr"] + "\n")
                    if not r["started"] and skip is not None and tname == "bare":
                        for c in crash_reports(r["pid"], r["t0"]):
                            out.write(f"  crash: {c}\n")
                        for x in denials(r["pid"], r["t0"]):
                            out.write(f"  deny: {x}\n")
                except Exception:
                    line = f"{name:36s} {tname:8s} TOOL ERROR"
                    summary.append(line)
                    out.write(f"\n===== {line}\n{traceback.format_exc()}\n")
                out.flush()
        out.write("\n===== summary\n" + "\n".join(summary) + "\n")
    print("\n".join(summary) + f"\n\nreport: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
