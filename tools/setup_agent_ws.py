#!/usr/bin/env python3
"""Write the operating agent's settings for this machine, from the templates in agent_ws.

    python tools/setup_agent_ws.py            # write agent_ws/.agents/settings.json and the pi permissions,
                                              # and copy in the starter skills the agent does not have yet
    python tools/setup_agent_ws.py --check    # only say whether the settings are up to date (exit 1 if not);
                                              # names starter skills not copied yet, but copies nothing (exit 0)
    python tools/setup_agent_ws.py --no-starter-skills    # the settings only

Claude Code's read rules need absolute paths for anything outside agent_ws, so the settings are rendered for where
this repository and the home folder are, instead of being committed with one machine's paths (sprint 2026-10-13,
D6). From `agent_ws/.agents/settings.template.json` ({REPO}, {HOME}) it writes `settings.json` (gitignored), with a
Read deny added for:

- every top-level entry of the home folder (dotfiles and folders alike), except the folders on the way to the
  repository: keys, tokens, other people's files, other agents' memories;
- every entry beside the repository (other projects in the same folder);
- the data folders (robotd's logs, programs and episodes, the sim's; stack/paths.py, ROBOT_USE_HOME).

Folders made later are not covered until this is run again. The pi permission system's allowlist
(`agent_ws/.pi/extensions/pi-permission-system/config.template.json`) is rendered the same way. Codex has no read
rules (its permissions are looser: see the README).

Starter skills (`skills/<name>/`, sprint 2026-10-15-starter-skill): each is copied into `agent_ws/.agents/skills/`
when the agent has no skill of that name - never over its own - as a draft, with one history line saying where it
came from. From then on it is the agent's, under the skill lifecycle in agent_ws/AGENTS.md. A starter skill to be
copied that holds a symbolic link is refused, before anything is copied (a link could point the agent at a file it
is denied); a starter skill folder that is itself a link is not a starter skill. The agent's settings deny it the
`skills/` folder itself: the starter skill states the grasp geometry that some experiments withhold.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLAUDE_TEMPLATE = REPO / "agent_ws/.agents/settings.template.json"
CLAUDE_OUT = REPO / "agent_ws/.agents/settings.json"
PI_TEMPLATE = REPO / "agent_ws/.pi/extensions/pi-permission-system/config.template.json"
PI_OUT = REPO / "agent_ws/.pi/extensions/pi-permission-system/config.json"
STARTER_SKILLS = REPO / "skills"
AGENT_SKILLS = REPO / "agent_ws/.agents/skills"
SEEDED_NOTE = ("Seeded from robot-use's starter skills (skills/{name}): validated on a reference rig, not on this "
               "robot. A draft: use it, refine it and validate it here under the skill lifecycle.")

# Never rendered without these, whatever the template says: the template sits in the agent's own folder, so a
# weakened template must not become weaker settings (review of sprint 2026-10-13 task 7). {REPO} as in the template.
REQUIRED_DENY = ("Bash", "Read(/{REPO}/robot.yaml)", "Read(/{REPO}/calib/**)", "Read(/{REPO}/tools/**)",
                 "Read(/{REPO}/docs/**)", "Read(/{REPO}/skills/**)", "Edit(/{REPO}/skills/**)",
                 "Read(/{REPO}/stack/twin_mujoco/**)", "Read(/{REPO}/stack/robotd/**)",
                 "Edit(/{REPO}/stack/**)", "Edit(/{REPO}/lib/**)", "Edit(/{REPO}/tools/**)", "Edit(/{REPO}/robot.yaml)",
                 "Edit(/{REPO}/agent_ws/.agents/settings.json)", "Edit(/{REPO}/agent_ws/.claude/settings.json)",
                 "Edit(/{REPO}/agent_ws/.agents/settings.template.json)",
                 "Edit(/{REPO}/agent_ws/.claude/settings.template.json)", "Edit(/{REPO}/agent_ws/.pi/**)",
                 "Edit(/{REPO}/agent_ws/.codex/**)", "Edit(/{REPO}/agent_ws/.mcp.json)")
ALLOWED_ALLOW = ("mcp__so101",)
# The templates as reviewed. A template that differs is refused, so nothing in it (hooks, a permission mode, an env,
# a dropped deny; pi's yoloMode or an allow) reaches the agent unreviewed, however it was changed. To change a
# template: edit it, have the change reviewed, then update its hash here (this file is Edit-denied to the agent).
TEMPLATE_SHA256 = {
    "agent_ws/.agents/settings.template.json": "11814e6b4a8deb6ce607dc3a24ee0a77a81f3bd21570c1bca4666e5cfa245e11",
    "agent_ws/.pi/extensions/pi-permission-system/config.template.json":
        "af46456fb8660a81b26073114f16080eccbed4b420e52413f181e2cc76c1dcb5",
}
ALLOWED_DIRS = ("{REPO}/lib/robot_use", "{REPO}/stack/rig")
GLOB = set("*?[]{}()\\")                          # characters that would make a path a pattern (or break a rule)


class TemplateError(ValueError):
    pass


# what the operating agent must always be able to read (its brief, its library, the robot described in words)
MUST_READ = ("agent_ws/AGENTS.md", "agent_ws/.agents/skills/.gitkeep", "lib/robot_use/__init__.py",
             "stack/rig/embodiment.md")


def _rule_rx(path_pattern: str):
    """Claude Code's gitignore-style matching: `**` crosses `/`, `*` does not."""
    import re
    return re.compile(re.escape(path_pattern).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
                      + "$")


def _locks_out(rule: str, repo: Path) -> bool:
    """Whether a Read deny would hide something the agent must read (e.g. the template's /tmp rule for a repository
    cloned under /tmp)."""
    if not rule.startswith("Read(//"):
        return False
    rx = _rule_rx(rule[len("Read(/"):-1])
    return any(rx.match(str(repo / f)) for f in MUST_READ)


def _template_text(path: Path) -> str:
    import hashlib
    raw = path.read_bytes()
    want = TEMPLATE_SHA256.get(str(path.relative_to(REPO)) if path.is_relative_to(REPO) else "", None)
    got = hashlib.sha256(raw).hexdigest()
    if want != got:
        raise TemplateError(f"{path.name} is not the reviewed template (sha256 {got[:12]}..., expected "
                            f"{(want or 'none')[:12]}...): have the change reviewed, then update TEMPLATE_SHA256 in "
                            "tools/setup_agent_ws.py")
    return raw.decode()


def _abs_rule(kind: str, path: Path, tree: bool) -> str:
    """Claude Code's absolute form: //path (a path starting at the file system root)."""
    return f"{kind}(/{path}{'/**' if tree else ''})"


def _deny_entry(e: Path, repo: Path, out: list, skipped: list):
    if GLOB & set(str(e)):
        skipped.append(e)                         # cannot be written as a literal rule
        return
    if e.is_symlink() and e.is_dir():
        out.append(_abs_rule("Read", e, True))    # the link and its contents, always (rules match the path as given)
        out.append(_abs_rule("Read", e, False))
        target = e.resolve()
        if target == repo or target in repo.parents or repo in target.parents:
            return                                # its target is the repository, above it or in it: not that rule,
                                                  # which would lock the agent out (the repo's own rules cover it)
        if GLOB & set(str(target)):
            skipped.append(target)
        else:
            out.append(_abs_rule("Read", target, True))
    elif e.is_dir():
        out.append(_abs_rule("Read", e, True))
    else:
        out.append(_abs_rule("Read", e, False))


def generated_denies(repo: Path, home: Path, skipped: list | None = None) -> list[str]:
    """Read denies for every entry of the home folder and of each folder between it and the repository, except
    the way to the repository, plus the data folders."""
    from stack import paths

    out, skipped = [], skipped if skipped is not None else []
    if repo == home or repo in home.parents:
        raise TemplateError(f"the repository ({repo}) must be inside a folder of its own, not the home folder")
    on_path = {repo, *repo.parents}
    levels = [home] + [a for a in reversed(repo.parents) if a != home and home in a.parents]
    if home not in repo.parents and repo != home:
        levels = [home, repo.parent]              # the repo lives outside home: its neighbours and home
    for level in levels:
        try:
            for e in sorted(level.iterdir()):
                if e not in on_path:
                    _deny_entry(e, repo, out, skipped)
        except OSError:
            pass
    for d in paths.all_dirs():                       # read and edit: the agent's record is not the agent's to rewrite,
        if GLOB & set(str(d)):                       # even if ROBOT_USE_HOME were inside its working folder
            skipped.append(d)
        else:
            out += [_abs_rule("Read", d, True), _abs_rule("Edit", d, True)]
    return out


def _sub(text: str, repo: Path, home: Path) -> str:
    """Substitute inside JSON strings: escaped, and refused if the path would read as a pattern."""
    for name, path in (("{REPO}", repo), ("{HOME}", home)):
        if name in text and GLOB & set(str(path)):
            raise TemplateError(f"{path} contains a pattern character ({''.join(sorted(GLOB))}); move it")
        text = text.replace(name, json.dumps(str(path))[1:-1])
    return text


def render_claude(repo: Path, home: Path, skipped: list | None = None) -> dict:
    s = json.loads(_sub(_template_text(CLAUDE_TEMPLATE), repo, home))
    s.pop("_comment", None)
    perms = s["permissions"]
    deny = perms["deny"]
    req = [r.replace("{REPO}", str(repo)) for r in REQUIRED_DENY]
    if missing := [r for r in req if r not in deny]:
        raise TemplateError(f"the template lacks required denies: {missing}")
    if set(perms.get("allow", [])) - set(ALLOWED_ALLOW):
        raise TemplateError(f"the template allows more than {ALLOWED_ALLOW}: {perms['allow']}")
    if set(perms.get("additionalDirectories", [])) - {d.replace("{REPO}", str(repo)) for d in ALLOWED_DIRS}:
        raise TemplateError(f"the template adds directories beyond {ALLOWED_DIRS}")
    for rule in generated_denies(repo, home, skipped):
        if rule not in deny:
            deny.append(rule)
    perms["deny"] = [r for r in deny if not _locks_out(r, repo)]
    return s


def render_pi(repo: Path) -> str:
    text = _sub(_template_text(PI_TEMPLATE), repo, Path.home().resolve())
    perm = json.loads(text)["permission"]                         # valid JSON, and still deny-by-default
    if perm.get("*") != "deny" or any(perm.get(t, {}).get("*") != "deny" for t in ("read", "grep", "find")):
        raise TemplateError("the pi template must deny by default (\"*\": \"deny\") overall and for read/grep/find")
    return text


def drift() -> dict | None:
    """How this machine's rendered settings differ from what the templates give now, for the launchers: None when
    current, else {"files": [stale files], "added": [deny rules], "removed": [deny rules]}. Read-only: rendering
    is setup_agent_ws's own job (main), run deliberately, never a side effect of starting the stack."""
    claude = json.dumps(render_claude(REPO, Path.home().resolve()), indent=2, ensure_ascii=False) + "\n"
    files = [str(path.relative_to(REPO)) for path, text in ((CLAUDE_OUT, claude), (PI_OUT, render_pi(REPO)))
             if not path.exists() or path.read_text() != text]
    if not files:
        return None
    try:
        before = set(json.loads(CLAUDE_OUT.read_text())["permissions"]["deny"])
    except (OSError, ValueError, KeyError):
        before = set()
    after = set(json.loads(claude)["permissions"]["deny"])
    return {"files": files, "added": sorted(after - before), "removed": sorted(before - after)}


def starter_skills(src: Path = STARTER_SKILLS) -> list[Path]:
    """The starter skill folders: each directory in `src` with a SKILL.md."""
    if not src.is_dir():
        return []
    return sorted(d for d in src.iterdir() if d.is_dir() and not d.is_symlink() and not d.name.startswith((".", "_"))
                  and (d / "SKILL.md").is_file())


def install_starter_skills(src: Path = STARTER_SKILLS, dest: Path = AGENT_SKILLS, dry_run: bool = False,
                           today: str | None = None) -> tuple[list[str], list[str]]:
    """Copy each starter skill the agent does not have into `dest`, with a fresh history.jsonl holding one "seeded"
    line (D5: the source's own history, if any, stays behind). Returns (copied, kept): a skill of that name already
    there - a folder, a file or a link - is kept as it is (D2). The copy is staged beside it and renamed into place,
    so an interrupted run leaves no half skill that a later run would mistake for the agent's own."""
    today = today or datetime.date.today().isoformat()
    copied, kept = [], []
    todo = []
    for d in starter_skills(src):
        target = dest / d.name
        if target.exists() or target.is_symlink():
            kept.append(d.name)
            continue
        if links := [p for p in d.rglob("*") if p.is_symlink()]:
            raise ValueError(f"starter skill {d.name}: symbolic links are not copied ({', '.join(map(str, links))})")
        todo.append((d, target))
    for d, target in todo:
        copied.append(d.name)
        if dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        tmp = dest / f".{d.name}.starter-tmp"
        if tmp.is_dir() and not tmp.is_symlink():          # left by an interrupted run
            shutil.rmtree(tmp)
        elif tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        shutil.copytree(d, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "history.jsonl"))
        line = {"date": today, "episode": None, "program": None, "event": "changed", "outcome": None,
                "notes": SEEDED_NOTE.format(name=d.name)}
        (tmp / "history.jsonl").write_text(json.dumps(line) + "\n")
        tmp.rename(target)
    return copied, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report whether the files are up to date")
    ap.add_argument("--no-starter-skills", action="store_true",
                    help="do not copy the starter skills (skills/) into the agent's skills folder")
    a = ap.parse_args()
    home = Path.home().resolve()
    skipped = []
    try:
        claude = json.dumps(render_claude(REPO, home, skipped), indent=2, ensure_ascii=False) + "\n"
        pi = render_pi(REPO)
    except TemplateError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for e in skipped:
        print(f"WARNING: {e} has a pattern character in its name, so it cannot be denied by name; the agent may "
              "read it. Rename or move it.", file=sys.stderr)
    stale = []
    for out, text in ((CLAUDE_OUT, claude), (PI_OUT, pi)):
        old = out.read_text() if out.exists() else None
        if old == text:
            print(f"up to date: {out.relative_to(REPO)}")
            continue
        stale.append(out)
        if out == CLAUDE_OUT and old:
            try:
                before = set(json.loads(old)["permissions"]["deny"])
                after = set(json.loads(text)["permissions"]["deny"])
                for r in sorted(after - before):
                    print(f"  + {r}")
                for r in sorted(before - after):
                    print(f"  - {r}")
            except (ValueError, KeyError):
                pass
        if not a.check:
            out.write_text(text)
            print(f"wrote {out.relative_to(REPO)}")
    if not a.no_starter_skills:
        try:
            copied, kept = install_starter_skills(dry_run=a.check)
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        for name in copied:
            print(f"starter skill {name}: " + ("not copied yet (--check)" if a.check else
                                                f"copied into {AGENT_SKILLS.relative_to(REPO)}/{name} as a draft"))
        for name in kept:
            print(f"starter skill {name}: the agent has its own; left as it is")
    if a.check and stale:
        print("out of date: run python tools/setup_agent_ws.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
