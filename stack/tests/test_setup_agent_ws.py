"""tools/setup_agent_ws.py: the operating agent's settings rendered for wherever the repository is."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import setup_agent_ws  # noqa: E402


def test_rendered_for_another_machine_it_denies_home_and_neighbours_but_not_the_way_to_the_repo(tmp_path,
                                                                                                monkeypatch):
    home = tmp_path / "home" / "ada"
    repo = home / "src" / "robot-use"
    for d in (repo, home / ".ssh", home / "Documents", home / "src" / "other-project"):
        d.mkdir(parents=True)
    (home / ".bash_history").write_text("")
    monkeypatch.setenv("ROBOT_USE_HOME", str(tmp_path / "data"))
    s = setup_agent_ws.render_claude(repo, home)
    text = json.dumps(s)
    assert "{REPO}" not in text and "{HOME}" not in text and "_comment" not in s
    deny = s["permissions"]["deny"]
    for rule in (f"Read(/{home}/.ssh/**)", f"Read(/{home}/Documents/**)", f"Read(/{home}/.bash_history)",
                 f"Read(/{home}/src/other-project/**)", f"Read(/{repo}/robot.yaml)", f"Read(/{repo}/calib/**)",
                 f"Read(/{tmp_path}/data/agent_logs/**)", f"Read(/{tmp_path}/data/episodes/**)",
                 f"Read(/{tmp_path}/data/sim/**)", f"Edit(/{tmp_path}/data/episodes/**)", f"Edit(/{repo}/stack/**)",
                 "Bash"):
        assert rule in deny, rule
    for rule in (f"Read(/{home}/src/**)", f"Read(/{repo}/**)"):                    # the way to the agent's cwd
        assert rule not in deny, rule
    assert s["permissions"]["additionalDirectories"] == [f"{repo}/lib/robot_use", f"{repo}/stack/rig"]
    assert s["claudeMdExcludes"] == [f"{repo}/AGENTS.md", f"{repo}/CLAUDE.md"]
    pi = json.loads(setup_agent_ws.render_pi(repo))["permission"]
    assert pi["read"][f"{repo}/agent_ws/*"] == "allow" and "{REPO}" not in json.dumps(pi)


def _template_copy(tmp_path, monkeypatch, edit):
    t = json.loads(setup_agent_ws.CLAUDE_TEMPLATE.read_text())
    edit(t)
    f = tmp_path / "settings.template.json"
    f.write_text(json.dumps(t))
    monkeypatch.setattr(setup_agent_ws, "CLAUDE_TEMPLATE", f)


@pytest.mark.parametrize("edit", [
    lambda t: t["permissions"]["deny"].remove("Read(/{REPO}/stack/runner/**)"),        # not in REQUIRED_DENY
    lambda t: t.update(hooks={"PreToolUse": [{"hooks": [{"type": "command", "command": "sh"}]}]}),
    lambda t: t["permissions"].update(defaultMode="bypassPermissions"),
    lambda t: t.update(autoMemoryEnabled=True),
])
def test_any_change_to_the_template_is_refused_until_reviewed(tmp_path, monkeypatch, edit):
    """Re-review of task 7: a template changed in any way (by hand, a commit, an agent without rules) is refused;
    only a reviewed change, with its hash updated in the script, renders."""
    _template_copy(tmp_path, monkeypatch, edit)
    with pytest.raises(setup_agent_ws.TemplateError, match="not the reviewed template"):
        setup_agent_ws.render_claude(REPO, Path.home())


@pytest.mark.parametrize("edit, match", [
    (lambda t: t["permissions"]["deny"].remove("Bash"), "required denies"),
    (lambda t: t["permissions"]["deny"].remove("Read(/{REPO}/robot.yaml)"), "required denies"),
    (lambda t: t["permissions"]["deny"].remove("Edit(/{REPO}/agent_ws/.agents/settings.template.json)"),
     "required denies"),
    (lambda t: t["permissions"]["allow"].append("Bash(ls:*)"), "allows more"),
    (lambda t: t["permissions"]["additionalDirectories"].append("{REPO}"), "directories"),
])
def test_even_a_reviewed_template_must_keep_the_required_rules(tmp_path, monkeypatch, edit, match):
    """The second guard, behind the hash: the must-have rules and the allow list."""
    _template_copy(tmp_path, monkeypatch, edit)
    monkeypatch.setattr(setup_agent_ws, "_template_text", lambda p: p.read_text())
    with pytest.raises(setup_agent_ws.TemplateError, match=match):
        setup_agent_ws.render_claude(REPO, Path.home())


def test_the_pi_template_must_deny_by_default(tmp_path, monkeypatch):
    t = json.loads(setup_agent_ws.PI_TEMPLATE.read_text())
    t["permission"]["read"]["*"] = "allow"
    f = tmp_path / "config.template.json"
    f.write_text(json.dumps(t))
    monkeypatch.setattr(setup_agent_ws, "PI_TEMPLATE", f)
    with pytest.raises(setup_agent_ws.TemplateError, match="not the reviewed template"):
        setup_agent_ws.render_pi(REPO)
    monkeypatch.setattr(setup_agent_ws, "_template_text", lambda p: p.read_text())
    with pytest.raises(setup_agent_ws.TemplateError, match="deny by default"):
        setup_agent_ws.render_pi(REPO)


def test_the_committed_templates_are_the_reviewed_ones():
    setup_agent_ws.render_claude(REPO, Path.home().resolve())          # no TemplateError
    setup_agent_ws.render_pi(REPO)


def test_a_link_to_the_repository_is_not_denied_and_the_home_root_is_refused(tmp_path, monkeypatch):
    home = tmp_path / "h"
    repo = home / "code" / "robot-use"
    repo.mkdir(parents=True)
    (home / "ru").symlink_to(repo)                                     # another way to the repo
    (home / "c").symlink_to(home / "code")                               # a link to a folder above it
    (repo / "tools").mkdir()
    (home / "rigtools").symlink_to(repo / "tools")                       # a link into it
    (home / "h").symlink_to(home)                                        # a link to home itself
    monkeypatch.setenv("ROBOT_USE_HOME", str(tmp_path / "data"))
    deny = setup_agent_ws.render_claude(repo, home)["permissions"]["deny"]
    for rule in (f"Read(/{repo}/**)", f"Read(/{home}/code/**)", f"Read(/{home}/**)"):      # no lock-out
        assert rule not in deny, rule
    for link in ("ru", "c", "rigtools", "h"):                          # but each link's own path is denied
        assert f"Read(/{home}/{link}/**)" in deny and f"Read(/{home}/{link})" in deny, link
    with pytest.raises(setup_agent_ws.TemplateError, match="folder of its own"):
        setup_agent_ws.render_claude(home, home)


def test_deeper_layouts_odd_names_and_symlinks(tmp_path, monkeypatch):
    home = tmp_path / "h"
    repo = home / "a" / "b" / "robot-use"
    for d in (repo, home / "a" / "notes", home / "a" / "b" / "other", home / "real"):
        d.mkdir(parents=True)
    (home / "weird[1]").mkdir()
    (home / "link").symlink_to(home / "real")
    monkeypatch.setenv("ROBOT_USE_HOME", str(tmp_path / "data"))
    skipped = []
    deny = setup_agent_ws.render_claude(repo, home, skipped)["permissions"]["deny"]
    for rule in (f"Read(/{home}/a/notes/**)", f"Read(/{home}/a/b/other/**)", f"Read(/{home}/link/**)",
                 f"Read(/{home}/link)", f"Read(/{home}/real/**)"):
        assert rule in deny, rule
    for rule in (f"Read(/{home}/a/**)", f"Read(/{home}/a/b/**)"):
        assert rule not in deny, rule
    assert skipped == [home / "weird[1]"] and not any("weird" in r for r in deny)


def test_a_preview_takes_the_backend_of_the_robotd_it_is_for():
    """Review of task 7: a preview for real arms must get the real run's box, never the sim's."""
    from stack.twin_mujoco.preview import robotd_backend
    assert robotd_backend("http://127.0.0.1:9") == "hardware"            # unreachable: assume the strict case


def test_a_repository_under_tmp_keeps_its_own_files_readable(tmp_path, monkeypatch):
    """The template denies /tmp; a clone there must not lock the agent out of its brief (found in a clean-copy run)."""
    repo = Path("/tmp") / tmp_path.name / "robot-use" if not str(tmp_path).startswith("/tmp/") else tmp_path / "robot-use"
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("ROBOT_USE_HOME", str(tmp_path / "data"))
    deny = setup_agent_ws.render_claude(repo, home)["permissions"]["deny"]
    assert "Read(//tmp/**)" not in deny
    assert f"Read(/{repo}/robot.yaml)" in deny and f"Read(/{repo}/stack/robotd/**)" in deny     # still protected


def _rendered(tmp_path, monkeypatch):
    """This machine's settings, rendered into tmp_path (never the checkout's own files)."""
    claude, pi = tmp_path / "settings.json", tmp_path / "config.json"
    monkeypatch.setattr(setup_agent_ws, "CLAUDE_OUT", claude)
    monkeypatch.setattr(setup_agent_ws, "PI_OUT", pi)
    monkeypatch.setattr(setup_agent_ws, "REPO", tmp_path)
    monkeypatch.setattr(setup_agent_ws, "render_claude", lambda repo, home, skipped=None: {
        "permissions": {"deny": ["Read(/x/robot.yaml)", "Read(/x/calib/**)"]}})
    monkeypatch.setattr(setup_agent_ws, "render_pi", lambda repo: '{"pi": 1}\n')
    claude.write_text(json.dumps(setup_agent_ws.render_claude(None, None), indent=2) + "\n")
    pi.write_text('{"pi": 1}\n')
    return claude


def test_the_launchers_refuse_stale_agent_permissions_and_never_write_them(tmp_path, monkeypatch, capsys):
    """Review of 8c/8d (2026-10-14): a launcher must not start the stack while the agent's rendered permissions differ
    from the reviewed templates (after a rename, a stale file left robot.yaml readable), and must not rewrite them
    itself (a re-render under another HOME, e.g. sudo, would drop the real home's denies)."""
    import launch_common
    claude = _rendered(tmp_path, monkeypatch)
    assert setup_agent_ws.drift() is None and launch_common.agent_settings_current()
    claude.write_text(json.dumps({"permissions": {"deny": ["Read(/x/rig.yaml)", "Read(/x/calib/**)"]}}))
    before = claude.read_text()
    assert setup_agent_ws.drift()["added"] == ["Read(/x/robot.yaml)"]
    assert not launch_common.agent_settings_current()                       # the real arms and the sim: refuse
    err = capsys.readouterr().err
    assert "cannot start" in err and "+ Read(/x/robot.yaml)" in err and "- Read(/x/rig.yaml)" in err
    assert launch_common.agent_settings_current(refuse=False)               # a mock run: a warning only
    assert "WARNING" in capsys.readouterr().err
    assert claude.read_text() == before                                     # nothing written, either way


def test_the_launchers_refuse_when_the_permissions_cannot_be_checked(tmp_path, monkeypatch, capsys):
    import launch_common
    _rendered(tmp_path, monkeypatch)

    def refused(repo, home, skipped=None):
        raise setup_agent_ws.TemplateError("settings.template.json is not the reviewed template")
    monkeypatch.setattr(setup_agent_ws, "render_claude", refused)
    assert not launch_common.agent_settings_current()
    assert "cannot be checked" in capsys.readouterr().err
    (tmp_path / "settings.json").unlink()                                   # not written yet: warn, start
    assert launch_common.agent_settings_current()
    assert "not written yet" in capsys.readouterr().err


def _starter(src: Path, name: str, extra: dict | None = None):
    d = src / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: \"[draft] x\"\nstatus: draft\n---\n")
    (d / "skill.py").write_text("print('hi')\n")
    for k, v in (extra or {}).items():
        (d / k).write_text(v)


def test_starter_skills_are_copied_when_absent_and_never_over_the_agents_own(tmp_path):
    src, dest = tmp_path / "skills", tmp_path / "agent" / "skills"
    _starter(src, "pick", {"history.jsonl": '{"notes": "the reference rig\'s own trail"}\n'})
    _starter(src, "push")
    (src / "README.md").write_text("not a skill")
    (src / "half").mkdir()                                       # no SKILL.md: not a skill
    own = dest / "push"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("the agent's own")
    copied, kept = setup_agent_ws.install_starter_skills(src, dest, today="2026-10-15")
    assert copied == ["pick"] and kept == ["push"]
    assert (own / "SKILL.md").read_text() == "the agent's own" and not (own / "skill.py").exists()
    assert (dest / "pick" / "skill.py").read_text() == "print('hi')\n"
    lines = (dest / "pick" / "history.jsonl").read_text().splitlines()
    assert len(lines) == 1                                       # a fresh history: the source's trail stays behind
    h = json.loads(lines[0])
    assert h["date"] == "2026-10-15" and h["event"] == "changed" and "reference rig" in h["notes"]
    assert "not on this robot" in h["notes"] and "skills/pick" in h["notes"]
    assert sorted(p.name for p in dest.iterdir()) == ["pick", "push"]      # no staging folder left behind
    (dest / "pick" / "skill.py").write_text("the agent's revision\n")      # once copied, it is the agent's
    assert setup_agent_ws.install_starter_skills(src, dest) == ([], ["pick", "push"])
    assert (dest / "pick" / "skill.py").read_text() == "the agent's revision\n"


def test_a_starter_skill_is_kept_out_by_a_file_or_link_of_its_name_and_a_dry_run_writes_nothing(tmp_path):
    src, dest = tmp_path / "skills", tmp_path / "agent" / "skills"
    _starter(src, "pick")
    _starter(src, "push")
    dest.mkdir(parents=True)
    (dest / "pick").symlink_to(tmp_path / "nowhere")            # a dangling link still holds the name
    assert setup_agent_ws.install_starter_skills(src, dest, dry_run=True) == (["push"], ["pick"])
    assert not (dest / "push").exists()
    (dest / ".push.starter-tmp").mkdir()                        # left by an interrupted run: replaced, not kept
    (dest / ".push.starter-tmp" / "junk").write_text("x")
    assert setup_agent_ws.install_starter_skills(src, dest) == (["push"], ["pick"])
    assert not (dest / "push" / "junk").exists() and not (dest / ".push.starter-tmp").exists()


def test_the_shipped_starter_skills_arrive_as_drafts():
    skills = setup_agent_ws.starter_skills()
    assert [d.name for d in skills] == ["pick-and-place"]
    for d in skills:
        text = (d / "SKILL.md").read_text()
        front = text.split("---")[1]
        assert f"name: {d.name}" in front and "status: draft" in front and 'description: "[draft]' in front
        assert "**Source:**" in text and not (d / "history.jsonl").exists()
        compile((d / "skill.py").read_text(), str(d / "skill.py"), "exec")


def test_setup_copies_the_starter_skills_unless_told_not_to(tmp_path, monkeypatch, capsys):
    src, dest = tmp_path / "skills", tmp_path / "agent" / "skills"
    _starter(src, "pick")
    monkeypatch.setattr(setup_agent_ws, "STARTER_SKILLS", src)
    monkeypatch.setattr(setup_agent_ws, "AGENT_SKILLS", dest)
    monkeypatch.setattr(setup_agent_ws, "REPO", tmp_path)
    for name in ("CLAUDE_OUT", "PI_OUT"):
        monkeypatch.setattr(setup_agent_ws, name, tmp_path / f"{name}.json")
    monkeypatch.setattr(setup_agent_ws, "render_claude", lambda repo, home, skipped=None: {"permissions": {"deny": []}})
    monkeypatch.setattr(setup_agent_ws, "render_pi", lambda repo: "{}")
    install = setup_agent_ws.install_starter_skills
    monkeypatch.setattr(setup_agent_ws, "install_starter_skills",
                        lambda dry_run=False: install(src, dest, dry_run=dry_run))
    for argv, copied in ((["--no-starter-skills"], False), (["--check"], False), ([], True)):
        monkeypatch.setattr(sys, "argv", ["setup_agent_ws.py", *argv])
        assert setup_agent_ws.main() == 0, argv        # --check: the settings are current; a missing skill is no error
        assert (dest / "pick").exists() == copied, argv
    out = capsys.readouterr().out
    assert "starter skill pick: not copied yet (--check)" in out and "starter skill pick: copied" in out
    _starter(src, "linked")
    (src / "linked" / "notes.md").symlink_to(tmp_path / "secret")
    monkeypatch.setattr(sys, "argv", ["setup_agent_ws.py"])
    assert setup_agent_ws.main() == 2 and not (dest / "linked").exists()     # a link in a starter skill is refused
    assert "symbolic links" in capsys.readouterr().err
    (dest / "linked").mkdir()                       # the agent has its own: the starter's link is never looked at
    assert setup_agent_ws.main() == 0


def test_the_agent_is_denied_the_starter_skills_folder(tmp_path):
    deny = setup_agent_ws.render_claude(REPO, Path.home().resolve())["permissions"]["deny"]
    assert f"Read(/{REPO}/skills/**)" in deny and f"Edit(/{REPO}/skills/**)" in deny
