# Isolating the operating agent (planned, not done)

The agent that operates the robot runs as Claude Code in `agent_ws/`. Today it is confined by Claude Code's
own settings (`agent_ws/.agents/settings.json`, read through the `.claude` symlink): Read and Edit **deny lists**, `claudeMdExcludes`, auto memory
off, and no Bash. That is enough for the experiments so far, but it is best-effort by design:

- deny lists name what is forbidden, so anything new and unlisted (a file in the home directory, a new repo,
  a new temp directory) is readable until someone adds a rule; deny always wins, so "deny everything except"
  cannot be written;
- Claude Code applies Read rules to Grep and Glob as a best effort;
- the agent runs as the developer's Unix user, so the kernel lets it see everything the developer can.

What happened without it (2026-10-03, first agent runs): the agent grepped `rig.yaml` (now `robot.yaml`) for the grasp offset that
Experiment 0 withholds, read the developer's Claude memory, and had the developer's `AGENTS.md` and auto
memory loaded (auto memory is keyed by git root). A later review found the withheld facts also in `/tmp`
(pytest and twin files, the developer's scratchpads), old episode programs and Claude's file history.

## The robust boundary: a separate Unix user

Run the agent's Claude Code as its own user, `robot-agent`, which the kernel only lets see what it needs:

| it needs | access |
|---|---|
| its workspace (brief, tasks, settings, its skills) | read + write to `agent_ws/` (or a copy owned by `robot-agent`) |
| the robot description and the API | read: `stack/rig/embodiment.md`, `lib/robot_use/README.md` |
| the robot | the MCP server on `127.0.0.1:8771` (the MCP server, runner and twin keep running as the developer) |
| Claude itself | its own `~/.claude` (its own login) and the Anthropic API |

Everything else - the developer's home, the rest of the repo, `/tmp` of other users, other repos, episodes and
logs - is simply not readable, without having to be listed.

### Steps (need sudo; run by the developer)

```bash
# 1. the user and a group for the shared files
sudo adduser --disabled-password --gecos "robot-use operating agent" robot-agent
sudo groupadd -f robot-agent-read

# 2. the developer's home stops being world-readable (check nothing else relies on it first)
chmod 750 /home/<dev>

# 3. a workspace the agent owns, outside the developer's home
sudo mkdir -p /srv/robot-agent
sudo cp -r $REPO/agent_ws /srv/robot-agent/agent_ws
sudo cp $REPO/stack/rig/embodiment.md $REPO/lib/robot_use/README.md \
        /srv/robot-agent/agent_ws/
sudo chown -R robot-agent:robot-agent /srv/robot-agent
#    then in /srv/robot-agent/agent_ws/AGENTS.md: import ./embodiment.md and ./README.md instead of ../stack/...,
#    and in its .mcp.json keep http://127.0.0.1:8771/mcp

# 4. its own Claude Code login
sudo -iu robot-agent claude          # log in once; approve the project's MCP server
```

Keeping the copies current: the two documents and the brief are copied, not linked (the agent cannot follow a
link into a directory it cannot read). A small `tools/agent_ws_sync.py` (to write) would refresh
`/srv/robot-agent/agent_ws` from the repo before each experiment and copy the agent's skills and notes back for
review - the repo stays the source of truth.

### Optional: network

The agent only needs the MCP port and the Anthropic API. robotd's API (`127.0.0.1:8770`) is reachable by any
local user; the agent has no Bash or HTTP tool today, but to make it a kernel rule:

```bash
sudo iptables -A OUTPUT -o lo -p tcp --dport 8770 -m owner --uid-owner robot-agent -j REJECT
sudo iptables -A OUTPUT -o lo -p tcp --dport 8772 -m owner --uid-owner robot-agent -j REJECT
```

### Check it

As `robot-agent`, from its workspace, in a headless run (`claude -p ...`): reading `$REPO/robot.yaml`,
`/home/<dev>/.claude/`, `/tmp/pytest-of-<dev>/` and `~<dev>/.tensortron/robot-use/` must fail with "Permission
denied" from the kernel (not a Claude Code rule); reading its own `embodiment.md` and calling the MCP tools must
work. Keep the Claude Code deny lists as a second layer.
