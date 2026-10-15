"""Run the MCP server (streamable HTTP) in front of a running robotd.

    .venv/bin/python -m stack.mcp_server                  # LAN, port 8771, path /mcp
    .venv/bin/python -m stack.mcp_server --host 127.0.0.1 # this machine only

Clients connect to http://<robot-host>:8771/mcp . No authentication yet (design doc): anyone on the LAN
who can reach the port can use every tool, including motion (still approved by a human on the operator
page unless robotd's approval_mode is "none", which robotd only allows with this server on localhost).
"""

import argparse
import logging

from mcp.server.transport_security import TransportSecuritySettings

from .server import RobotdClient, build_server, local_hosts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--robotd", default="http://127.0.0.1:8770", help="robotd API base URL")
    ap.add_argument("--surface", default="agent", choices=["agent", "debug"],
                    help="agent: programs only (default); debug: the direct motion tools, for people")
    ap.add_argument("--tools", default="all", choices=["all", "joint", "cartesian"],
                    help="debug surface: which way of commanding the arms to offer (design doc phase 6b)")
    ap.add_argument("--episodes", default=None,
                    help="where episodes and programs are recorded (default $ROBOT_USE_HOME/episodes, ~/.tensortron/robot-use/episodes)")
    ap.add_argument("--no-preview", action="store_true", help="the twin preview off for this session: every program runs directly on the live robot (the agent is told; robotd still checks every step)")
    ap.add_argument("--allow-host", action="append", default=[],
                    help="extra Host header value clients use to reach this server (e.g. a DNS name)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import json, urllib.request
        with urllib.request.urlopen(a.robotd.rstrip("/") + "/config", timeout=3) as r:
            rcfg = json.loads(r.read())
        if rcfg.get("approval_mode") == "none" and a.host not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit("[FATAL] robotd runs with approval_mode 'none' (no human in the loop): this server "
                             "must then bind to 127.0.0.1 until authentication exists")
    except (OSError, ValueError):
        logging.warning("robotd not reachable at %s yet; tools will report that until it is", a.robotd)
    hosts = local_hosts() + a.allow_host
    lan_ip = next((h for h in hosts if h[0].isdigit() and not h.startswith("127.")), "127.0.0.1")
    public = f"http://{lan_ip if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{h}:*" for h in hosts] + hosts,
        allowed_origins=[f"http://{h}:*" for h in hosts])
    mcp = build_server(RobotdClient(a.robotd), public_url=public, tool_set=a.tools, surface=a.surface,
                       episodes_dir=a.episodes, preview=not a.no_preview)
    logging.info("MCP server on http://%s:%d/mcp (public %s/mcp), robotd %s, allowed hosts %s%s",
                 a.host, a.port, public, a.robotd, hosts, "; TWIN PREVIEW OFF" if a.no_preview else "")
    mcp.run("streamable-http", host=a.host, port=a.port, transport_security=security)


if __name__ == "__main__":
    main()
