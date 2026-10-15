"""robotd configuration: ``stack/robotd.yaml`` (wiring, behaviour) + ``robot.yaml`` (geometry), as dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import paths
from ..rig.kinematics import ArmMount
from ..rig.rig import Rig, load_rig
from .motion import ContactConfig, Supervisor
from .safety import SafetyConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "robotd.yaml"


@dataclass
class ArmConfig:
    name: str
    port: str
    calibration_id: str
    mount: ArmMount


@dataclass
class CameraConfig:
    name: str
    path: str
    width: int
    height: int
    fps: int = 30


@dataclass
class Endpoint:
    host: str
    port: int


@dataclass
class OverlayConfig:
    camera: str                            # its model: rig.camera_model(camera), from files or estimates


@dataclass
class RobotdConfig:
    arms: dict[str, ArmConfig]
    cameras: dict[str, CameraConfig]
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    overlay: OverlayConfig | None = None
    motion_enabled: bool = False
    loop_hz: float = 50.0
    api: Endpoint = field(default_factory=lambda: Endpoint("127.0.0.1", 8770))
    operator: Endpoint = field(default_factory=lambda: Endpoint("0.0.0.0", 8772))
    log_dir: Path = field(default_factory=paths.agent_logs)
    approval_mode: str = "first_motion"
    armed_timeout_s: float = 1800.0
    idle_rest_s: float = 600.0                 # holding with no command this long -> slowly to rest
    mcp_localhost_only: bool = False           # required for approval_mode "none" (no auth yet)
    backend: str = "hardware"                  # what the arms are: "hardware", "mock" or "sim" (set at start-up,
                                               # never from the file; the runner's sandbox choice reads it)
    contact: ContactConfig = field(default_factory=ContactConfig)
    settle_tol_deg: float = 3.0
    settle_timeout_s: float = 2.0
    rig: Rig | None = None


# Geometry lives in robot.yaml only; these SafetyConfig fields are filled from it and refused here.
GEOMETRY_KEYS = {"board_top_z", "board_front_x", "board_half_width_y", "obstacles", "work_surface_z"}


def _hull_margins(raw) -> dict:
    """Partial overrides merged into the defaults; every margin a positive number of metres (review 2026-10-05)."""
    out = {k: dict(v) for k, v in SafetyConfig().hull_margins_m.items()}
    if not isinstance(raw, dict):
        raise ValueError("safety.hull_margins_m: a mapping {target: {part: metres}}")
    for target, parts in raw.items():
        if target not in out or not isinstance(parts, dict):
            raise ValueError(f"safety.hull_margins_m: targets are {sorted(out)}, each a mapping of parts")
        for part, v in parts.items():
            if part not in out[target]:
                raise ValueError(f"safety.hull_margins_m.{target}: parts are {sorted(out[target])}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 0.2:
                raise ValueError(f"safety.hull_margins_m.{target}.{part}: {v!r} - a margin in metres, 0 < m < 0.2")
            out[target][part] = float(v)
    return out


def load_config(path: Path | str = DEFAULT_CONFIG) -> RobotdConfig:
    raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    rig = load_rig(raw.get("rig"))
    arms = {}
    for name, a in (raw.get("arms") or {}).items():
        if name not in ("left", "right"):
            raise ValueError(f"arms: only 'left' and 'right' are supported, got {name!r}")
        extra = set(a) - {"port"}
        if extra:
            raise ValueError(f"arms.{name}: {sorted(extra)} belong in robot.yaml (geometry and calibration live there)")
        arms[name] = ArmConfig(name, a["port"], rig.arm(name)["calibration_id"], rig.arm_mount(name))
    cameras = {}
    for n in raw.get("cameras") or []:
        c = rig.camera(n)
        m = c["mode"]
        cameras[n] = CameraConfig(n, c["device"], int(m["width"]), int(m["height"]), int(m.get("fps", 30)))
    fields = {f.name for f in dataclasses.fields(SafetyConfig)}
    s_raw = raw.get("safety") or {}
    unknown = set(s_raw) - fields
    if unknown:
        raise ValueError(f"safety: unknown keys {sorted(unknown)}")
    geo = set(s_raw) & GEOMETRY_KEYS
    if geo:
        raise ValueError(f"safety: {sorted(geo)} are geometry; they come from robot.yaml")
    if "rest_roll_deg" in s_raw and s_raw["rest_roll_deg"] not in (0, -90):
        raise ValueError("safety.rest_roll_deg: 0 or -90 (the two rest rolls checked in the model; sprint 2026-10-05)")
    if "hull_margins_m" in s_raw:
        s_raw = {**s_raw, "hull_margins_m": _hull_margins(s_raw["hull_margins_m"])}
    board = rig.board
    safety = SafetyConfig(**s_raw, work_surface_z=0.0, board_top_z=board.thickness_m, board_front_x=board.front_x_m,
                          board_half_width_y=board.half_width_y_m, obstacles=tuple(rig.obstacles()))
    overlay = None
    ov_cam = raw.get("overlay_camera")
    if ov_cam:
        overlay = OverlayConfig(ov_cam)
    cfg = RobotdConfig(arms=arms, cameras=cameras, safety=safety, overlay=overlay, rig=rig,
                       motion_enabled=bool(raw.get("motion_enabled", False)),
                       loop_hz=float(raw.get("loop_hz", 50)),
                       log_dir=Path(raw["log_dir"]).expanduser() if raw.get("log_dir") else paths.agent_logs())
    if "api" in raw:
        cfg.api = Endpoint(raw["api"]["host"], int(raw["api"]["port"]))
    if "operator" in raw:
        cfg.operator = Endpoint(raw["operator"]["host"], int(raw["operator"]["port"]))
    if cfg.api.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("api.host must be localhost: the robotd API has no authentication (design doc)")
    cfg.approval_mode = str(raw.get("approval_mode", cfg.approval_mode))
    if cfg.approval_mode not in Supervisor.MODES:
        raise ValueError(f"approval_mode must be one of {Supervisor.MODES}")
    cfg.armed_timeout_s = float(raw.get("armed_timeout_s", cfg.armed_timeout_s))
    cfg.idle_rest_s = float(raw.get("idle_rest_s", cfg.idle_rest_s))
    cfg.mcp_localhost_only = bool(raw.get("mcp_localhost_only", False))
    c_raw = raw.get("contact") or {}
    unknown = set(c_raw) - {f.name for f in dataclasses.fields(ContactConfig)}
    if unknown:
        raise ValueError(f"contact: unknown keys {sorted(unknown)}")
    cfg.contact = ContactConfig(**c_raw)
    if cfg.approval_mode == "none" and not cfg.mcp_localhost_only:
        raise ValueError("approval_mode 'none' (no human in the loop) requires mcp_localhost_only: true "
                         "until authentication exists (design doc, Level 3 warning)")
    return cfg
