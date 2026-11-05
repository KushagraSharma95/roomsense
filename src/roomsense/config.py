"""Room configuration: loaded from YAML into plain dataclasses, with clear errors for bad input."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATES = ("vacant", "occupied", "conversation")


class ConfigError(ValueError):
    """Raised when a room config is missing something or has a bad value."""


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]


@dataclass
class CameraConfig:
    image_points: list[tuple[float, float]] | None = None
    floor_points: list[tuple[float, float]] | None = None


@dataclass
class DetectorConfig:
    kind: str = "yolox"  # "yolox" (ONNX person detector) or "motion" (background subtraction, no model)
    model_path: str = "models/yolox_nano.onnx"
    score_threshold: float = 0.35
    nms_threshold: float = 0.45
    threads: int = 2
    frame_stride: int = 1


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.25
    min_hits: int = 3
    max_missed: int = 15


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    window_s: float = 10.0


@dataclass
class FusionConfig:
    occupied_after_s: float = 2.0
    vacant_after_s: float = 300.0
    degraded_vacant_after_s: float = 900.0
    conversation_min_people: int = 2
    conversation_speech_ratio: float = 0.2
    conversation_after_s: float = 10.0
    conversation_end_s: float = 60.0
    speech_presence_ratio: float = 0.1
    count_window_s: float = 3.0


@dataclass
class PolicyConfig:
    min_hold_s: float = 30.0
    actions: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class MqttConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    topic_prefix: str = "roomsense"


@dataclass
class QrcConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1710
    user: str = ""
    password: str = ""
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ApiConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class OutputsConfig:
    console: bool = True
    jsonl_path: str | None = None
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    qrc: QrcConfig = field(default_factory=QrcConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


@dataclass
class RoomConfig:
    room_id: str
    zones: list[Zone]
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def _section(cls, raw: Any, where: str):
    """Build a dataclass from a dict, rejecting unknown keys so typos don't pass silently."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where}' must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in '{where}': {', '.join(sorted(unknown))}")
    return cls(**raw)


def _points(raw: Any, where: str, min_len: int) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < min_len:
        raise ConfigError(f"'{where}' needs at least {min_len} [x, y] points")
    out = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ConfigError(f"'{where}' has a point that is not [x, y]: {p!r}")
        out.append((float(p[0]), float(p[1])))
    return out


def _as_value(v: Any) -> str:
    # YAML 1.1 reads bare on/off/yes/no as booleans; treat them as the words people meant.
    if v is True:
        return "on"
    if v is False:
        return "off"
    return str(v)


def parse_config(raw: dict[str, Any]) -> RoomConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    room_id = raw.get("room_id")
    if not room_id:
        raise ConfigError("'room_id' is required")

    zones_raw = raw.get("zones") or []
    if not zones_raw:
        raise ConfigError("define at least one zone (a zone named 'room' marks the room boundary)")
    zones = []
    for i, z in enumerate(zones_raw):
        if not isinstance(z, dict) or "name" not in z:
            raise ConfigError(f"zone #{i + 1} needs a 'name'")
        zones.append(Zone(str(z["name"]), _points(z.get("polygon"), f"zones.{z['name']}.polygon", 3)))
    names = [z.name for z in zones]
    if len(set(names)) != len(names):
        raise ConfigError("zone names must be unique")

    cam_raw = raw.get("camera") or {}
    camera = CameraConfig()
    if cam_raw:
        img = _points(cam_raw.get("image_points"), "camera.image_points", 4)
        flr = _points(cam_raw.get("floor_points"), "camera.floor_points", 4)
        if len(img) != len(flr):
            raise ConfigError("camera.image_points and camera.floor_points must have the same length")
        camera = CameraConfig(img, flr)

    detector = _section(DetectorConfig, raw.get("detector"), "detector")
    if detector.kind not in ("yolox", "motion"):
        raise ConfigError("detector.kind must be 'yolox' or 'motion'")
    if detector.frame_stride < 1:
        raise ConfigError("detector.frame_stride must be at least 1")

    policy_raw = raw.get("policy") or {}
    actions_raw = policy_raw.get("actions") or {}
    unknown_states = set(actions_raw) - set(STATES)
    if unknown_states:
        raise ConfigError(f"policy.actions has unknown state(s): {', '.join(sorted(unknown_states))}")
    actions = {
        state: {str(target): _as_value(value) for target, value in (targets or {}).items()}
        for state, targets in actions_raw.items()
    }
    policy = PolicyConfig(float(policy_raw.get("min_hold_s", 30.0)), actions)

    out_raw = raw.get("outputs") or {}
    outputs = OutputsConfig(
        console=bool(out_raw.get("console", True)),
        jsonl_path=out_raw.get("jsonl_path"),
        mqtt=_section(MqttConfig, out_raw.get("mqtt"), "outputs.mqtt"),
        qrc=_section(QrcConfig, out_raw.get("qrc"), "outputs.qrc"),
        api=_section(ApiConfig, out_raw.get("api"), "outputs.api"),
    )

    return RoomConfig(
        room_id=str(room_id),
        zones=zones,
        camera=camera,
        detector=detector,
        tracker=_section(TrackerConfig, raw.get("tracker"), "tracker"),
        audio=_section(AudioConfig, raw.get("audio"), "audio"),
        fusion=_section(FusionConfig, raw.get("fusion"), "fusion"),
        policy=policy,
        outputs=outputs,
    )


def load_config(path: str | Path) -> RoomConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"could not parse {path}: {e}") from e
    return parse_config(raw)
