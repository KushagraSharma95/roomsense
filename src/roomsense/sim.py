"""A synthetic 4-hour day in a meeting room, used to test the logic end to end and to compare three
ways of deciding whether a room is occupied:

  1. a motion sensor (PIR-style): occupied while motion was seen in the last `vacant_after_s`
  2. camera only: the tracker and fusion logic with the microphone ignored
  3. camera + microphone: the full pipeline

Everything is generated: positions, detector misses and false alarms, speech, motion. The day
deliberately contains the two situations that break single-sensor systems (someone reading very
still, and a presentation with the lights down so the camera can't see). The numbers show the logic
behaves as designed under these assumptions. They are not field measurements.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters.sinks import JsonlSink, Sink
from .audio import SpeechActivity
from .config import RoomConfig
from .fusion import RoomState, RoomStateEstimator
from .pipeline import RoomPipeline
from .spatial import FloorMapper
from .vision.detector import Detection


@dataclass(frozen=True)
class Segment:
    person: int
    start_min: float
    end_min: float
    seat: tuple[float, float]
    activity: str  # desk | reading | meeting | quiet | phone
    talk: float = 0.0  # chance this person is speaking at a given moment
    enter: bool = True
    leave: bool = True


# activity -> (chance per second that a motion sensor fires, chance per frame the detector misses the person)
ACTIVITY = {
    "walking": (1.0, 0.05),
    "desk": (0.02, 0.10),
    "reading": (0.004, 0.10),
    "meeting": (0.10, 0.25),  # around a table people hide each other
    "quiet": (0.02, 0.10),
    "phone": (0.05, 0.10),
}

DEFAULT_DAY = [
    Segment(1, 20, 50, (3.0, 5.0), "desk", leave=False),
    Segment(1, 50, 80, (3.0, 5.0), "reading", enter=False),
    Segment(2, 90.0, 140, (1.4, 2.6), "meeting", talk=0.15),
    Segment(3, 90.5, 140, (2.6, 2.6), "meeting", talk=0.15),
    Segment(4, 91.0, 130, (1.4, 3.9), "meeting", talk=0.15),
    Segment(5, 91.5, 140, (2.6, 3.9), "meeting", talk=0.15),
    Segment(6, 150.0, 200, (3.0, 5.0), "quiet"),
    Segment(7, 150.5, 200, (1.0, 5.0), "quiet"),
    Segment(8, 205.0, 230, (3.0, 5.0), "phone", talk=0.5),
]
DEFAULT_DARK = [(110.0, 132.0)]  # minutes with the lights down for a presentation
ENTRANCE = (0.6, 0.4)
WALK_S = 6.0

SYSTEMS = ("motion sensor", "camera only", "camera + microphone")


def _position(seg: Segment, t: float, rng: np.random.Generator) -> tuple[tuple[float, float], str]:
    start, end = seg.start_min * 60, seg.end_min * 60
    seat, door = np.array(seg.seat), np.array(ENTRANCE)
    if seg.enter and t < start + WALK_S:
        a = (t - start) / WALK_S
        return tuple(door + a * (seat - door)), "walking"
    if seg.leave and t > end - WALK_S:
        a = (end - t) / WALK_S
        return tuple(door + a * (seat - door)), "walking"
    return tuple(seat + rng.normal(0, 0.04, 2)), seg.activity


def _box_for(mapper: FloorMapper, x: float, y: float, rng: np.random.Generator) -> tuple[float, ...]:
    """A plausible person box whose foot point projects to (x, y) on the floor."""
    (u, v), (ul, _), (ur, _) = mapper.to_image([(x, y), (x - 0.25, y), (x + 0.25, y)])
    w = max(8.0, abs(ur - ul))
    h = 3.2 * w
    j = rng.normal(0, 0.03, 4) * w
    return (u - w / 2 + j[0], v - h + j[1], u + w / 2 + j[2], v + j[3])


def simulate(
    cfg: RoomConfig,
    day: list[Segment] | None = None,
    dark: list[tuple[float, float]] | None = None,
    minutes: float = 240,
    dt: float = 0.5,
    seed: int = 7,
    sinks: list[Sink] | None = None,
) -> dict:
    day = DEFAULT_DAY if day is None else day
    dark = DEFAULT_DARK if dark is None else dark
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(cfg)
    cfg.audio.enabled = True

    speech = SpeechActivity(cfg.audio.window_s)
    pipeline = RoomPipeline(cfg, detector=None, sinks=sinks or [], speech=speech)
    camera_only = RoomStateEstimator(cfg.fusion, use_audio=False)
    room_poly = next((z.polygon for z in cfg.zones if z.name == "room"), None)
    xs = [p[0] for p in room_poly] if room_poly else [0, 4]
    ys = [p[1] for p in room_poly] if room_poly else [0, 6]

    steps = int(minutes * 60 / dt)
    rec = {k: np.zeros(steps) for k in ("t", "truth", "people", "speech", "meeting", "dark")}
    states = {s: [] for s in SYSTEMS}
    last_motion = float("-inf")
    pir_on = False

    for i in range(steps):
        t = i * dt
        t_min = t / 60
        is_dark = any(a <= t_min < b for a, b in dark)
        present = [s for s in day if s.start_min * 60 <= t < s.end_min * 60]

        detections, talking, motion, in_meeting = [], False, False, 0
        for seg in present:
            (x, y), activity = _position(seg, t, rng)
            p_motion, p_miss = ACTIVITY[activity]
            motion |= rng.random() < min(1.0, p_motion * dt)
            if activity == "meeting":
                in_meeting += 1
            if activity != "walking" and rng.random() < seg.talk:
                talking = True
            if not is_dark and rng.random() >= p_miss:
                detections.append(Detection(_box_for(pipeline.mapper, x, y, rng), 0.8))
        if not is_dark and rng.random() < 0.002:  # an occasional false alarm somewhere in the room
            fx, fy = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
            detections.append(Detection(_box_for(pipeline.mapper, fx, fy, rng), 0.4))

        # The speech detector works on 30 ms frames. Model it per frame: while someone talks, about 70% of
        # frames are voiced and caught; otherwise 2% of frames are false alarms (fans, doors, keyboards).
        n_frames = max(1, int(round(dt / 0.03)))
        frames = rng.random(n_frames) < (0.7 if talking else 0.02)
        speech.add(t, frames.tolist(), dt / n_frames)

        r = pipeline.process_detections(t, detections, vision_ok=not is_dark,
                                        vision_note="too dark (simulated)" if is_dark else "ok")
        zone_counts = {k: len(v) for k, v in r.zones.items() if k != "room"}
        camera_only.update(t, len(r.positions), zone_counts, None, not is_dark)

        if motion:
            last_motion = t
            pir_on = True
        elif pir_on and t - last_motion >= cfg.fusion.vacant_after_s:
            pir_on = False

        states["camera + microphone"].append(r.state)
        states["camera only"].append(camera_only.state)
        states["motion sensor"].append(RoomState.OCCUPIED if pir_on else RoomState.VACANT)
        for k, v in (("t", t), ("truth", len(present)), ("people", r.people),
                     ("speech", r.speech_ratio or 0.0), ("meeting", in_meeting >= 2), ("dark", is_dark)):
            rec[k][i] = v

    pipeline.close()
    metrics = score(rec, states, dt)
    return {"metrics": metrics, "series": rec, "states": states, "dt": dt}


def score(rec: dict, states: dict[str, list[RoomState]], dt: float) -> dict:
    occupied = rec["truth"] > 0
    out: dict = {"systems": {}}
    for name, series in states.items():
        st = np.array([s.value for s in series])
        vacant = st == RoomState.VACANT.value
        offs = int(np.sum((vacant[1:] & ~vacant[:-1]) & occupied[1:]))
        changes = int(np.sum(st[1:] != st[:-1]))
        entry = {
            "dark_while_occupied_min": round(float(np.sum(vacant & occupied) * dt / 60), 1),
            "switched_off_on_people": offs,
            "on_while_empty_min": round(float(np.sum(~vacant & ~occupied) * dt / 60), 1),
            "state_changes": changes,
        }
        if name == "camera + microphone":  # the only one that can tell a meeting from quiet work
            pred = st == RoomState.CONVERSATION.value
            truth = rec["meeting"] > 0
            tp = float(np.sum(pred & truth))
            entry["meeting_precision"] = round(tp / max(1.0, float(np.sum(pred))), 3)
            entry["meeting_recall"] = round(tp / max(1.0, float(np.sum(truth))), 3)
        out["systems"][name] = entry
    seen = rec["dark"] == 0
    out["people_count_mae"] = round(float(np.mean(np.abs(rec["people"][seen] - rec["truth"][seen]))), 3)
    out["people_count_exact"] = round(float(np.mean(rec["people"][seen] == rec["truth"][seen])), 3)
    return out


def format_table(metrics: dict) -> str:
    rows = [("", "off on people", "dark while occupied", "on while empty", "changes", "meeting P/R")]
    for name, m in metrics["systems"].items():
        pr = f"{m['meeting_precision']:.2f}/{m['meeting_recall']:.2f}" if "meeting_recall" in m else "n/a"
        rows.append((name, str(m["switched_off_on_people"]), f"{m['dark_while_occupied_min']} min",
                     f"{m['on_while_empty_min']} min", str(m["state_changes"]), pr))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    lines.append(f"people count (camera, when it can see): mean abs error {metrics['people_count_mae']}, "
                 f"exact {metrics['people_count_exact']:.0%} of the time")
    return "\n".join(lines)


def plot_timeline(result: dict, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec, states, dt = result["series"], result["states"], result["dt"]
    t_min = rec["t"] / 60
    colours = {RoomState.VACANT: "#d9d9d9", RoomState.OCCUPIED: "#5aa469", RoomState.CONVERSATION: "#3b6fb6"}

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1, 2]})
    ax = axes[0]
    ax.step(t_min, rec["truth"], where="post", color="black", lw=1.4, label="people actually in the room")
    ax.step(t_min, rec["people"], where="post", color="#e07b39", lw=1.0, label="camera estimate")
    ax.set_ylabel("people")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(t_min, rec["speech"], color="#7a5195", lw=0.9)
    ax.set_ylabel("speech\nratio")
    ax.set_ylim(0, 1)

    for a in axes[:2]:
        for start, on in _runs(rec["dark"] > 0):
            if on:
                a.axvspan(t_min[start[0]], t_min[min(start[1], len(t_min) - 1)], color="#222222", alpha=0.12, lw=0)

    ax = axes[2]
    occupied = rec["truth"] > 0
    for row, name in enumerate(SYSTEMS):
        st = states[name]
        for (a, b), s in _runs(st):
            ax.broken_barh([(t_min[a], (b - a) * dt / 60)], (row - 0.35, 0.7), color=colours[s], lw=0)
        bad = np.array([s == RoomState.VACANT for s in st]) & occupied
        for (a, b), on in _runs(bad):
            # outline switch-offs only; the few seconds it takes to confirm a new arrival are not marked
            if on and a > 0 and st[a - 1] != RoomState.VACANT:
                ax.broken_barh([(t_min[a], (b - a) * dt / 60)], (row - 0.42, 0.84), facecolor="none",
                               edgecolor="#c0392b", lw=1.2)
    ax.set_yticks(range(len(SYSTEMS)), SYSTEMS)
    ax.set_xlabel("minutes")
    ax.set_xlim(0, t_min[-1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#c0392b"))
    ax.legend(handles, ["vacant", "occupied", "conversation", "switched off on people"], loc="lower right",
              fontsize=8, ncol=4, frameon=False, bbox_to_anchor=(1, 1.0))
    fig.suptitle("Synthetic 4-hour day: grey bands = lights down, camera can't see", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _runs(values) -> list[tuple[tuple[int, int], object]]:
    """Run-length encode a sequence into ((start, end), value) pairs."""
    out, start = [], 0
    values = list(values)
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append(((start, i), values[start]))
            start = i
    return out


def run_simulation(cfg: RoomConfig, out_dir: str | Path, seed: int = 7, extra_sinks: list[Sink] | None = None,
                   plot: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "decisions.jsonl"
    log_path.unlink(missing_ok=True)
    sinks: list[Sink] = [JsonlSink(log_path), *(extra_sinks or [])]
    result = simulate(cfg, seed=seed, sinks=sinks)
    if plot:
        try:
            result["plot"] = str(plot_timeline(result, out_dir / "timeline.png"))
        except ImportError:
            result["plot"] = None
    return result
