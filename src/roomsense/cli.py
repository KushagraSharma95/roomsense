"""Command line: roomsense {run, simulate, bench, models}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .adapters.sinks import ConsoleSink, JsonlSink, SharedState, Sink
from .config import ConfigError, RoomConfig, load_config

DEFAULT_CONFIG = "configs/room.example.yaml"


def build_sinks(cfg: RoomConfig, shared: SharedState | None = None, jsonl: bool = True) -> list[Sink]:
    sinks: list[Sink] = []
    if cfg.outputs.console:
        sinks.append(ConsoleSink())
    if jsonl and cfg.outputs.jsonl_path:
        sinks.append(JsonlSink(cfg.outputs.jsonl_path))
    if cfg.outputs.mqtt.enabled:
        from .adapters.mqtt import MqttSink

        sinks.append(MqttSink(cfg.outputs.mqtt, cfg.room_id))
    if cfg.outputs.qrc.enabled:
        from .adapters.qrc import QrcSink

        sinks.append(QrcSink(cfg.outputs.qrc))
    if shared is not None:
        sinks.append(shared)
    return sinks


def cmd_models(args: argparse.Namespace) -> int:
    from .models import download

    for name in args.names:
        path = download(name, args.dir)
        print(f"{name}: {path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .sim import format_table, run_simulation

    cfg = load_config(args.config)
    sinks = build_sinks(cfg, jsonl=False)  # the simulation writes its own decision log into --out
    core = None
    if args.qrc_mock:
        from .adapters.qrc import QrcSink
        from .adapters.qrc_mock import MockCore

        core = MockCore()
        host, port = core.start()
        cfg.outputs.qrc.host, cfg.outputs.qrc.port, cfg.outputs.qrc.enabled = host, port, True
        cfg.outputs.qrc.user = cfg.outputs.qrc.password = ""
        sinks.append(QrcSink(cfg.outputs.qrc))
        print(f"mock Q-SYS Core listening on {host}:{port}")
    result = run_simulation(cfg, args.out, seed=args.seed, extra_sinks=sinks, plot=not args.no_plot)
    print()
    print(format_table(result["metrics"]))
    print(f"\ndecision log: {Path(args.out) / 'decisions.jsonl'}")
    if result.get("plot"):
        print(f"timeline:     {result['plot']}")
    if core is not None:
        print(f"mock Core controls after the day: {core.controls} components: {core.components}")
        core.stop()
    if args.json:
        Path(args.out, "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from .bench import benchmark
    from .vision.detector import YoloxDetector

    detector = YoloxDetector(args.model, threads=args.threads)
    frames = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while len(frames) < 50:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if not frames:
        frames = [np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)]
    print(json.dumps(benchmark(detector, frames, runs=args.runs), indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import cv2

    from .audio import MicFeed, SpeechActivity, SpeechDetector, WavFeed
    from .pipeline import RoomPipeline, draw_overlay
    from .vision.detector import MotionDetector, YoloxDetector

    cfg = load_config(args.config)
    d = cfg.detector
    detector = (YoloxDetector(d.model_path, d.score_threshold, d.nms_threshold, d.threads)
                if d.kind == "yolox" else MotionDetector())

    shared = None
    if args.api or cfg.outputs.api.enabled:
        from .api import serve_in_background

        shared = SharedState()
        serve_in_background(shared, cfg.outputs.api.host, cfg.outputs.api.port)
        print(f"API on http://{cfg.outputs.api.host}:{cfg.outputs.api.port}  (/status, /decisions, /ws, /preview.jpg)")

    use_audio = cfg.audio.enabled and args.audio != "none"
    cfg.audio.enabled = use_audio
    speech = SpeechActivity(cfg.audio.window_s) if use_audio else None
    pipeline = RoomPipeline(cfg, detector, build_sinks(cfg, shared), speech)

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"could not open video source {args.video!r}", file=sys.stderr)
        return 2
    is_file = isinstance(source, str) and Path(source).exists()
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    t0 = time.monotonic()

    def clock() -> float:
        return time.monotonic() - t0

    wav = mic = None
    if use_audio:
        sd = SpeechDetector(cfg.audio.sample_rate)
        if args.audio == "mic":
            try:
                mic = MicFeed(sd, speech, clock)
            except (ImportError, OSError) as e:
                print(f"microphone unavailable ({e}); install with: pip install 'roomsense[audio]'",
                      file=sys.stderr)
                return 2
            mic.start()
        else:
            wav = WavFeed(args.audio, sd, speech)

    writer = None
    show = args.show
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps if is_file else clock()
            frame_idx += 1
            if wav:
                wav.advance_to(t)
            r = pipeline.process_frame(t, frame)

            if show or args.record or shared is not None:
                vis = draw_overlay(frame, r, pipeline, blur=not args.no_blur)
                if shared is not None:
                    shared.set_preview(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                if args.record:
                    if writer is None:
                        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if show:
                    try:
                        cv2.imshow("roomsense", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        print("no display support in this OpenCV build; use --api and open /preview.jpg, "
                              "or install opencv-python instead of opencv-python-headless", file=sys.stderr)
                        show = False
            if args.max_seconds and t >= args.max_seconds:
                break
            if is_file and args.realtime:
                time.sleep(max(0.0, t - clock()))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if mic:
            mic.stop()
        pipeline.close()
    ev = pipeline.estimator.last_evidence
    print(f"processed {frame_idx} frames; final state {pipeline.estimator.state.value}, "
          f"people {ev.people if ev else 0}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roomsense", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run on a webcam, video file or stream")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--video", default="0", help="camera index, file path or RTSP URL (default: 0)")
    p.add_argument("--audio", default="none", help="'mic', a 16-bit WAV file, or 'none' (default)")
    p.add_argument("--api", action="store_true", help="serve /status, /decisions, /ws and /preview.jpg")
    p.add_argument("--show", action="store_true", help="open a preview window (needs opencv-python)")
    p.add_argument("--record", help="write the annotated video to this .mp4 file")
    p.add_argument("--no-blur", action="store_true", help="don't blur people in previews and recordings")
    p.add_argument("--realtime", action="store_true", help="play video files at their real speed")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("simulate", help="run a synthetic 4-hour day and compare against a motion sensor")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default="out/sim")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--json", action="store_true", help="also write metrics.json")
    p.add_argument("--qrc-mock", action="store_true", help="send decisions to a local mock Q-SYS Core")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("bench", help="measure detector latency on this CPU")
    p.add_argument("--model", default="models/yolox_nano.onnx")
    p.add_argument("--video", help="take frames from this video (default: random 1280x720 frames)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("models", help="download detector weights")
    msub = p.add_subparsers(dest="models_cmd", required=True)
    pd = msub.add_parser("download")
    pd.add_argument("names", nargs="*", default=["nano"], help="nano and/or tiny (default: nano)")
    pd.add_argument("--dir", default="models")
    pd.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
