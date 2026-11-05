import json
from pathlib import Path

from roomsense.cli import main
from roomsense.config import load_config
from roomsense.sim import simulate

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "room.example.yaml"


def test_simulated_day_fused_system_never_switches_off_on_people():
    cfg = load_config(EXAMPLE)
    cfg.outputs.console = False
    result = simulate(cfg, seed=3)
    m = result["metrics"]["systems"]
    assert m["camera + microphone"]["switched_off_on_people"] == 0
    assert m["motion sensor"]["switched_off_on_people"] >= 1  # still readers get the lights switched off
    assert m["camera only"]["switched_off_on_people"] >= 1  # the dark presentation is lost without audio
    assert m["camera + microphone"]["meeting_recall"] > 0.8
    assert result["metrics"]["people_count_mae"] < 0.5


def test_simulate_cli_writes_log_and_metrics(tmp_path):
    out = tmp_path / "sim"
    assert main(["simulate", "--config", str(EXAMPLE), "--out", str(out), "--no-plot", "--json",
                 "--qrc-mock"]) == 0
    lines = (out / "decisions.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert {"t", "target", "value", "state", "reason", "evidence"} <= set(first)
    assert json.loads((out / "metrics.json").read_text())["systems"]
