import json
from pathlib import Path

import pytest

from roomsense.adapters.qrc import QrcClient, QrcError, QrcSink
from roomsense.adapters.qrc_mock import MockCore
from roomsense.adapters.sinks import SharedState
from roomsense.config import ConfigError, MqttConfig, QrcConfig, load_config, parse_config
from roomsense.control import Action, Decision

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "room.example.yaml"


def _decision(target: str, value: str) -> Decision:
    return Decision(12.0, "r1", Action(target, value), "conversation", "3 people in view and speech", {"people": 3})


def test_example_config_loads_and_quoted_words_survive_yaml():
    cfg = load_config(EXAMPLE)
    assert cfg.room_id == "huddle-room-1"
    assert cfg.policy.actions["vacant"]["lights"] == "off"
    assert cfg.policy.actions["conversation"]["av"] == "on"
    assert {z.name for z in cfg.zones} >= {"room", "table"}


def test_bare_yaml_booleans_become_on_off():
    cfg = parse_config(
        {
            "room_id": "x",
            "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
            "policy": {"actions": {"vacant": {"lights": False}, "occupied": {"av": True}}},
        }
    )
    assert cfg.policy.actions == {"vacant": {"lights": "off"}, "occupied": {"av": "on"}}


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}]}, "room_id"),
        ({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0]]}]}, "at least 3"),
        (
            {
                "room_id": "x",
                "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                "detector": {"kind": "magic"},
            },
            "detector.kind",
        ),
        (
            {"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}], "tracker": {"iou": 0.3}},
            "unknown key",
        ),
        (
            {
                "room_id": "x",
                "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                "policy": {"actions": {"party": {"lights": "disco"}}},
            },
            "unknown state",
        ),
        (
            {"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}], "fusoin": {}},
            "unknown key",
        ),
        (
            {
                "room_id": "x",
                "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                "fusion": {"vacant_after_s": "60"},
            },
            "must be a number",
        ),
        (
            {
                "room_id": "x",
                "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                "outputs": {"jsonl": "a.jsonl"},
            },
            "unknown key",
        ),
    ],
)
def test_bad_configs_fail_with_a_clear_message(bad, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(bad)


@pytest.fixture
def core():
    c = MockCore(user="ops", password="secret")
    host, port = c.start()
    c.address = (host, port)
    yield c
    c.stop()


def test_qrc_client_logon_set_and_get(core):
    client = QrcClient(*core.address)
    with pytest.raises(QrcError) as e:
        client.control_set("LightingScene", 2)
    assert e.value.code == 10  # logon required
    assert client.logon("ops", "secret") is True
    client.control_set("LightingScene", 2, ramp=1.5)
    assert client.control_get(["LightingScene"]) == [{"Name": "LightingScene", "Value": 2}]
    with pytest.raises(QrcError) as e:
        client.control_get(["NoSuchControl"])
    assert e.value.code == 8
    assert client.status()["State"] == "Active"
    assert any(n.get("method") == "EngineStatus" for n in client.notifications)  # push on connect was kept
    client.close()


def test_qrc_sink_maps_actions_to_controls(core):
    cfg = QrcConfig(
        enabled=True,
        host=core.address[0],
        port=core.address[1],
        user="ops",
        password="secret",
        controls={
            "lights.meeting": {"control": "LightingScene", "value": 2},
            "av.on": {"component": "Display", "control": "power", "value": 1},
        },
    )
    sink = QrcSink(cfg, keepalive_s=3600)
    sink.publish_decision(_decision("lights", "meeting"))
    sink.publish_decision(_decision("av", "on"))
    sink.publish_decision(_decision("hvac", "occupied"))  # unmapped: ignored, no error
    assert sink.flush(timeout=3)
    sink.close()
    assert core.controls == {"LightingScene": 2}
    assert core.components == {"Display": {"power": 1}}
    assert core.log[0]["method"] == "Logon"
    assert sorted(m["method"] for m in core.log[1:]) == ["Component.Set", "Control.Set"]


def test_qrc_sink_never_blocks_and_catches_up_after_an_outage():
    import time

    probe = MockCore()
    host, port = probe.start()
    probe.stop()  # the Core is down
    cfg = QrcConfig(
        enabled=True,
        host=host,
        port=port,
        user="ops",
        password="secret",
        controls={"lights.off": {"control": "LightingScene", "value": 0}},
    )
    sink = QrcSink(cfg, keepalive_s=3600, retry_s=0.05)
    start = time.monotonic()
    sink.publish_decision(_decision("lights", "off"))
    assert time.monotonic() - start < 0.05  # the video loop is never held up by the network
    assert not sink.flush(timeout=0.3)  # still undelivered while the Core is away
    core = MockCore(user="ops", password="secret")
    core.start(host, port)  # the Core comes back
    try:
        assert sink.flush(timeout=5)
        assert core.controls == {"LightingScene": 0}  # the room's latest wish arrives after reconnecting
        assert core.log[0]["method"] == "Logon"  # and the new connection logged on first
    finally:
        sink.close()
        core.stop()


class FakeMqtt:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.messages.append((topic, payload, retain))


def test_mqtt_sink_says_online_again_after_reconnecting():
    from roomsense.adapters.mqtt import MqttSink

    fake = FakeMqtt()
    sink = MqttSink(MqttConfig(enabled=True, topic_prefix="bldg"), "r1", client=fake)
    sink._on_connect(fake, None, {}, 0, None)  # what paho calls after every (re)connect
    assert fake.messages == [("bldg/r1/online", "true", True)]


def test_mqtt_sink_topics():
    from roomsense.adapters.mqtt import MqttSink

    fake = FakeMqtt()
    sink = MqttSink(MqttConfig(enabled=True, topic_prefix="bldg"), "r1", client=fake)
    sink.publish_status({"state": "occupied", "people": 1})
    sink.publish_decision(_decision("lights", "meeting"))
    topics = [m[0] for m in fake.messages]
    assert topics == ["bldg/r1/status", "bldg/r1/lights/set", "bldg/r1/decisions"]
    assert fake.messages[0][2] is True  # status is retained so new subscribers see the current state
    assert json.loads(fake.messages[1][1])["value"] == "meeting"


def test_api_status_and_decisions():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from roomsense.api import create_app

    state = SharedState()
    state.publish_status({"room": "r1", "state": "occupied", "people": 2})
    state.publish_decision(_decision("lights", "work"))
    client = TestClient(create_app(state))
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/status").json()["people"] == 2
    assert client.get("/decisions").json()[0]["value"] == "work"
    assert client.get("/preview.jpg").status_code == 204
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["state"] == "occupied"


def test_cli_rejects_unknown_model_names():
    from roomsense.cli import main

    with pytest.raises(SystemExit) as e:
        main(["models", "download", "nanoo"])
    assert e.value.code == 2
