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
    cfg = parse_config({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                        "policy": {"actions": {"vacant": {"lights": False}, "occupied": {"av": True}}}})
    assert cfg.policy.actions == {"vacant": {"lights": "off"}, "occupied": {"av": "on"}}


@pytest.mark.parametrize("bad, message", [
    ({"zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}]}, "room_id"),
    ({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0]]}]}, "at least 3"),
    ({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
      "detector": {"kind": "magic"}}, "detector.kind"),
    ({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
      "tracker": {"iou": 0.3}}, "unknown key"),
    ({"room_id": "x", "zones": [{"name": "room", "polygon": [[0, 0], [1, 0], [1, 1]]}],
      "policy": {"actions": {"party": {"lights": "disco"}}}}, "unknown state"),
])
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
    cfg = QrcConfig(enabled=True, host=core.address[0], port=core.address[1], user="ops", password="secret",
                    controls={"lights.meeting": {"control": "LightingScene", "value": 2},
                              "av.on": {"component": "Display", "control": "power", "value": 1}})
    sink = QrcSink(cfg, keepalive_s=3600)
    sink.publish_decision(_decision("lights", "meeting"))
    sink.publish_decision(_decision("av", "on"))
    sink.publish_decision(_decision("hvac", "occupied"))  # unmapped: ignored, no error
    sink.close()
    assert core.controls == {"LightingScene": 2}
    assert core.components == {"Display": {"power": 1}}
    assert [m["method"] for m in core.log] == ["Logon", "Control.Set", "Component.Set"]


class FakeMqtt:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.messages.append((topic, payload, retain))


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
