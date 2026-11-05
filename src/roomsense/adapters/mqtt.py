"""MQTT output, the usual bridge into building-management and IoT platforms.

Topics (prefix and room come from the config):
  <prefix>/<room>/status          retained JSON: state, people, zones, speech ratio
  <prefix>/<room>/<target>/set    JSON command, e.g. {"value": "meeting", "reason": "..."}
  <prefix>/<room>/decisions       every decision with its evidence
"""

from __future__ import annotations

import json

from ..config import MqttConfig
from ..control import Decision


class MqttSink:
    def __init__(self, cfg: MqttConfig, room_id: str, client=None) -> None:
        self.prefix = f"{cfg.topic_prefix}/{room_id}"
        if client is None:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"roomsense-{room_id}")
            client.will_set(f"{self.prefix}/online", "false", qos=1, retain=True)
            client.connect(cfg.host, cfg.port, keepalive=30)
            client.loop_start()
            client.publish(f"{self.prefix}/online", "true", qos=1, retain=True)
        self.client = client

    def publish_status(self, status: dict) -> None:
        self.client.publish(f"{self.prefix}/status", json.dumps(status), qos=0, retain=True)

    def publish_decision(self, d: Decision) -> None:
        cmd = {"value": d.action.value, "state": d.state, "reason": d.reason, "t": round(d.t, 3)}
        self.client.publish(f"{self.prefix}/{d.action.target}/set", json.dumps(cmd), qos=1)
        self.client.publish(f"{self.prefix}/decisions", json.dumps(d.as_dict()), qos=1)

    def close(self) -> None:
        self.client.publish(f"{self.prefix}/online", "false", qos=1, retain=True)
        if hasattr(self.client, "loop_stop"):
            self.client.loop_stop()
        if hasattr(self.client, "disconnect"):
            self.client.disconnect()
