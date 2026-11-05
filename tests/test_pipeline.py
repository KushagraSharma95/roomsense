import numpy as np

from roomsense.config import parse_config
from roomsense.pipeline import RoomPipeline, draw_overlay
from roomsense.vision.detector import Detection


def _cfg(**fusion):
    return parse_config(
        {
            "room_id": "r1",
            "zones": [{"name": "room", "polygon": [[0, 0], [640, 0], [640, 480], [0, 480]]}],
            "tracker": {"min_hits": 3},
            "audio": {"enabled": False},
            "fusion": {"occupied_after_s": 2, "vacant_after_s": 30, **fusion},
            "policy": {"min_hold_s": 5, "actions": {"vacant": {"lights": "off"}, "occupied": {"lights": "work"}}},
        }
    )


def test_devices_are_synced_once_after_start_up_in_an_empty_room():
    p = RoomPipeline(_cfg())
    sent = []
    for k in range(80):
        sent += p.process_detections(float(k), []).decisions
    assert [(d.action.key, d.t) for d in sent] == [("lights.off", 30.0)]
    assert "start-up" in sent[0].reason


def test_occupied_room_at_start_up_is_commanded_by_the_normal_state_change():
    p = RoomPipeline(_cfg())
    sent = []
    for k in range(20):
        sent += p.process_detections(float(k), [Detection((100, 100, 160, 300), 0.9)]).decisions
    assert [d.action.key for d in sent] == ["lights.work"]


def test_preview_blurs_people_before_their_track_is_confirmed():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)  # lots of detail, so blur is visible
    p = RoomPipeline(_cfg())
    r = p.process_detections(0.0, [Detection((100, 100, 160, 300), 0.9)])  # first sighting: not confirmed
    assert r.tracks == []
    plain = draw_overlay(frame, r, p, blur=False)
    blurred = draw_overlay(frame, r, p, blur=True)
    roi = (slice(120, 280), slice(110, 150))
    assert np.array_equal(plain[roi], frame[roi])
    assert np.abs(blurred[roi].astype(int) - frame[roi].astype(int)).mean() > 20
