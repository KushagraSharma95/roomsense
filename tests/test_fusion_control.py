from roomsense.config import FusionConfig
from roomsense.control import ControlPolicy
from roomsense.fusion import RoomState, RoomStateEstimator

CFG = FusionConfig(occupied_after_s=2, vacant_after_s=60, degraded_vacant_after_s=180, conversation_min_people=2,
                   conversation_speech_ratio=0.2, conversation_after_s=5, conversation_end_s=20,
                   speech_presence_ratio=0.1, count_window_s=3)
ACTIONS = {
    "vacant": {"lights": "off", "av": "standby"},
    "occupied": {"lights": "work"},
    "conversation": {"lights": "meeting", "av": "on"},
}


def run(est, seconds, people, speech=None, vision_ok=True, start=0.0, dt=1.0):
    changes = []
    t = start
    while t < start + seconds:
        c = est.update(t, people, {}, speech, vision_ok)
        if c:
            changes.append(c)
        t += dt
    return changes, t


def test_person_sitting_still_keeps_the_room_occupied():
    est = RoomStateEstimator(CFG)
    changes, t = run(est, 600, people=1, speech=0.0)  # ten minutes, no speech, no motion needed
    assert [c.new for c in changes] == [RoomState.OCCUPIED]
    assert "1 person in view" in changes[0].reason


def test_room_goes_vacant_only_after_the_timeout():
    est = RoomStateEstimator(CFG)
    _, t = run(est, 30, people=1)
    changes, t = run(est, 59, people=0, start=t)
    assert changes == [] and est.state == RoomState.OCCUPIED
    changes, _ = run(est, 5, people=0, start=t)
    assert changes[0].new == RoomState.VACANT and "nobody seen or heard" in changes[0].reason


def test_brief_false_detection_does_not_turn_the_room_on():
    est = RoomStateEstimator(CFG)
    assert est.update(0, 1) is None and est.update(1, 0) is None and est.update(2, 0) is None
    assert est.state == RoomState.VACANT


def test_meeting_is_recognised_and_ends():
    est = RoomStateEstimator(CFG)
    changes, t = run(est, 30, people=3, speech=0.4)
    assert [c.new for c in changes] == [RoomState.OCCUPIED, RoomState.CONVERSATION]
    changes, _ = run(est, 30, people=3, speech=0.0, start=t)
    assert [c.new for c in changes] == [RoomState.OCCUPIED]


def test_one_person_on_a_call_is_not_a_meeting():
    est = RoomStateEstimator(CFG)
    run(est, 60, people=1, speech=0.5)
    assert est.state == RoomState.OCCUPIED


def test_dark_room_with_talking_stays_occupied_on_audio():
    est = RoomStateEstimator(CFG)
    _, t = run(est, 30, people=3, speech=0.4)
    assert est.state == RoomState.CONVERSATION
    changes, _ = run(est, 300, people=0, speech=0.4, vision_ok=False, start=t)  # lights down, still talking
    assert changes == [] and est.state == RoomState.CONVERSATION


def test_camera_only_loses_a_dark_room_after_the_degraded_timeout():
    est = RoomStateEstimator(CFG, use_audio=False)
    _, t = run(est, 30, people=2, speech=0.4)
    changes, _ = run(est, 200, people=0, speech=0.4, vision_ok=False, start=t)
    assert changes[-1].new == RoomState.VACANT and "audio-only timeout" in changes[-1].reason


def _change(est, t, people, speech=None):
    return est.update(t, people, {}, speech)


def test_policy_tick_releases_held_change():
    est = RoomStateEstimator(CFG)
    policy = ControlPolicy("r1", ACTIONS, min_hold_s=30)
    out = []
    for t in range(0, 12):
        c = _change(est, t, 3, 0.5)
        if c:
            out += policy.on_state_change(c)
        out += policy.tick(t)
    assert [(d.action.target, d.action.value) for d in out] == [("lights", "work"), ("av", "on")]
    assert policy.pending["lights"].action.value == "meeting"
    released = policy.tick(40)
    assert [(d.action.target, d.action.value) for d in released] == [("lights", "meeting")]
    assert released[0].deferred_s > 0 and "speech" in released[0].reason
    assert policy.current == {"lights": "meeting", "av": "on"}


def test_held_change_is_dropped_when_the_room_changes_back():
    policy = ControlPolicy("r1", ACTIONS, min_hold_s=30)
    est = RoomStateEstimator(CFG)
    for t in range(0, 8):
        c = _change(est, t, 3, 0.5)
        if c:
            policy.on_state_change(c)
    assert "lights" in policy.pending  # meeting lights waiting for the hold to pass
    for t in range(8, 40):  # the meeting ends quickly: room is back to plain "occupied"
        c = _change(est, t, 3, 0.0)
        if c:
            assert c.new == RoomState.OCCUPIED
            policy.on_state_change(c)
    assert "lights" not in policy.pending and policy.tick(60) == []
    assert policy.current["lights"] == "work"


def test_speech_alone_cannot_switch_on_an_empty_room_the_camera_can_see():
    est = RoomStateEstimator(CFG)
    changes, _ = run(est, 60, people=0, speech=0.5)  # a TV talking in an empty room
    assert changes == [] and est.state == RoomState.VACANT


def test_speech_switches_on_a_room_the_camera_cannot_see():
    est = RoomStateEstimator(CFG)
    changes, _ = run(est, 10, people=0, speech=0.5, vision_ok=False)
    assert [c.new for c in changes] == [RoomState.OCCUPIED] and "speech" in changes[0].reason
