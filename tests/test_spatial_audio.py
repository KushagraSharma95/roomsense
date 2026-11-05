import numpy as np
import pytest

from roomsense.audio import SpeechActivity, SpeechDetector
from roomsense.config import CameraConfig, Zone
from roomsense.spatial import FloorMapper, ZoneMap, point_in_polygon

SR = 16000


def test_homography_maps_calibration_points_both_ways():
    cam = CameraConfig(image_points=[(100, 500), (700, 500), (600, 200), (200, 200)],
                       floor_points=[(0, 0), (4, 0), (4, 6), (0, 6)])
    m = FloorMapper(cam)
    floor = m.to_floor(cam.image_points)
    assert floor == pytest.approx(np.array(cam.floor_points), abs=1e-6)
    assert m.to_image(floor) == pytest.approx(np.array(cam.image_points), abs=1e-6)
    x, y = m.to_floor([(400, 350)])[0]  # a point between the near and far edge lands mid-room
    assert 1.5 < x < 2.5 and 1.0 < y < 5.0


def test_uncalibrated_mapper_is_identity():
    m = FloorMapper(None)
    assert not m.calibrated
    assert m.to_floor([(12.5, 40.0)])[0] == pytest.approx((12.5, 40.0))


def test_zones_and_room_boundary():
    zones = ZoneMap([Zone("room", [(0, 0), (4, 0), (4, 6), (0, 6)]), Zone("table", [(1, 2), (3, 2), (3, 4), (1, 4)])])
    pos = {1: (2.0, 3.0), 2: (0.5, 5.0), 3: (5.0, 1.0)}  # 3 is outside the room (corridor)
    assert zones.people_in_room(pos) == [1, 2]
    assert zones.assign(pos)["table"] == [1]
    assert point_in_polygon(2, 3, [(1, 2), (3, 2), (3, 4), (1, 4)])
    assert not point_in_polygon(0.5, 0.5, [(1, 2), (3, 2), (3, 4), (1, 4)])


def _speech_like(seconds: float, rng: np.random.Generator, level: float = 0.1) -> np.ndarray:
    """A crude voice: harmonics of a wandering pitch, shaped by three vowel formants, spoken in
    4 Hz syllables and phrases with pauses."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = np.zeros_like(t)
    for k in range(1, 25):
        f = k * 140.0
        formants = (3 * np.exp(-(((f - 600) / 150) ** 2)) + 2 * np.exp(-(((f - 1400) / 250) ** 2))
                    + np.exp(-(((f - 2500) / 300) ** 2)))
        voice += (1.0 / k) * (0.3 + formants) * np.sin(k * phase)
    syllables = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    phrases = (np.sin(2 * np.pi * 0.25 * t) > -0.3).astype(float)  # talk ~60% of the time, then pause
    sig = level * voice / np.abs(voice).max() * syllables * phrases
    return (sig + rng.normal(0, 0.002, len(t))).astype(np.float32)


def _ratio(samples: np.ndarray) -> float:
    flags = SpeechDetector(SR).process(samples)
    return float(np.mean(flags[100:]))  # skip the first 3 s while the noise floor settles


def test_speech_is_detected():
    assert _ratio(_speech_like(20, np.random.default_rng(1))) > 0.35


def test_steady_fan_noise_is_not_speech():
    rng = np.random.default_rng(2)
    brown = np.cumsum(rng.normal(0, 1, 20 * SR))
    brown = brown - np.convolve(brown, np.ones(400) / 400, mode="same")  # remove drift
    fan = (0.05 * brown / np.abs(brown).max()).astype(np.float32)
    assert _ratio(fan) < 0.05


def test_silence_and_clicks_are_not_speech():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.001, 20 * SR).astype(np.float32)
    for start in range(SR, 20 * SR, 2 * SR):  # a door click or keyboard clack every 2 s
        x[start:start + 200] += rng.normal(0, 0.3, 200).astype(np.float32)
    assert _ratio(x) < 0.08


def test_speech_activity_window():
    act = SpeechActivity(window_s=10)
    assert act.ratio(0) is None
    act.add(0.0, [True] * 50 + [False] * 50, 0.1)  # 5 s talking, 5 s quiet
    assert act.ratio(10.0) == pytest.approx(0.5, abs=0.02)
    assert act.ratio(30.0) is None  # all of it has aged out
