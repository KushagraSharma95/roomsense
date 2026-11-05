from pathlib import Path

import numpy as np
import pytest

from roomsense.vision.detector import Detection, decode_yolox, frame_quality, letterbox, make_grids, nms
from roomsense.vision.tracker import IouTracker, iou_matrix

MODEL = Path(__file__).resolve().parents[1] / "models" / "yolox_nano.onnx"


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 20], [1, 1, 11, 21], [50, 50, 60, 70]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.45) == [0, 2]


def test_letterbox_scales_and_pads():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, ratio = letterbox(frame, (416, 416))
    assert tensor.shape == (3, 416, 416)
    assert ratio == pytest.approx(0.65)
    assert tensor[0, 400, 0] == 114  # padding below the resized image


def test_decode_yolox_finds_planted_person_and_ignores_other_classes():
    size = (416, 416)
    grids, strides = make_grids(size)
    raw = np.zeros((len(grids), 85), dtype=np.float32)
    # a person centred on grid cell (10, 20) at stride 8 -> centre (84, 164) px, box 64 x 128 px
    i = 20 * 52 + 10
    raw[i, :4] = [0.5, 0.5, np.log(64 / 8), np.log(128 / 8)]
    raw[i, 4], raw[i, 5] = 0.9, 0.95
    # a confident "car" somewhere else must be ignored
    raw[i + 5, :4] = [0.5, 0.5, 1.0, 1.0]
    raw[i + 5, 4], raw[i + 5, 5 + 2] = 0.9, 0.95
    dets = decode_yolox(raw, ratio=0.5, grids=grids, strides=strides, score_threshold=0.3, nms_threshold=0.45)
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].box
    assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((168, 328))  # divided by the 0.5 scale
    assert (x2 - x1, y2 - y1) == pytest.approx((128, 256))
    assert dets[0].score == pytest.approx(0.9 * 0.95)


def test_frame_quality_flags_dark_and_blank_frames():
    rng = np.random.default_rng(0)
    assert frame_quality(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))[0]
    assert not frame_quality(np.full((240, 320, 3), 5, dtype=np.uint8))[0]  # lights off
    assert not frame_quality(np.full((240, 320, 3), 128, dtype=np.uint8))[0]  # lens covered


def _det(x, y, w=40, h=100):
    return Detection((x, y, x + w, y + h), 0.9)


def test_tracker_keeps_ids_when_two_people_cross():
    trk = IouTracker(iou_threshold=0.2, min_hits=2, max_missed=5)
    for step in range(30):
        a, b = 100 + step * 6, 280 - step * 6
        tracks = trk.update([_det(a, 100), _det(b, 104)])
    ids = sorted(t.track_id for t in tracks)
    assert ids == [1, 2]
    left_to_right = next(t for t in tracks if t.track_id == 1)
    assert left_to_right.box[0] > 200


def test_tracker_bridges_short_occlusion_and_ignores_one_frame_false_alarm():
    trk = IouTracker(iou_threshold=0.2, min_hits=3, max_missed=10, report_missed=4)
    for _ in range(5):
        trk.update([_det(100, 100)])
    for _ in range(3):  # hidden for three frames: still counted
        assert [t.track_id for t in trk.update([])] == [1]
    assert [t.track_id for t in trk.update([_det(100, 100)])] == [1]
    assert len(trk.update([_det(100, 100), _det(400, 300)])) == 1  # a single-frame blip is not a person
    assert len(trk.update([_det(100, 100)])) == 1


def test_tracker_stops_counting_someone_who_left():
    trk = IouTracker(min_hits=2, max_missed=9, report_missed=3)
    for _ in range(5):
        trk.update([_det(100, 100)])
    counts = [len(trk.update([])) for _ in range(12)]
    assert counts[:3] == [1, 1, 1] and counts[3:] == [0] * 9
    assert trk.tracks == []


def test_iou_matrix_shapes():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


@pytest.mark.skipif(not MODEL.exists(), reason="run `roomsense models download nano` first")
def test_onnx_detector_runs_on_cpu():
    from roomsense.vision.detector import YoloxDetector

    det = YoloxDetector(MODEL, threads=1)
    assert det.input_size == (416, 416)
    out = det.detect(np.full((480, 640, 3), 90, dtype=np.uint8))
    assert out == []  # a flat grey frame has nobody in it
