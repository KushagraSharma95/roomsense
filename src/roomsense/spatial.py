"""Image-to-floor mapping and zones.

A homography from four or more image/floor point pairs turns each person's foot point into floor
coordinates in metres. Zones are polygons on that floor plan, so "who is at the table" does not
depend on where the camera happens to be mounted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from .config import CameraConfig, Zone


class FloorMapper:
    def __init__(self, camera: CameraConfig | None = None) -> None:
        if camera and camera.image_points and camera.floor_points:
            src = np.asarray(camera.image_points, dtype=np.float64)
            dst = np.asarray(camera.floor_points, dtype=np.float64)
            h, _ = cv2.findHomography(src, dst, 0)
            if h is None:
                raise ValueError("camera points are degenerate (three of them on one line?)")
            self.h = h
            self.calibrated = True
        else:
            self.h = np.eye(3)
            self.calibrated = False
        self.h_inv = np.linalg.inv(self.h)

    @staticmethod
    def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        return cv2.perspectiveTransform(pts[None], h)[0]

    def to_floor(self, image_pts: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self._apply(self.h, np.asarray(image_pts))

    def to_image(self, floor_pts: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self._apply(self.h_inv, np.asarray(floor_pts))


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray casting. Points exactly on an edge may land either way, which is fine for occupancy."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


class ZoneMap:
    ROOM = "room"

    def __init__(self, zones: Iterable[Zone]) -> None:
        self.zones = list(zones)
        self.has_room_boundary = any(z.name == self.ROOM for z in self.zones)

    def assign(self, positions: dict[int, tuple[float, float]]) -> dict[str, list[int]]:
        """Map each zone name to the track IDs standing in it."""
        return {
            z.name: sorted(tid for tid, (x, y) in positions.items() if point_in_polygon(x, y, z.polygon))
            for z in self.zones
        }

    def people_in_room(self, positions: dict[int, tuple[float, float]]) -> list[int]:
        if not self.has_room_boundary:
            return sorted(positions)
        room = next(z for z in self.zones if z.name == self.ROOM)
        return sorted(tid for tid, (x, y) in positions.items() if point_in_polygon(x, y, room.polygon))
