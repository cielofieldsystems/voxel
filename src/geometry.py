"""Geometry helpers for camera projection and rays."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    if abs(float(np.dot(forward, up))) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, forward], axis=1)


def rotation_matrix_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Match the reference order: Rz(yaw) * Ry(roll) * Rx(pitch)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)

    r_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_y = np.array(
        [
            [cr, 0.0, sr],
            [0.0, 1.0, 0.0],
            [-sr, 0.0, cr],
        ],
        dtype=np.float64,
    )
    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )

    return r_z @ r_y @ r_x


def project_point(
    point_w: np.ndarray, eye: np.ndarray, r_cam_to_world: np.ndarray, intr: dict
) -> Optional[np.ndarray]:
    point_cam = r_cam_to_world.T @ (point_w - eye)
    if point_cam[2] <= 1e-6:
        return None
    u = intr["fx"] * (point_cam[0] / point_cam[2]) + intr["cx"]
    v = intr["fy"] * (point_cam[1] / point_cam[2]) + intr["cy"]
    if u < 0 or u >= intr["width"] or v < 0 or v >= intr["height"]:
        return None
    return np.array([u, v])


def ray_from_pixel(
    u: float, v: float, eye: np.ndarray, r_cam_to_world: np.ndarray, intr: dict
) -> Tuple[np.ndarray, np.ndarray]:
    x = (u - intr["cx"]) / intr["fx"]
    y = (v - intr["cy"]) / intr["fy"]
    dir_cam = normalize(np.array([x, y, 1.0]))
    dir_world = r_cam_to_world @ dir_cam
    return eye, dir_world
