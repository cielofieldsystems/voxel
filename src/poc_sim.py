"""Toy simulation of voxel backprojection using synthetic motion rays."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

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


def intersect_aabb(
    origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> Optional[Tuple[float, float]]:
    # Avoid divide-by-zero in slab test.
    safe_dir = np.where(np.abs(direction) < 1e-8, 1e-8, direction)
    inv = 1.0 / safe_dir
    t0s = (bmin - origin) * inv
    t1s = (bmax - origin) * inv
    tmin = float(np.max(np.minimum(t0s, t1s)))
    tmax = float(np.min(np.maximum(t0s, t1s)))
    if tmax < max(tmin, 0.0):
        return None
    return max(tmin, 0.0), tmax


def backproject_ray(
    accum: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
    step: float,
) -> None:
    hit = intersect_aabb(origin, direction, bmin, bmax)
    if hit is None:
        return
    t0, t1 = hit
    grid_size = np.array(accum.shape, dtype=np.float64)
    grid_scale = grid_size / (bmax - bmin)

    steps = int((t1 - t0) / step) + 1
    for i in range(steps):
        t = t0 + i * step
        p = origin + t * direction
        idx = ((p - bmin) * grid_scale).astype(int)
        if np.any(idx < 0) or np.any(idx >= grid_size):
            continue
        accum[tuple(idx)] += 1.0


def generate_cameras(
    rng: np.random.Generator, num_cams: int, radius: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cams = []
    for _ in range(num_cams):
        direction = normalize(rng.normal(size=3))
        eye = direction * radius
        r = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        cams.append((eye, r))
    return cams


def simulate_motion_points(num_frames: int) -> List[np.ndarray]:
    start = np.array([-0.6, -0.2, -0.1])
    end = np.array([0.6, 0.25, 0.3])
    positions = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        positions.append(start + t * (end - start))
    return positions


def top_voxels(
    accum: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, count: int
) -> List[Tuple[np.ndarray, float]]:
    flat = accum.ravel()
    if count >= flat.size:
        indices = np.argsort(flat)[::-1]
    else:
        indices = np.argpartition(flat, -count)[-count:]
        indices = indices[np.argsort(flat[indices])[::-1]]

    grid_size = np.array(accum.shape, dtype=np.float64)
    voxel_size = (bmax - bmin) / grid_size
    results = []
    for idx in indices[:count]:
        v = np.unravel_index(int(idx), accum.shape)
        center = bmin + (np.array(v) + 0.5) * voxel_size
        results.append((center, float(flat[idx])))
    return results


def main() -> None:
    rng = np.random.default_rng(7)

    grid_size = 64
    bmin = np.array([-1.5, -1.5, -1.5])
    bmax = np.array([1.5, 1.5, 1.5])
    accum = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    width, height = 640, 360
    fov_deg = 70.0
    fx = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    intr = {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fx,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }

    num_cams = 12
    cam_radius = 3.0
    cameras = generate_cameras(rng, num_cams, cam_radius)

    num_frames = 6
    target_positions = simulate_motion_points(num_frames)

    step = 0.05
    noise_rays = 8

    for eye, r in cameras:
        for t in range(num_frames - 1):
            p0 = project_point(target_positions[t], eye, r, intr)
            p1 = project_point(target_positions[t + 1], eye, r, intr)
            if p0 is None or p1 is None:
                continue

            # Use two motion pixels per frame pair (start and end).
            for u, v in (p0, p1):
                ray_origin, ray_dir = ray_from_pixel(float(u), float(v), eye, r, intr)
                backproject_ray(accum, ray_origin, ray_dir, bmin, bmax, step)

            # Add random motion rays to simulate noise.
            for _ in range(noise_rays):
                u = rng.uniform(0, width)
                v = rng.uniform(0, height)
                ray_origin, ray_dir = ray_from_pixel(float(u), float(v), eye, r, intr)
                backproject_ray(accum, ray_origin, ray_dir, bmin, bmax, step)

    best = top_voxels(accum, bmin, bmax, count=5)
    best_center = best[0][0]
    dists = [float(np.linalg.norm(best_center - p)) for p in target_positions]

    print("Top voxels (center xyz, score):")
    for center, score in best:
        center_str = ", ".join(f"{c: .3f}" for c in center)
        print(f"  [{center_str}] -> {score:.1f}")

    print()
    print(f"Best voxel center: {best_center}")
    print(f"Min distance to target path: {min(dists):.3f}")


if __name__ == "__main__":
    main()
