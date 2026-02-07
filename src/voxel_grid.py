"""Voxel grid accumulation and IO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VoxelGrid:
    grid: np.ndarray
    bmin: np.ndarray
    bmax: np.ndarray
    voxel_size: np.ndarray

    @property
    def dims(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.grid.shape)


def create_voxel_grid(dims: Tuple[int, int, int], bmin: np.ndarray, bmax: np.ndarray) -> VoxelGrid:
    grid = np.zeros(dims, dtype=np.float32)
    voxel_size = (bmax - bmin) / np.array(dims, dtype=np.float64)
    return VoxelGrid(grid=grid, bmin=bmin, bmax=bmax, voxel_size=voxel_size)


def ray_aabb_intersection(
    origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> Optional[Tuple[float, float]]:
    tmin = -float("inf")
    tmax = float("inf")

    for i in range(3):
        d = float(direction[i])
        if abs(d) < 1e-12:
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return None
            continue

        t1 = (bmin[i] - origin[i]) / d
        t2 = (bmax[i] - origin[i]) / d
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        tmin = max(tmin, t_near)
        tmax = min(tmax, t_far)
        if tmin > tmax:
            return None

    if tmax < 0.0:
        return None
    return max(tmin, 0.0), tmax


def cast_ray_dda(
    vox: VoxelGrid,
    origin: np.ndarray,
    direction: np.ndarray,
    weight: float,
    alpha: float = 0.0,
) -> None:
    hit = ray_aabb_intersection(origin, direction, vox.bmin, vox.bmax)
    if hit is None:
        return

    t0, t1 = hit
    p = origin + t0 * direction
    idx = np.floor((p - vox.bmin) / vox.voxel_size).astype(int)

    if np.any(idx < 0) or np.any(idx >= np.array(vox.dims)):
        return

    step = np.sign(direction).astype(int)
    step[step == 0] = 1

    next_boundary = vox.bmin + (idx + (step > 0).astype(int)) * vox.voxel_size
    with np.errstate(divide="ignore", invalid="ignore"):
        t_max = (next_boundary - origin) / direction
        t_delta = vox.voxel_size / np.abs(direction)

    t_max = np.where(np.isfinite(t_max), t_max, float("inf"))
    t_delta = np.where(np.isfinite(t_delta), t_delta, float("inf"))

    t_current = t0

    while t_current <= t1:
        vox.grid[idx[0], idx[1], idx[2]] += weight / (1.0 + alpha * t_current)

        axis = int(np.argmin(t_max))
        t_current = float(t_max[axis])
        idx[axis] += step[axis]
        if idx[axis] < 0 or idx[axis] >= vox.dims[axis]:
            break
        t_max[axis] += t_delta[axis]


def save_voxel_grid(bin_path: Path, vox: VoxelGrid) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as handle:
        handle.write(vox.grid.astype(np.float32).tobytes(order="C"))


def save_voxel_meta(
    json_path: Path,
    vox: VoxelGrid,
    units: str | None = None,
    units_per_grid_unit: float | None = None,
) -> None:
    meta = {
        "dims": list(vox.dims),
        "bmin": [float(v) for v in vox.bmin],
        "bmax": [float(v) for v in vox.bmax],
        "voxel_size": [float(v) for v in vox.voxel_size],
        "order": "xyz",
    }
    if units is not None:
        meta["units"] = units
    if units_per_grid_unit is not None:
        meta["units_per_grid_unit"] = float(units_per_grid_unit)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_voxel_grid(bin_path: Path, meta_path: Path) -> VoxelGrid:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = tuple(meta["dims"])
    bmin = np.array(meta["bmin"], dtype=np.float64)
    bmax = np.array(meta["bmax"], dtype=np.float64)
    voxel_size = np.array(meta["voxel_size"], dtype=np.float64)

    raw = bin_path.read_bytes()
    data = np.frombuffer(raw, dtype=np.float32)
    grid = data.reshape(dims)
    return VoxelGrid(grid=grid, bmin=bmin, bmax=bmax, voxel_size=voxel_size)


def top_voxels(vox: VoxelGrid, count: int) -> List[Tuple[np.ndarray, float]]:
    flat = vox.grid.ravel()
    if count >= flat.size:
        indices = np.argsort(flat)[::-1]
    else:
        indices = np.argpartition(flat, -count)[-count:]
        indices = indices[np.argsort(flat[indices])[::-1]]

    results = []
    for idx in indices[:count]:
        v = np.unravel_index(int(idx), vox.grid.shape)
        center = vox.bmin + (np.array(v, dtype=np.float64) + 0.5) * vox.voxel_size
        results.append((center, float(flat[idx])))
    return results
