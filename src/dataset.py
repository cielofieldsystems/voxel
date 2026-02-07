"""Dataset metadata loading and helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from geometry import rotation_matrix_yaw_pitch_roll


@dataclass
class FrameRecord:
    camera_id: str
    frame_index: int
    timestamp: float
    image_path: Path
    intrinsics: dict
    camera_position: np.ndarray
    camera_to_world: np.ndarray


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_intrinsics(entry: dict, base_dir: Path) -> dict:
    if "intrinsics" in entry and entry["intrinsics"] is not None:
        return entry["intrinsics"]
    if "intrinsics_file" in entry and entry["intrinsics_file"]:
        intr = _load_json(_resolve_path(base_dir, entry["intrinsics_file"]))
        return intr
    raise ValueError("Missing intrinsics for entry")


def _load_extrinsics(entry: dict, base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    if "extrinsics" in entry and entry["extrinsics"] is not None:
        extr = entry["extrinsics"]
        cam_pos = np.array(extr["camera_position"], dtype=np.float64)
        cam_to_world = np.array(extr["camera_to_world"], dtype=np.float64)
        return cam_pos, cam_to_world
    if "extrinsics_file" in entry and entry["extrinsics_file"]:
        extr = _load_json(_resolve_path(base_dir, entry["extrinsics_file"]))
        cam_pos = np.array(extr["camera_position"], dtype=np.float64)
        cam_to_world = np.array(extr["camera_to_world"], dtype=np.float64)
        return cam_pos, cam_to_world

    if "camera_position" in entry and all(k in entry for k in ("yaw", "pitch", "roll")):
        cam_pos = np.array(entry["camera_position"], dtype=np.float64)
        cam_to_world = rotation_matrix_yaw_pitch_roll(
            float(entry["yaw"]), float(entry["pitch"]), float(entry["roll"])
        )
        return cam_pos, cam_to_world

    raise ValueError("Missing extrinsics for entry")


def load_dataset(metadata_path: Path) -> List[FrameRecord]:
    metadata_path = metadata_path.resolve()
    base_dir = metadata_path.parent
    data = _load_json(metadata_path)
    if not isinstance(data, list):
        raise ValueError("metadata.json must be a list of frame entries")

    frames: List[FrameRecord] = []
    for entry in data:
        intr = _load_intrinsics(entry, base_dir)
        cam_pos, cam_to_world = _load_extrinsics(entry, base_dir)
        image_path = _resolve_path(base_dir, entry["image_file"])

        frames.append(
            FrameRecord(
                camera_id=str(entry.get("camera_id", entry.get("camera_index", "cam_000"))),
                frame_index=int(entry.get("frame_index", 0)),
                timestamp=float(entry.get("timestamp", entry.get("frame_index", 0))),
                image_path=image_path,
                intrinsics=intr,
                camera_position=cam_pos,
                camera_to_world=cam_to_world,
            )
        )
    return frames


def group_by_camera(frames: Iterable[FrameRecord]) -> Dict[str, List[FrameRecord]]:
    grouped: Dict[str, List[FrameRecord]] = {}
    for frame in frames:
        grouped.setdefault(frame.camera_id, []).append(frame)

    for cam_id in grouped:
        grouped[cam_id].sort(key=lambda f: f.frame_index)
    return grouped
