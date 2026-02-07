"""Live voxel backprojection pipeline using streaming cameras."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from geometry import ray_from_pixel
from motion import compute_motion
from voxel_grid import (
    VoxelGrid,
    cast_ray_dda,
    create_voxel_grid,
    save_voxel_grid,
    save_voxel_meta,
    top_voxels,
)


@dataclass
class CameraConfig:
    camera_id: str
    source: Union[int, str]
    intrinsics: dict
    camera_position: np.ndarray
    camera_to_world: np.ndarray


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_intrinsics(intr_dir: Path, camera_id: str) -> dict:
    path = intr_dir / f"{camera_id}_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {path}")
    return _load_json(path)


def _load_extrinsics(extr_dir: Path, camera_id: str) -> Tuple[np.ndarray, np.ndarray]:
    path = extr_dir / f"{camera_id}_extrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing extrinsics file: {path}")
    extr = _load_json(path)
    cam_pos = np.array(extr["camera_position"], dtype=np.float64)
    cam_to_world = np.array(extr["camera_to_world"], dtype=np.float64)
    return cam_pos, cam_to_world


def _parse_source(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def _scale_intrinsics(intr: dict, width: int, height: int) -> dict:
    if intr["width"] == width and intr["height"] == height:
        return intr

    scale_x = width / float(intr["width"])
    scale_y = height / float(intr["height"])
    return {
        **intr,
        "fx": float(intr["fx"]) * scale_x,
        "fy": float(intr["fy"]) * scale_y,
        "cx": float(intr["cx"]) * scale_x,
        "cy": float(intr["cy"]) * scale_y,
        "width": int(width),
        "height": int(height),
    }


def _undistort_gray(gray: np.ndarray, intr: dict) -> np.ndarray:
    dist = np.array(intr.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
    if not np.any(dist):
        return gray
    mtx = np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return cv2.undistort(gray, mtx, dist)


def _init_cameras(args: argparse.Namespace) -> List[CameraConfig]:
    sources = [_parse_source(v) for v in args.sources]
    if args.camera_ids:
        if len(args.camera_ids) != len(sources):
            raise ValueError("camera_ids must match number of sources")
        camera_ids = args.camera_ids
    else:
        camera_ids = [f"cam_{i:03d}" for i in range(len(sources))]

    intr_dir = Path(args.intrinsics_dir)
    extr_dir = Path(args.extrinsics_dir)

    configs = []
    for camera_id, source in zip(camera_ids, sources):
        intr = _load_intrinsics(intr_dir, camera_id)
        cam_pos, cam_to_world = _load_extrinsics(extr_dir, camera_id)
        configs.append(CameraConfig(camera_id, source, intr, cam_pos, cam_to_world))
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live voxel backprojection pipeline")
    parser.add_argument("--sources", nargs="+", required=True, help="Camera indices or URLs")
    parser.add_argument("--camera-ids", nargs="+", default=None, help="Optional camera IDs")
    parser.add_argument("--intrinsics-dir", required=True, help="Folder with intrinsics JSON files")
    parser.add_argument("--extrinsics-dir", required=True, help="Folder with extrinsics JSON files")
    parser.add_argument("--grid-center", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--grid-extent", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-motion-pixels", type=int, default=20000)
    parser.add_argument("--decay", type=float, default=0.98)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None, help="Output directory for snapshots")
    parser.add_argument("--save-every", type=int, default=0, help="Save every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
    parser.add_argument("--undistort", action="store_true", help="Undistort frames using intrinsics")
    parser.add_argument("--width", type=int, default=0, help="Optional capture width")
    parser.add_argument("--height", type=int, default=0, help="Optional capture height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = _init_cameras(args)

    caps = []
    for cfg in cameras:
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {cfg.source}")
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        caps.append(cap)

    grid_center = np.array(args.grid_center, dtype=np.float64)
    bmin = grid_center - 0.5 * args.grid_extent
    bmax = grid_center + 0.5 * args.grid_extent
    vox = create_voxel_grid((args.grid_size,) * 3, bmin, bmax)

    prev_frames: List[Optional[np.ndarray]] = [None] * len(cameras)
    rng = np.random.default_rng(7)

    frame_count = 0
    start = time.perf_counter()

    try:
        while True:
            for idx, (cfg, cap) in enumerate(zip(cameras, caps)):
                ok, frame = cap.read()
                if not ok:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                cfg.intrinsics = _scale_intrinsics(cfg.intrinsics, gray.shape[1], gray.shape[0])
                if args.undistort:
                    gray = _undistort_gray(gray, cfg.intrinsics)

                if prev_frames[idx] is None:
                    prev_frames[idx] = gray
                    continue

                coords, weights = compute_motion(prev_frames[idx], gray, args.threshold, args.blur)
                prev_frames[idx] = gray
                if coords.size == 0:
                    continue

                if args.max_motion_pixels and coords.shape[0] > args.max_motion_pixels:
                    sample = rng.choice(coords.shape[0], args.max_motion_pixels, replace=False)
                    coords = coords[sample]
                    weights = weights[sample]

                for (v, u), w in zip(coords, weights):
                    origin, direction = ray_from_pixel(
                        float(u),
                        float(v),
                        cfg.camera_position,
                        cfg.camera_to_world,
                        cfg.intrinsics,
                    )
                    cast_ray_dda(vox, origin, direction, float(w), args.alpha)

            frame_count += 1
            if 0.0 < args.decay < 1.0:
                vox.grid *= args.decay

            if args.print_every and frame_count % args.print_every == 0:
                elapsed = time.perf_counter() - start
                fps = frame_count / max(elapsed, 1e-6)
                top = top_voxels(vox, args.top_k)
                print(f"frames={frame_count} fps={fps:.1f}")
                for center, score in top:
                    center_str = ", ".join(f"{c: .3f}" for c in center)
                    print(f"  [{center_str}] -> {score:.1f}")

            if args.output and args.save_every and frame_count % args.save_every == 0:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                save_voxel_grid(out_dir / "voxel_grid.bin", vox)
                save_voxel_meta(out_dir / "voxel_grid.json", vox)

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        for cap in caps:
            cap.release()


if __name__ == "__main__":
    main()
