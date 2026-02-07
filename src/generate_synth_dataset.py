"""Generate a small synthetic multi-camera dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from geometry import look_at, project_point


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def draw_spot(
    img: np.ndarray, center: np.ndarray, intensity: float = 255.0, sigma: float = 1.2
) -> None:
    u = float(center[0])
    v = float(center[1])

    if sigma <= 0:
        ui, vi = int(round(u)), int(round(v))
        if ui < 1 or vi < 1 or ui >= img.shape[1] - 1 or vi >= img.shape[0] - 1:
            return
        img[vi, ui] = intensity
        img[vi - 1, ui] = intensity * 0.6
        img[vi + 1, ui] = intensity * 0.6
        img[vi, ui - 1] = intensity * 0.6
        img[vi, ui + 1] = intensity * 0.6
        return

    radius = max(1, int(math.ceil(3 * sigma)))
    x0 = max(0, int(math.floor(u)) - radius)
    x1 = min(img.shape[1] - 1, int(math.floor(u)) + radius)
    y0 = max(0, int(math.floor(v)) - radius)
    y1 = min(img.shape[0] - 1, int(math.floor(v)) + radius)

    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = xx - u
    dy = yy - v
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    img[y0 : y1 + 1, x0 : x1 + 1] += intensity * weights


def generate_cameras(rng: np.random.Generator, num_cams: int, radius: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    cams = []
    for _ in range(num_cams):
        direction = normalize(rng.normal(size=3))
        eye = direction * radius
        r = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        cams.append((eye, r))
    return cams


def simulate_motion_points(
    num_frames: int,
    start: np.ndarray,
    end: np.ndarray,
    mode: str,
) -> List[np.ndarray]:
    if num_frames < 2:
        return [start.copy()]

    if mode == "spline":
        # Cubic Bezier path, re-sampled by arc length for near-constant speed.
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return [start.copy() for _ in range(num_frames)]

        # Build a stable lateral basis for a 3D curve.
        up = np.array([0.0, 1.0, 0.0])
        d_hat = direction / length
        if abs(float(np.dot(d_hat, up))) > 0.95:
            up = np.array([0.0, 0.0, 1.0])
        side = normalize(np.cross(direction, up))
        lift = normalize(np.cross(side, direction))

        amp = 0.25 * length
        c1 = start + 0.33 * direction + 0.60 * amp * side + 0.20 * amp * lift
        c2 = start + 0.66 * direction - 0.40 * amp * side + 0.35 * amp * lift

        samples_n = 2000
        ts = np.linspace(0.0, 1.0, samples_n, dtype=np.float64)
        omt = 1.0 - ts
        pts = (
            (omt**3)[:, None] * start[None, :]
            + 3.0 * (omt**2)[:, None] * ts[:, None] * c1[None, :]
            + 3.0 * omt[:, None] * (ts**2)[:, None] * c2[None, :]
            + (ts**3)[:, None] * end[None, :]
        )
        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(cum[-1])
        if total < 1e-9:
            return [start.copy() for _ in range(num_frames)]

        positions = []
        for i in range(num_frames):
            t = i / (num_frames - 1)
            target = t * total
            j = int(np.searchsorted(cum, target, side="right") - 1)
            j = max(0, min(j, samples_n - 2))
            d0 = float(cum[j])
            d1 = float(cum[j + 1])
            if d1 <= d0 + 1e-12:
                positions.append(pts[j].copy())
            else:
                w = (target - d0) / (d1 - d0)
                positions.append((1.0 - w) * pts[j] + w * pts[j + 1])
        return positions

    positions = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        if mode == "arc":
            wobble = np.array(
                [
                    0.0,
                    0.25 * np.sin(2.0 * np.pi * t),
                    0.2 * np.cos(2.0 * np.pi * t),
                ]
            )
            positions.append((1 - t) * start + t * end + wobble)
        else:
            positions.append(start + t * (end - start))
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset")
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("--num-cams", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--noise", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--path-scale", type=float, default=1.0)
    parser.add_argument("--spot-sigma", type=float, default=1.2)
    parser.add_argument("--path-mode", type=str, default="arc", choices=["line", "arc", "spline"])
    parser.add_argument("--units", type=str, default="meters")
    parser.add_argument("--units-per-grid-unit", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    fx = 0.5 * args.width / np.tan(np.deg2rad(args.fov) / 2)
    intr = {
        "width": args.width,
        "height": args.height,
        "fx": float(fx),
        "fy": float(fx),
        "cx": args.width / 2.0,
        "cy": args.height / 2.0,
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    cams = generate_cameras(rng, args.num_cams, args.radius)
    base_start = np.array([-0.6, -0.2, -0.1])
    base_end = np.array([0.6, 0.25, 0.3])
    targets = simulate_motion_points(
        args.num_frames,
        base_start * args.path_scale,
        base_end * args.path_scale,
        args.path_mode,
    )

    metadata = []

    for cam_idx, (eye, r) in enumerate(cams):
        cam_dir = frames_root / f"cam_{cam_idx:03d}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(args.num_frames):
            img = rng.normal(12.0, args.noise, size=(args.height, args.width)).astype(np.float32)
            img = np.clip(img, 0.0, 255.0)

            target = targets[frame_idx]
            uv = project_point(target, eye, r, intr)
            if uv is not None:
                draw_spot(img, uv, sigma=args.spot_sigma)

            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            filename = f"frame_{frame_idx:06d}.png"
            out_path = cam_dir / filename
            cv2.imwrite(str(out_path), img_u8)

            metadata.append(
                {
                    "camera_id": f"cam_{cam_idx:03d}",
                    "frame_index": frame_idx,
                    "timestamp": frame_idx / args.fps,
                    "image_file": str(out_path.relative_to(out_dir)),
                    "intrinsics": intr,
                    "extrinsics": {
                        "camera_position": [float(v) for v in eye],
                        "camera_to_world": r.tolist(),
                    },
                }
            )

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dataset_meta = {
        "units": args.units,
        "units_per_grid_unit": float(args.units_per_grid_unit),
        "description": "Synthetic multi-camera dataset",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
