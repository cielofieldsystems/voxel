"""End-to-end voxel backprojection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from dataset import FrameRecord, group_by_camera, load_dataset
from geometry import ray_from_pixel
from motion import compute_motion, read_gray
from voxel_grid import VoxelGrid, create_voxel_grid, cast_ray_dda, save_voxel_grid, save_voxel_meta, top_voxels


def load_dataset_meta(dataset_root: Path) -> dict:
    meta_path = dataset_root / "dataset.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def accumulate_frames(
    frames: List[FrameRecord],
    vox: VoxelGrid,
    threshold: float,
    blur_ksize: int,
    alpha: float,
    max_motion_pixels: int,
) -> None:
    prev_img = None

    for frame in frames:
        curr_img = read_gray(str(frame.image_path))
        if prev_img is None:
            prev_img = curr_img
            continue

        coords, weights = compute_motion(prev_img, curr_img, threshold, blur_ksize)
        if coords.size == 0:
            prev_img = curr_img
            continue

        if max_motion_pixels and coords.shape[0] > max_motion_pixels:
            indices = np.random.choice(coords.shape[0], max_motion_pixels, replace=False)
            coords = coords[indices]
            weights = weights[indices]

        for (v, u), w in zip(coords, weights):
            origin, direction = ray_from_pixel(
                float(u),
                float(v),
                frame.camera_position,
                frame.camera_to_world,
                frame.intrinsics,
            )
            cast_ray_dda(vox, origin, direction, float(w), alpha)

        prev_img = curr_img


def run_pipeline(
    frames: List[FrameRecord],
    grid_center: np.ndarray,
    grid_extent: float,
    grid_size: int,
    threshold: float,
    blur_ksize: int,
    alpha: float,
    max_motion_pixels: int,
) -> VoxelGrid:
    bmin = grid_center - 0.5 * grid_extent
    bmax = grid_center + 0.5 * grid_extent
    vox = create_voxel_grid((grid_size, grid_size, grid_size), bmin, bmax)

    grouped = group_by_camera(frames)
    for cam_frames in grouped.values():
        accumulate_frames(cam_frames, vox, threshold, blur_ksize, alpha, max_motion_pixels)

    return vox


def refine_bounds(top: List[Tuple[np.ndarray, float]], pad: float) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.array([c for c, _ in top], dtype=np.float64)
    bmin = centers.min(axis=0) - pad
    bmax = centers.max(axis=0) + pad
    return bmin, bmax


def write_top_voxels(path: Path, top: List[Tuple[np.ndarray, float]]) -> None:
    payload = [
        {
            "center": [float(v) for v in center],
            "score": float(score),
        }
        for center, score in top
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxel backprojection pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset root with metadata.json")
    parser.add_argument("--metadata", default=None, help="Path to metadata.json")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--grid-center", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--grid-extent", type=float, default=4.0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--max-motion-pixels", type=int, default=20000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine-size", type=int, default=128)
    parser.add_argument("--refine-pad", type=float, default=0.5)
    parser.add_argument("--units", default="grid_units")
    parser.add_argument("--units-per-grid-unit", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    metadata_path = Path(args.metadata) if args.metadata else dataset_root / "metadata.json"

    dataset_meta = load_dataset_meta(dataset_root)
    units = dataset_meta.get("units", args.units)
    units_per_grid_unit = float(dataset_meta.get("units_per_grid_unit", args.units_per_grid_unit))

    frames = load_dataset(metadata_path)
    grid_center = np.array(args.grid_center, dtype=np.float64)

    vox = run_pipeline(
        frames,
        grid_center,
        args.grid_extent,
        args.grid_size,
        args.threshold,
        args.blur,
        args.alpha,
        args.max_motion_pixels,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_voxel_grid(out_dir / "voxel_grid.bin", vox)
    save_voxel_meta(out_dir / "voxel_grid.json", vox, units=units, units_per_grid_unit=units_per_grid_unit)

    top = top_voxels(vox, args.top_k)
    write_top_voxels(out_dir / "top_voxels.json", top)

    if args.refine:
        bmin, bmax = refine_bounds(top, args.refine_pad)
        vox_refined = create_voxel_grid((args.refine_size,) * 3, bmin, bmax)
        for cam_frames in group_by_camera(frames).values():
            accumulate_frames(
                cam_frames,
                vox_refined,
                args.threshold,
                args.blur,
                args.alpha,
                args.max_motion_pixels,
            )
        save_voxel_grid(out_dir / "voxel_grid_refined.bin", vox_refined)
        save_voxel_meta(
            out_dir / "voxel_grid_refined.json",
            vox_refined,
            units=units,
            units_per_grid_unit=units_per_grid_unit,
        )


if __name__ == "__main__":
    main()
