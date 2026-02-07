"""Render a GIF of voxel peak motion and estimate speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from geometry import ray_from_pixel  # noqa: E402
from motion import compute_motion, read_gray  # noqa: E402
from voxel_grid import create_voxel_grid, cast_ray_dda, top_voxels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render voxel motion GIF")
    parser.add_argument("--dataset", required=True, help="Dataset root (with metadata.json)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--duration", type=int, default=120, help="Frame duration in ms")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--max-motion-pixels", type=int, default=20000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-first", type=int, default=0, help="Skip the first N frame indices")
    parser.add_argument("--top-k", type=int, default=5, help="Blend top-k voxels for centroid")
    parser.add_argument("--candidate-k", type=int, default=40, help="Top-K voxel candidates before tracking gate")
    parser.add_argument("--track-gate", type=float, default=0.0, help="Max jump distance (grid units) before gating")
    parser.add_argument("--smooth-window", type=int, default=0, help="Trailing average window size")
    parser.add_argument(
        "--integration-window",
        type=int,
        default=1,
        help="Accumulate over the last N frame-differences (1 = per-step)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["top", "random"],
        default="top",
        help="How to downsample motion pixels when capped",
    )
    parser.add_argument("--units", default=None, help="Override units label")
    parser.add_argument("--units-per-grid-unit", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    output_dir = (
        Path(args.output)
        if args.output
        else project_root / "output" / dataset_root.name
    )
    frames_dir = output_dir / "voxel_gif_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_root / "metadata.json"
    dataset_meta_path = dataset_root / "dataset.json"
    voxel_meta_path = output_dir / "voxel_grid.json"

    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}")
    if not voxel_meta_path.exists():
        raise SystemExit(f"Missing voxel meta: {voxel_meta_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    voxel_meta = json.loads(voxel_meta_path.read_text(encoding="utf-8"))
    dataset_meta = (
        json.loads(dataset_meta_path.read_text(encoding="utf-8"))
        if dataset_meta_path.exists()
        else {}
    )

    units = (
        args.units
        if args.units is not None
        else dataset_meta.get("units", voxel_meta.get("units", "grid_units"))
    )
    units_per_grid_unit = (
        float(args.units_per_grid_unit)
        if args.units_per_grid_unit is not None
        else float(dataset_meta.get("units_per_grid_unit", voxel_meta.get("units_per_grid_unit", 1.0)))
    )

    bmin = np.array(voxel_meta["bmin"], dtype=np.float64)
    bmax = np.array(voxel_meta["bmax"], dtype=np.float64)
    dims = tuple(voxel_meta["dims"])

    frames_by_cam: dict[str, dict[int, dict]] = {}
    frame_times: dict[int, list[float]] = {}

    for entry in metadata:
        cam_id = entry.get("camera_id", entry.get("camera_index", "cam_000"))
        frame_idx = int(entry.get("frame_index", 0))
        frames_by_cam.setdefault(cam_id, {})[frame_idx] = entry
        frame_times.setdefault(frame_idx, []).append(float(entry.get("timestamp", frame_idx)))

    frame_indices = sorted(frame_times.keys())

    frame_time_avg = {k: float(np.mean(v)) for k, v in frame_times.items()}

    camera_positions: list[np.ndarray] = []
    camera_forwards: list[np.ndarray] = []
    for cam_id, frames in frames_by_cam.items():
        if not frames:
            continue
        entry = frames[min(frames.keys())]
        extr = entry["extrinsics"]
        cam_pos = np.array(extr["camera_position"], dtype=np.float64)
        cam_to_world = np.array(extr["camera_to_world"], dtype=np.float64)
        forward = cam_to_world[:, 2]
        camera_positions.append(cam_pos)
        camera_forwards.append(forward)

    positions: list[np.ndarray] = []
    scores: list[float] = []
    pos_frame_indices: list[int] = []

    rng = np.random.default_rng(7)
    prev_center = None
    integration_window = max(1, int(args.integration_window))
    work_frame_indices = frame_indices[1:]
    if args.skip_first:
        work_frame_indices = work_frame_indices[int(args.skip_first) :]
    if args.max_frames and len(work_frame_indices) > args.max_frames:
        work_frame_indices = work_frame_indices[: args.max_frames]

    for idx in work_frame_indices:
        vox = create_voxel_grid(dims, bmin, bmax)
        for cam_id, frames in frames_by_cam.items():
            # Optionally integrate multiple diffs per output frame to stabilize estimates.
            start_idx = max(1, idx - integration_window + 1)
            for diff_idx in range(start_idx, idx + 1):
                if diff_idx - 1 not in frames or diff_idx not in frames:
                    continue
                f0 = frames[diff_idx - 1]
                f1 = frames[diff_idx]

                img0 = read_gray(str(dataset_root / f0["image_file"]))
                img1 = read_gray(str(dataset_root / f1["image_file"]))

                coords, weights = compute_motion(img0, img1, args.threshold, args.blur)
                if coords.size == 0:
                    continue
                if args.max_motion_pixels and coords.shape[0] > args.max_motion_pixels:
                    if args.sample_mode == "random":
                        sample = rng.choice(coords.shape[0], args.max_motion_pixels, replace=False)
                    else:
                        sample = np.argpartition(weights, -args.max_motion_pixels)[-args.max_motion_pixels :]
                    coords = coords[sample]
                    weights = weights[sample]

                intr = f1["intrinsics"]
                extr = f1["extrinsics"]
                cam_pos = np.array(extr["camera_position"], dtype=np.float64)
                cam_to_world = np.array(extr["camera_to_world"], dtype=np.float64)

                for (v, u), w in zip(coords, weights):
                    origin, direction = ray_from_pixel(float(u), float(v), cam_pos, cam_to_world, intr)
                    cast_ray_dda(vox, origin, direction, float(w), alpha=0.0)

        candidates = top_voxels(vox, max(1, max(args.top_k, args.candidate_k)))
        selected = candidates[: max(1, args.top_k)]

        # If multiple peaks exist, prefer continuity (simple gating).
        if prev_center is not None and args.track_gate and args.track_gate > 0.0:
            within = [
                (c, s) for (c, s) in candidates if float(np.linalg.norm(c - prev_center)) <= args.track_gate
            ]
            if within:
                selected = within[: max(1, args.top_k)]
            else:
                # No plausible peak near the previous estimate; hold position rather than jumping.
                positions.append(prev_center)
                scores.append(0.0)
                pos_frame_indices.append(int(idx))
                continue

        centers = np.array([c for c, _ in selected], dtype=np.float64)
        weights = np.array([s for _, s in selected], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            center = (centers * weights[:, None]).sum(axis=0) / weight_sum
        elif prev_center is not None:
            # No votes this frame; hold the previous estimate instead of snapping to an arbitrary voxel.
            center = prev_center
        else:
            # No estimate yet and no votes; skip this frame.
            continue

        positions.append(center)
        scores.append(weight_sum)
        pos_frame_indices.append(int(idx))
        prev_center = center

    if args.smooth_window > 0 and positions:
        smoothed = []
        for i in range(len(positions)):
            start = max(0, i - args.smooth_window)
            window = np.array(positions[start : i + 1], dtype=np.float64)
            smoothed.append(window.mean(axis=0))
        positions = smoothed

    speeds = [None]
    for i in range(1, len(positions)):
        dt = frame_time_avg[pos_frame_indices[i]] - frame_time_avg[pos_frame_indices[i - 1]]
        dt = max(dt, 1e-6)
        dist = float(np.linalg.norm(positions[i] - positions[i - 1]))
        speeds.append(dist * units_per_grid_unit / dt)

    speed_payload = {
        "frame_indices": pos_frame_indices,
        "positions": [[float(v) for v in p] for p in positions],
        "speeds": speeds,
        "units": f"{units}/s",
        "units_per_grid_unit": units_per_grid_unit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "speed_estimates.json").write_text(
        json.dumps(speed_payload, indent=2), encoding="utf-8"
    )

    images = []
    speed_units = f"{units}/s"
    if units.lower() in {"meter", "meters", "metre", "metres"}:
        speed_units = "m/s"
    elif units.lower() in {"foot", "feet"}:
        speed_units = "ft/s"
    for i, center in enumerate(positions):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

        if camera_positions:
            cams = np.array(camera_positions, dtype=np.float64)
            fwds = np.array(camera_forwards, dtype=np.float64)

            ax.scatter(
                cams[:, 0],
                cams[:, 1],
                cams[:, 2],
                c="tab:blue",
                s=55,
                marker="s",
                label="Cameras (arrow = view dir)",
            )
            arrow_len = 0.6 * float(np.max(bmax - bmin)) / 4.0
            ax.quiver(
                cams[:, 0],
                cams[:, 1],
                cams[:, 2],
                fwds[:, 0],
                fwds[:, 1],
                fwds[:, 2],
                length=arrow_len,
                normalize=True,
                color="tab:blue",
                linewidth=1.2,
                arrow_length_ratio=0.25,
            )

        trail = np.array(positions[: i + 1])
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="tab:red", alpha=0.65, label="Track")
        ax.scatter(
            [trail[0, 0]],
            [trail[0, 1]],
            [trail[0, 2]],
            c="tab:green",
            s=45,
            marker="o",
            label="Start",
        )
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            c="red",
            s=80,
            marker="*",
            label="Current",
        )

        ax.set_xlim(bmin[0], bmax[0])
        ax.set_ylim(bmin[1], bmax[1])
        ax.set_zlim(bmin[2], bmax[2])
        ax.set_xlabel(f"X ({units})")
        ax.set_ylabel(f"Y ({units})")
        ax.set_zlabel(f"Z ({units})")
        ax.view_init(elev=20, azim=45)

        speed_text = "N/A" if speeds[i] is None else f"{speeds[i]:.3f} {speed_units}"
        ax.set_title(
            f"Voxel centroid (frame {pos_frame_indices[i]})\n"
            f"speed ~ {speed_text}"
        )
        # Place legend outside the axes so it doesn't cover the plot.
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            framealpha=0.85,
        )

        frame_path = frames_dir / f"frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

        images.append(Image.open(frame_path))

    if images:
        gif_path = output_dir / "voxel_motion.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=args.duration,
            loop=0,
        )
        print(gif_path)
    else:
        print("No frames generated")


if __name__ == "__main__":
    main()
