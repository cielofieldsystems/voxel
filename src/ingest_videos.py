"""Extract frames from videos and build a metadata skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest videos into dataset layout")
    parser.add_argument("--out", required=True, help="Output dataset folder")
    parser.add_argument("--videos", nargs="+", required=True, help="Video files per camera")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per video")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--intrinsics-dir", default=None, help="Folder with intrinsics JSON files")
    parser.add_argument("--extrinsics-dir", default=None, help="Folder with extrinsics JSON files")
    return parser.parse_args()


def find_calibration_file(folder: str | None, camera_id: str, kind: str) -> str | None:
    if not folder:
        return None
    folder_path = Path(folder)
    candidate = folder_path / f"{camera_id}_{kind}.json"
    if candidate.exists():
        return str(candidate)
    return None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    metadata = []

    for cam_idx, video_path in enumerate(args.videos):
        camera_id = f"cam_{cam_idx:03d}"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        cam_dir = frames_root / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.stride == 0:
                filename = f"frame_{saved:06d}.png"
                out_path = cam_dir / filename
                cv2.imwrite(str(out_path), frame)

                entry = {
                    "camera_id": camera_id,
                    "frame_index": saved,
                    "timestamp": saved / args.fps,
                    "image_file": str(out_path.relative_to(out_dir)),
                }

                intr_file = find_calibration_file(args.intrinsics_dir, camera_id, "intrinsics")
                extr_file = find_calibration_file(args.extrinsics_dir, camera_id, "extrinsics")
                if intr_file:
                    intr_path = Path(intr_file)
                    try:
                        entry["intrinsics_file"] = str(intr_path.relative_to(out_dir))
                    except ValueError:
                        entry["intrinsics_file"] = str(intr_path)
                if extr_file:
                    extr_path = Path(extr_file)
                    try:
                        entry["extrinsics_file"] = str(extr_path.relative_to(out_dir))
                    except ValueError:
                        entry["extrinsics_file"] = str(extr_path)

                metadata.append(entry)
                saved += 1

                if args.max_frames and saved >= args.max_frames:
                    break

            frame_idx += 1

        cap.release()

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
