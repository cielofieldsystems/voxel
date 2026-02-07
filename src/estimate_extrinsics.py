"""Estimate camera extrinsics from a chessboard image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate extrinsics using a chessboard")
    parser.add_argument("--image", required=True, help="Image with visible chessboard")
    parser.add_argument("--intrinsics", required=True, help="Intrinsics JSON")
    parser.add_argument("--pattern", nargs=2, type=int, required=True, help="Chessboard pattern cols rows")
    parser.add_argument("--square-size", type=float, required=True, help="Square size in world units")
    parser.add_argument("--output", required=True, help="Output extrinsics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    intr = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
    mtx = np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(intr.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)

    pattern_cols, pattern_rows = args.pattern
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= args.square_size

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)
    if not found:
        raise RuntimeError("Chessboard not found in image")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    rmat, _ = cv2.Rodrigues(rvec)
    cam_to_world = rmat.T
    camera_position = (-cam_to_world @ tvec).reshape(-1)

    output = {
        "camera_position": [float(v) for v in camera_position],
        "camera_to_world": cam_to_world.tolist(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
