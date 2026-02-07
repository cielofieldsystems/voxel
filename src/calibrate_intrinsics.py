"""Compute camera intrinsics from chessboard images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate intrinsics from chessboard images")
    parser.add_argument("--images", nargs="+", required=True, help="Calibration images")
    parser.add_argument("--pattern", nargs=2, type=int, required=True, help="Chessboard pattern cols rows")
    parser.add_argument("--square-size", type=float, required=True, help="Square size in world units")
    parser.add_argument("--output", required=True, help="Output intrinsics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern_cols, pattern_rows = args.pattern
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []
    image_size = None

    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)
        if not found:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

    if not objpoints:
        raise RuntimeError("No chessboard corners detected.")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    fx, fy = float(mtx[0, 0]), float(mtx[1, 1])
    cx, cy = float(mtx[0, 2]), float(mtx[1, 2])

    output = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(image_size[0]),
        "height": int(image_size[1]),
        "dist": dist.flatten().tolist(),
        "reprojection_error": float(ret),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
