"""Visualize voxel grids by plotting top-percentile voxels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from voxel_grid import load_voxel_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View voxel grid")
    parser.add_argument("--bin", required=True, help="Path to voxel_grid.bin")
    parser.add_argument("--meta", required=True, help="Path to voxel_grid.json")
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--save", default=None, help="Save image path")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vox = load_voxel_grid(Path(args.bin), Path(args.meta))
    flat = vox.grid.ravel()
    threshold = np.percentile(flat, args.percentile)

    coords = np.argwhere(vox.grid >= threshold)
    if coords.size == 0:
        print("No voxels above threshold")
        return

    if args.max_points and coords.shape[0] > args.max_points:
        indices = np.random.choice(coords.shape[0], args.max_points, replace=False)
        coords = coords[indices]

    points = vox.bmin + (coords + 0.5) * vox.voxel_size
    values = vox.grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, s=2, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
