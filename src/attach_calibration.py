"""Attach intrinsics/extrinsics file references to metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach calibration references to metadata")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--intrinsics-dir", default=None, help="Folder with intrinsics JSON")
    parser.add_argument("--extrinsics-dir", default=None, help="Folder with extrinsics JSON")
    parser.add_argument("--output", default=None, help="Output metadata path (defaults to overwrite)")
    return parser.parse_args()


def find_file(folder: str | None, camera_id: str, kind: str) -> Path | None:
    if not folder:
        return None
    folder_path = Path(folder)
    candidate = folder_path / f"{camera_id}_{kind}.json"
    return candidate if candidate.exists() else None


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    base_dir = metadata_path.parent
    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    for entry in data:
        camera_id = entry.get("camera_id", entry.get("camera_index", "cam_000"))

        intr_path = find_file(args.intrinsics_dir, camera_id, "intrinsics")
        if intr_path:
            try:
                entry["intrinsics_file"] = str(intr_path.relative_to(base_dir))
            except ValueError:
                entry["intrinsics_file"] = str(intr_path)

        extr_path = find_file(args.extrinsics_dir, camera_id, "extrinsics")
        if extr_path:
            try:
                entry["extrinsics_file"] = str(extr_path.relative_to(base_dir))
            except ValueError:
                entry["extrinsics_file"] = str(extr_path)

    out_path = Path(args.output) if args.output else metadata_path
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
