# Pipeline

This is the intended end-to-end data flow for voxel backprojection.

## Inputs
- Frames per camera (PNG or JPEG).
- Camera calibration (intrinsics + extrinsics).
- Metadata describing frame timing and camera IDs.

## Dataset layout
```
data/
  demo_001/
    dataset.json
    metadata.json
    frames/
      cam_000/
        frame_000001.png
        frame_000002.png
      cam_001/
        frame_000001.png
        frame_000002.png
```

## Metadata schema (sketch)
Each entry links a frame to a camera pose.
```
[
  {
    "camera_id": "cam_000",
    "frame_index": 1,
    "timestamp": 1712345678.123,
    "image_file": "frames/cam_000/frame_000001.png",
    "intrinsics": {
      "fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0,
      "width": 1920, "height": 1080, "dist": [0,0,0,0,0]
    },
    "extrinsics": {
      "camera_position": [x,y,z],
      "camera_to_world": [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
    }
  }
]
```
Notes:
- You can inline `intrinsics` and `extrinsics` as shown, or reference files with `intrinsics_file` and `extrinsics_file`.

## Dataset metadata (optional)
`dataset.json` captures unit labels for grid outputs.
```
{
  "units": "meters",
  "units_per_grid_unit": 100.0,
  "description": "Synthetic multi-camera dataset"
}
```

## Processing steps
1) Ingest frames
   - Convert to grayscale and optionally denoise.
2) Motion extraction
   - Compute frame difference (optical flow could be added later).
   - Keep sparse pixels with motion magnitude.
3) Backprojection
   - For each motion pixel, cast a ray into world space.
   - Accumulate weights into a voxel grid.
4) Postprocess
   - Threshold or take top percentile.
   - Cluster peaks and track over time (optional).
5) Visualize
   - Render voxel points or export to a point cloud.

## Outputs
- `voxel_grid.bin` + `voxel_grid.json` for grid metadata.
- `voxel_motion.gif` (optional) for the voxel centroid animation.
- `speed_estimates.json` (optional) with per-frame position and speed.
- Optional `detections.json` with centroid and confidence per time step (future).

## Scripts
- `src/generate_synth_dataset.py` - creates a synthetic dataset for testing.
- `src/ingest_videos.py` - extracts frames from real videos.
- `src/calibrate_intrinsics.py` and `src/estimate_extrinsics.py` - camera calibration helpers.
- `src/pipeline.py` - runs motion -> voxel accumulation.
- `src/view_voxels.py` - visualizes voxel grids.
- `src/live_pipeline.py` - runs the live streaming pipeline.
- `src/render_voxel_gif.py` - renders a voxel-peak GIF and writes speed estimates.
