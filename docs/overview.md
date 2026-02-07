# Overview

This project explores multi-camera voxel backprojection for detecting tiny, fast-moving objects.

## Pipeline
1) Calibrate cameras (intrinsics, extrinsics, time sync).
2) Compute motion maps per camera (abs(frame_t - frame_{t-1}) baseline; optical flow optional).
3) For each motion pixel, cast a ray into 3D and add weight to intersected voxels.
4) Accumulate across cameras and frames.
5) Find peaks in the voxel grid (clustering and tracking are optional).

## Outputs and units
- `voxel_grid.bin` + `voxel_grid.json` are the core artifacts.
- Optional `voxel_motion.gif` and `speed_estimates.json` visualize the centroid motion.
- `dataset.json` can define units and `units_per_grid_unit` so speeds are in real units.

## Why it works (intuition)
Noise and near-field clutter generate rays that rarely intersect across many views. A real moving object produces consistent intersections, which stack up.

## Calibration sensitivity
Small angular errors create large spatial errors at long range. Typical mitigations:
- checkerboards or AprilTags for intrinsics
- multi-view bundle adjustment or surveyed baselines for extrinsics
- time alignment within a frame interval

## Limitations
- Atmospheric distortion, occlusion, clouds, and haze
- Horizon and line-of-sight constraints
- SNR for very dim targets; long integration may be needed
- Compute grows with grid size and number of rays
