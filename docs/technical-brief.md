# Technical brief

## One-liner
Voxel backprojection turns motion pixels from multiple cameras into 3D vote accumulation; where the votes intersect, something is actually moving.

## Problem framing
Small, fast objects are only a few pixels wide, so appearance-based detection is weak. Motion is the strongest signal, but a single 2D view cannot resolve distance. Multi-view geometry fixes that.

## Architecture (what happens and why)
1) Calibrate cameras (intrinsics + extrinsics)
   - Needed so each pixel can be turned into a 3D ray in a common coordinate frame.
2) Extract motion per camera
   - Use frame differencing to isolate moving pixels (optical flow optional).
   - Motion is robust when objects are tiny and low contrast.
3) Backproject motion pixels into a voxel grid
   - Each motion pixel casts a ray; voxel cells accumulate votes.
   - True targets create intersections; clutter becomes thin, isolated streaks.
4) Postprocess and visualize
   - Threshold high-score voxels to find targets (clustering optional).
   - Visualize intersections or export detections.

## Multi-target note
- "Votes" are just voxel score increments along each ray.
- Multiple moving objects create multiple peaks; clustering and temporal tracking separate them.

## Why this design
- Geometry-first: no ML needed to see tiny targets.
- Statistically strong: intersections rise above noise.
- Parallel-friendly: each pixel ray can be processed independently.

## Validation approach
- Start with synthetic data to validate geometry and accumulation.
- Move to real cameras once calibration and motion extraction are solid.

## Implementation notes
- `src/generate_synth_dataset.py` for synthetic multi-camera frames + metadata.
- `src/ingest_videos.py` for real video ingestion.
- `src/pipeline.py` runs motion extraction and voxel accumulation.
- `src/view_voxels.py` visualizes top percentile voxels.
- `src/live_pipeline.py` runs the streaming version.
- `src/render_voxel_gif.py` renders the voxel-peak GIF and speed estimates in real units.

## Risks and mitigations
- Calibration error: use checkerboards/AprilTags and refine extrinsics.
- Time sync drift: align timestamps or use hardware sync.
- Lens distortion: undistort frames using intrinsics.
- Atmospheric noise: integrate over more frames and tighten thresholds.
- Compute cost: start coarse, then refine around peaks; use sparse grids.
