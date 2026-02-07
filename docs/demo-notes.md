# Demo notes

This captures two reproducible synthetic demo presets:
- Fast iteration (quick to run)
- Presentation GIF (240 frames, smoother and longer)

Defaults used for both (not overridden):
- `--num-cams 8`
- `--fov 70`
- `--radius 3.0`
- `--fps 30`

Common rationale:
- `path-mode spline` creates a more dynamic, curved trajectory.
- The spline is re-sampled by arc length for near-constant speed (more natural motion).
- Units are explicit so speed estimates are interpretable.

## Preset A: fast iteration (shorter GIF)

Synthetic dataset:
```powershell
python src\generate_synth_dataset.py --out data\demo_001 --num-frames 120 --width 1280 --height 720 --path-scale 3.0 --spot-sigma 2.0 --noise 1.0 --units meters --units-per-grid-unit 100 --path-mode spline
```

Voxel accumulation:
```powershell
python src\pipeline.py --dataset data\demo_001 --output output\demo_001 --grid-center 0 0 0 --grid-extent 4 --grid-size 80 --threshold 8 --blur 3 --max-motion-pixels 6000
```

GIF render:
```powershell
python src\render_voxel_gif.py --dataset data\demo_001 --output output\demo_001 --threshold 8 --blur 3 --max-motion-pixels 6000 --top-k 12 --candidate-k 120 --track-gate 0.6 --smooth-window 5 --integration-window 3 --skip-first 5 --max-frames 70
```

## Preset B: presentation GIF (240 frames)

Synthetic dataset:
```powershell
python src\generate_synth_dataset.py --out data\demo_001 --num-frames 253 --width 1280 --height 720 --path-scale 3.0 --spot-sigma 2.0 --noise 1.0 --units meters --units-per-grid-unit 100 --path-mode spline
```

Voxel accumulation:
```powershell
python src\pipeline.py --dataset data\demo_001 --output output\demo_001 --grid-center 0 0 0 --grid-extent 4 --grid-size 80 --threshold 6 --blur 3 --max-motion-pixels 6000
```

GIF render (240 frames):
```powershell
python src\render_voxel_gif.py --dataset data\demo_001 --output output\demo_001 --threshold 6 --blur 3 --max-motion-pixels 6000 --top-k 12 --candidate-k 200 --track-gate 0.12 --smooth-window 5 --integration-window 3 --skip-first 12 --max-frames 240
```

Notes:
- `--candidate-k` + `--track-gate` enforce continuity. If no plausible peak is near the previous estimate, the tracker holds position rather than jumping.
- `--skip-first` burns in the estimator and avoids early unstable frames.
- Speed labels use `m/s` when units are meters.

## Outputs to share
- `data/demo_001/metadata.json` - per-frame camera poses and timestamps.
- `data/demo_001/dataset.json` - units (`meters`) and scaling (`100` per grid unit).
- `output/demo_001/voxel_grid.bin` + `output/demo_001/voxel_grid.json` - voxel grid and metadata.
- `output/demo_001/top_voxels.json` - top voxel centers + scores.
- `output/demo_001/voxel_motion.gif` - animated voxel centroid with speed.
- `output/demo_001/speed_estimates.json` - per-frame positions and speed values.
- `output/demo_001/voxels.png` and `output/demo_001/rays.png` - optional debug visuals.

## Speed sanity check
Expected speed from the synthetic spline path (Preset B):
- Path length ~4.116 grid units over 252 steps at 30 FPS (~8.4 s).
- Expected average speed ~0.49 grid-units/s.
- With `100` meters per grid unit, this is ~49 m/s.

Minor variation is expected due to voxel discretization, thresholding, and per-frame top-k centroids.
