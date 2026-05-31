# Drawn Path Library

User-drawn light-painting paths are stored here as JSON stroke files.

Directory layout:

- `user/`: the canonical user-drawn path used for current experiments.

Generated or converted reference paths should be regenerated on demand and not
kept in this directory unless they become a new canonical input.

JSON contract:

- `coordinate_space`: `normalized`, `pixel`, or `world`.
- `plane`: `xz` or `xy`.
- `width_m` and `height_m`: base canvas size in meters.
- `path_scale`: optional positive multiplier applied to `width_m` and `height_m`.
- `strokes`: list of stroke objects.
- Each stroke has `points` and optional `led`.
- Stroke segments are LED ON by default.
- Connector motion between strokes is inserted automatically with LED OFF.

Run a user path:

```powershell
python -m src.train.train_phase_a_pid --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --wind-mode M0 --seed 42
```

Scale the same JSON without editing the file:

```powershell
python -m src.train.train_phase_b --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --path-scale 0.8 --wind-mode M1 --total-timesteps 128
```
