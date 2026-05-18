# Drawn Path Library

User-drawn light-painting paths are stored here as JSON stroke files.

Directory layout:

- `user/`: paths drawn by the project user and used for current experiments.
- `examples/`: small sample paths for smoke tests and documentation.

JSON contract:

- `coordinate_space`: `normalized`, `pixel`, or `world`.
- `plane`: `xz` or `xy`.
- `strokes`: list of stroke objects.
- Each stroke has `points` and optional `led`.
- Stroke segments are LED ON by default.
- Connector motion between strokes is inserted automatically with LED OFF.

Run a user path:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_a_pid --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --wind-mode M0 --seed 42
```
