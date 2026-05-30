# Corner Hint GUI

Use `tools/corner_hint_editor.html` to manually annotate corner points on a
reference path and export `corner_distance_hints_m` for training/evaluation.

## Reference Snapshot

Generate the current DG reference snapshot:

```powershell
.\.venv310-final\Scripts\python.exe tools\export_corner_reference.py --trajectory letter --label DG --letter-plane xz --output data\corner_hints\DG_reference.json
```

Open `tools/corner_hint_editor.html` in a browser, load
`data/corner_hints/DG_reference.json`, place or drag the manual corner markers,
then download the exported JSON.

The editor also accepts trajectory CSV files that contain `ref_x`, `ref_y`, and
`ref_z` columns, such as final artifact rollout CSVs.

## Evaluation

Pass the exported hint file into any script that uses `reference_factory`:

```powershell
.\.venv310-final\Scripts\python.exe -m src.train.train_phase_b --trajectory letter --label DG --letter-plane xz --corner-hints-path data\corner_hints\DG_manual_corners.json
```

The supplied hint distances replace the auto-detected corner indices on the
compiled `LightPaintRef`. The same updated corner set is then used by reward,
teacher/action filter logic, and corner metrics.
