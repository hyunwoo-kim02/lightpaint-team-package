# DG Final Goal Spec

This is the active DG-only target for the current iteration. It replaces the
previous all-trajectory 105-run target for this workstream, but keeps the same
PyBullet, residual-action, and artifact integrity expectations.

## Objective

Make the `DG` light-painting path reliable under `M0`, `M1`, and `M2`, using the
three strongest internal repository results in
`shared_dg_m0_m1_m2_repro_20260527` as the baseline:

- `DG_M0_seed7`
- `DG_M1_seed11`
- `DG_M2_seed29`

The target is not a smoke test. Completion requires reproducible PyBullet
evaluation artifacts for all three runs, reusable model weights, and a verifier
or equivalent audit proving every metric below.

This iteration is DG-focused. The practical goal is to raise visual quality for
the DG letter in each disturbance mode while making LED on/off behavior
unambiguous: the learned policy must not miss intended LED-on strokes, must not
paint through LED-off connector motion, and must not introduce flicker that
makes the letter look broken.

## Canonical DG Reference

Use the existing internal DG reference as the canonical path:

- Reference builder: `make_letter_ref("DG", plane="xz")`
- Reference name: `letter_DG_xz_smooth`
- Trajectory arguments: `--trajectory letter --label DG --letter-plane xz`
- Manual corner hints:
  `data/corner_hints/DG_manual_corners_10.json`

The `image_to_waypoints` path-generation experiment is not the canonical target.
It was useful for inspection, but its skeleton/waypoint extraction is not used
for final DG training or evaluation. The current internal `letter_DG_xz_smooth`
path remains the path contract; only corner/curve signaling and policy behavior
are allowed to improve against it.

## Canonical DG Matrix

The DG-only canonical matrix is intentionally non-Cartesian:

| run_id | trajectory | wind_mode | seed | baseline weight |
|---|---|---|---:|---|
| `DG_M0_seed7` | `DG` | `M0` | 7 | `shared_dg_m0_m1_m2_repro_20260527/runs/DG_M0_seed7/weights/reusable_weight.zip` |
| `DG_M1_seed11` | `DG` | `M1` | 11 | `shared_dg_m0_m1_m2_repro_20260527/runs/DG_M1_seed11/weights/reusable_weight.zip` |
| `DG_M2_seed29` | `DG` | `M2` | 29 | `shared_dg_m0_m1_m2_repro_20260527/runs/DG_M2_seed29/weights/reusable_weight.zip` |

All evaluations use `--trajectory letter --label DG --letter-plane xz`,
`--corner-hints-path data/corner_hints/DG_manual_corners_10.json`,
`--trained-action-filter corner_tangent_decel`, `--device cuda`, and full
reference-duration rollouts with no manual `--max-steps` shortening.

## Baseline Metrics

The shared bundle establishes the initial reusable-weight baseline:

| run_id | IoU | path_rmse_m | off_target_ratio | led_precision | led_recall | led_flicker_rate |
|---|---:|---:|---:|---:|---:|---:|
| `DG_M0_seed7` | 0.859649 | 0.014171 | 0.039216 | 0.994949 | 0.989950 | 0.015094 |
| `DG_M1_seed11` | 0.828947 | 0.018550694 | 0.091346 | 1.000000 | 1.000000 | 0.097378 |
| `DG_M2_seed29` | 0.816594 | 0.018157 | 0.096618 | 1.000000 | 0.941520 | 0.074906 |

## Pass Criteria

Each canonical DG run must pass the original final-goal per-run thresholds:

- `painting_iou >= 0.75`
- `painting_precision >= 0.85`
- `painting_recall >= 0.85`
- `painting_dice >= 0.80`
- `off_target_ratio <= 0.10`
- `path_rmse_m <= 0.05`
- `path_max_m <= 0.15`
- `corner_path_rmse_m <= phaseB_zero.corner_path_rmse_m * 0.80`
- `corner_speed_ratio` in `[0.65, 0.90]`
- `corner_overshoot_m <= phaseB_zero.corner_overshoot_m * 0.80`
- `crash_rate == 0`
- `out_of_bounds_rate == 0`
- `rpm_saturation_ratio <= 0.10`
- `action_norm_max <= 2.000001`
- `action_rate_max <= 4.000001`

The DG LED target is stricter than the previous generic goal:

- `led_precision >= 0.995`
- `led_recall >= 0.995`
- `led_flicker_rate <= phaseB_zero.led_flicker_rate + 0.05`
- LED-off connector segments must be reported explicitly. A run that reaches the
  painting thresholds by leaving the LED on through connector motion is not a
  valid DG pass.
- LED-on stroke gaps must be reported explicitly. A run that reaches acceptable
  path tracking but visibly drops LED brightness on intended strokes is not a
  valid DG pass.

This initial shared bundle is a strong path/painting baseline, but by itself it
shows why the DG workstream is LED-sensitive: `DG_M0_seed7` and `DG_M2_seed29`
need LED recall improvement before they can be treated as final LED evidence.
The later `Current Final Baseline` section records the verified DG-only artifact
that closes this LED gap; new candidates must preserve or improve on that
artifact, not merely beat the initial shared bundle table.

## Per-Wind Improvement Targets

The hard pass/fail gate is the criteria above. In addition, candidate policies
should be compared against both the initial shared bundle and the verified
current final baseline for the same wind mode:

| wind_mode | primary quality target | LED target |
|---|---|---|
| `M0` | Preserve the high-quality no-disturbance shape; do not regress IoU, path RMSE, or off-target ratio beyond measurement noise. | If starting from the initial bundle, close the LED recall gap; if starting from the current final baseline, preserve strict LED pass. |
| `M1` | Keep the letter readable under weak disturbance and avoid quality collapse relative to M0. | Preserve perfect or near-perfect precision/recall while reducing unnecessary flicker where possible. |
| `M2` | Improve or preserve path/painting quality under stronger disturbance; off-target ratio must stay below the DG threshold. | If starting from the initial bundle, raise LED recall to the strict DG target; if starting from the current final baseline, preserve strict LED pass without painting through connectors. |

If a candidate improves LED metrics but worsens visual quality, it is not a
final candidate. If it improves path quality by masking LED failures, it is also
not a final candidate. The desired solution improves the DG letter as seen on
the painting canvas and keeps LED timing correct in all three wind modes.

## Corner Hint And LED Improvement Plan

The next DG experiment must isolate reference signaling from policy learning:

1. Re-evaluate the three shared weights on `letter_DG_xz_smooth` with
   `DG_manual_corners_10.json` and no additional learning.
2. Confirm that the 10 manual corner hints are applied and that the baseline
   metrics remain comparable to the current final baseline.
3. If the reference/hint change is stable, continue from the same reusable
   weights with DG-only training focused on LED on/off correctness and
   disturbance-mode quality.
4. Evaluate every candidate with the same three canonical DG runs and the same
   corner-hint file.
5. Keep a candidate only if it meets all DG pass criteria and does not trade LED
   correctness for worse painting quality.

The intended measurement is the delta in `corner_path_rmse_m`,
`corner_speed_ratio`, `path_rmse_m`, `painting_iou`, `led_precision`,
`led_recall`, and `led_flicker_rate` against the baseline artifact.

## Completion Evidence

Completion requires:

- Per-run `summary.json` for the three canonical DG runs.
- `metrics.csv`, `aggregate_metrics.csv`, `matrix_results.csv`, and a manifest
  that records the exact shared weight used for each run.
- Experiment tracking files under `teams/lightpaint/experiments/run-*`.
- A final audit showing that all three runs meet the DG pass criteria above.

## Current Final Baseline

The current verified DG-only baseline is:

- Artifact root:
  `artifacts/dg_goal/dg_final_shared3_led_20260529`
- Experiment record:
  `teams/lightpaint/experiments/run-dg-final-shared3-led-v1`
- Audit:
  `artifacts/dg_goal/dg_final_shared3_led_20260529/final_audit.md`

All three canonical runs pass the original final-goal criteria and the stricter
DG LED criteria (`led_precision >= 0.995`, `led_recall >= 0.995`).

The reproduction mismatch investigation found that `DG_M0_seed7` in the shared
bundle was evaluated with a connector-inclusive target mask
(`target_total_px=220`). Current DG reference evaluation, and the shared M1/M2
artifacts, use `target_total_px=209`. The final M0 baseline therefore records
`LIGHTPAINT_LEGACY_REFERENCE_MASK_CONNECTORS=1` explicitly rather than treating
the mismatch as a weight issue.
