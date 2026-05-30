# Final Goal Spec

This document is the project-level target for LightPaint RL work. Code changes
must move the project toward this target without weakening the problem,
evaluation, or PyBullet physics contract.

## Objective

Train and evaluate a Phase B policy that controls PID target-velocity residuals
and LED residuals so that a PyBullet drone can draw target light-painting
trajectories accurately under no disturbance and external-force disturbances.

The final system must work across multiple trajectories, wind modes, and seeds.
Single smoke runs and one-letter successes are not sufficient evidence of
completion.

## Evaluation Matrix

Final evaluation must cover:

- Trajectories: `square`, `L`, `DG`, `CAT`, `Pig`, `RL`, `drawn`
- Wind modes: `M0`, `M1`, `M2`
- Seeds: canonical `final` evaluation uses `7`, `11`, `17`, `23`, `29`
  (5 seeds). The `letters` and `standard` profiles use 3 seeds for iteration,
  but they are not final-completion evidence.

For every trajectory/wind/seed combination, store a per-run `summary.json`.
For the full sweep, store aggregate artifacts:

- `batch_summary.json`
- `metrics.csv`
- `aggregate_metrics.csv`
- `matrix_results.csv`: trajectory x wind x seed result table
- `best_model_manifest.json`

`metrics.csv` and `aggregate_metrics.csv` must be regenerated from the same
run matrix as `matrix_results.csv`. Stale aggregate files are not acceptable:
the verifier cross-checks `n`, pass counts, and pass rates against the matrix.

`batch_summary.json` must include `batch_final_goal_criteria` with:

- `matrix_complete`
- `all_runs_final_goal_pass`
- `failed_final_goal_runs`
- `robustness_relative_to_m0_pass`
- `batch_final_goal_pass`

The batch pass is true only when the expected trajectory x wind x seed matrix is
complete, every run passes its per-run `final_goal_pass`, and M1/M2 do not
collapse relative to M0.

When the batch does not pass, `failed_final_goal_runs` must list the failing
trajectory/wind/seed combinations and the failed per-run criteria so that the
next experiment can target the actual failure mode rather than guessing from a
single aggregate score.

## Final Metric Thresholds

The final goal is passed only when the aggregate evaluation satisfies these
thresholds and the worst failed combinations are explicitly reported.

### Painting Quality

- `painting_iou >= 0.75`
- `painting_precision >= 0.85`
- `painting_recall >= 0.85`
- `painting_dice >= 0.80`
- `off_target_ratio <= 0.10`

### Path Tracking

- `path_rmse_m <= 0.05`
- `path_max_m <= 0.15`
- `straight_path_rmse_m <= phaseB_zero.straight_path_rmse_m * 1.10`

### Corner Behavior

- `corner_path_rmse_m <= phaseB_zero.corner_path_rmse_m * 0.80`
- `corner_speed_ratio = corner_mean_speed_mps / reference_speed_mps` in
  `[0.65, 0.90]`
- `corner_overshoot_m` improves at least 20% over `phaseB_zero`

### LED Control

- `led_precision >= 0.90`
- `led_recall >= 0.90`
- `led_flicker_rate` must not be worse than the scripted baseline by more than
  `+0.05` absolute flicker-rate tolerance.

### Disturbance Robustness

- Evaluate all metrics on `M0`, `M1`, and `M2`.
- M1/M2 `painting_iou` must not drop more than 20% relative to M0.
- M1/M2 `path_rmse_m` must not worsen more than 30% relative to M0.
- `crash_rate == 0` or a documented near-zero tolerance.
- `out_of_bounds_rate == 0`.

### Physical Feasibility

- `rpm_saturation_ratio <= 0.10`
- `action_norm_mean`, `action_norm_max`, `action_rate_mean`, and
  `action_rate_max` must be reported.
- `action_norm_max <= 2.000001` and `action_rate_max <= 4.000001`, matching the
  theoretical bounds of a 4D `Box(-1, 1)` residual action and one-step action
  delta.

## Phase B Contract

These contracts must not be broken:

- `action_space = Box(-1, 1, shape=(4,), dtype=np.float32)`
- `action[0:3]` means PID target velocity residual.
- `action[3]` means LED residual command.
- `action=0` must match PID plus scripted LED baseline as closely as possible.
- `target_vel = v_ref + delta_v`.
- `LED command = led_ref + delta_led` must affect actual painting/update flow.
- M1/M2 disturbances must apply nonzero external force through PyBullet
  `applyExternalForce`.
- PyBullet evaluation must not be replaced by fallback-env evaluation.

## Allowed Change Areas

- Reward weights and reward terms when they are based on actual path, painting,
  LED, disturbance, or control outputs.
- PPO/BC hyperparameters and curriculum.
- Batch experiment and final evaluation orchestration.
- Dashboard presets, result comparison, and artifact visualization.
- Metrics and tests that make the final criteria harder to fake.

## Forbidden Shortcuts

- Lowering metric thresholds to create a pass.
- Weakening `overall_pass` or `final_goal_pass` formulas without a separate,
  documented evaluation-policy change.
- Making target masks, paths, scales, or drawn trajectories easier without
  explicitly marking the experiment as a different target.
- Treating crash, out-of-bounds, or force-application failures as success.
- Hiding PyBullet errors with broad `try/except`.
- Reporting fallback-env results as PyBullet results.
- Reporting `total_timesteps=0` or short smoke runs as learning success.
- Reporting only best seed, best trajectory, or M0-only results as final
  completion.

## Curriculum

Recommended final training flow:

1. M0 sanity: short run to verify env/API only.
2. M0 full: train every trajectory in the matrix long enough to learn baseline
   painting and corner behavior. A single square or one-letter run is only a
   local probe.
3. M1 robustness: continue every trajectory/seed from its M0 model under weak
   disturbance.
4. M2 robustness: continue every trajectory/seed from its M1 model under
   stronger disturbance.
5. Final eval: run `eval_only=true` over the full evaluation matrix.

The canonical batch profiles are:

- `sanity`: execution check only, not learning evidence.
- `letters`: all built-in letter trajectories, M0/M1/M2, 3 seeds, 100k PPO
  timesteps per run.
- `standard`: all trajectories, M0/M1/M2, 3 seeds, 100k PPO timesteps per run.
- `final`: all trajectories, M0/M1/M2, 5 seeds, 300k PPO timesteps per run.

Use `letters` for letter-focused reward/controller iteration, `standard` for
team-wide parameter exploration across all reference types, and `final` for
candidate claims that the project goal is met. Short smoke runs must never be
reported as learning success.

The dashboard and CLI should make this curriculum easy to run while preserving
all run metadata, source model paths, metrics, and artifacts.

The canonical final/batch evaluation uses
`--trained-action-filter corner_tangent_decel` and `--teacher-window-m 0.15`.
This does not replace the learned Phase B action: the policy still outputs the
4D residual action that is passed to the environment, and the filter only keeps
corner-window deceleration along the reference tangent while preserving the LED
residual. Use `--trained-action-filter none` for raw-policy ablations, and mark
those results separately from the canonical final-goal batch.

Long batch runs must be restartable. The batch runner should reuse an existing
per-run `summary.json` by default and rebuild `metrics.csv`,
`aggregate_metrics.csv`, `best_model_manifest.json`, and `batch_summary.json`
without re-running completed jobs. Reuse is allowed only when the saved
configuration still matches the current run, `artifacts.model_path` exists, and
for `final` profile runs the saved `final_goal_criteria` still matches
`summary.metrics`. The saved `total_timesteps` and `max_steps_override` must
also match the current command, so shortened smoke rollouts are not silently
reused as final evidence. Use `--no-resume` only when intentionally recomputing a run.
Curriculum wind order is strict: M1 must follow M0 for the same
trajectory/seed, and M2 must follow M1. Plans that skip the required prior wind
stage must fail preflight instead of silently training with the wrong source
model.

After a training batch, the final evaluation can be rerun without additional
learning by passing the training batch's `best_model_manifest.json` with
`--eval-only --model-manifest <path>`. Dashboard users should use the Batch tab's
`Eval-only` mode for this step so the final claim is based on a dedicated
evaluation artifact set. Relative model paths inside the manifest are resolved
relative to the manifest file, with package-root fallback only when that target
already exists.

Final completion is not proven until
`python -m src.train.verify_final_goal_batch <final-eval-output-dir>` exits with
code 0. The verifier requires `profile=final`, all required artifacts, complete
matrix coverage, every run's `final_goal_pass=true`, robustness criteria passing,
all per-run final criterion keys to be present and true, and
`batch_final_goal_criteria.batch_final_goal_pass=true`. It also recomputes each
per-run `final_goal_criteria` from `summary.metrics`, so success flags without
matching metric evidence are rejected. Batch-level robustness is likewise
recomputed from `matrix_results.csv`; stale or manually weakened
`batch_final_goal_criteria` fields are not accepted. The verifier also checks
that `matrix_results.csv` and `metrics.csv` numeric fields match each per-run
`summary.json` `phaseB_trained` metrics, and that `aggregate_metrics.csv`
group statistics are regenerated from the same matrix. For `final`
profile claims, the verifier also follows the eval batch's source training
`best_model_manifest.json` and checks that the source was a non-dry, non-eval
training batch over the canonical final matrix with `total_timesteps=300000`; a
short model or dry-run plan evaluated later under a final-shaped eval batch is
not acceptable evidence.
All recorded model paths must also point to readable Stable-Baselines zip
archives containing at least `data` and `policy.pth`; placeholder files or
manually fabricated zip paths are not acceptable evidence.
The source training matrix must also preserve curriculum provenance:
for each trajectory/seed, M0 starts without `load_model`, M1 must load the
matching M0 model, and M2 must load the matching M1 model. A final-shaped
artifact set that skips those links is not acceptable evidence.
Each final eval per-run `summary.json` must also show
`requested_total_timesteps=300000` and `total_timesteps=0`, because final eval is
an evaluation-only replay of the trained source model, not additional learning.
Canonical final runs must also leave `--max-steps` unset so each trajectory is
rolled out for its full reference duration plus settling time. A manually
shortened rollout is a smoke/debug run, not final evidence.
The canonical `drawn` input is `data/drawn_paths/user/user_drawn_path.json`
with SHA-256
`36715a93285c49f264cbd3ec9336825afba48615d125dc24310fa082027a9e2c`.
Batch summaries, source model manifests, and per-run drawn summaries must keep
that path/hash provenance. Replacing it with an easier JSON invalidates the
canonical final claim.

For the canonical end-to-end run, use
`python -m src.train.run_final_goal_pipeline --profile final --run-name final_candidate`.
The pipeline executes the training batch, reruns an eval-only batch from the
training batch's `best_model_manifest.json`, then runs the verifier. A pipeline
`--dry-run` produces train/eval plans without claiming completion. Non-dry-run
pipeline execution records a `preflight` block in `pipeline_summary.json` and
must fail before training if PyBullet/SB3 dependencies are missing or the
canonical `drawn` JSON input cannot be found. If the drawn input changes, pass
the intended file explicitly with `--drawn-path` and treat the result as tied to
that input artifact.
