"""Environment checks for the LightPaint PyBullet environment.

Checks:
    A1: Dict obs structure (4 keys + correct shapes)
    A2: target_mask.sum() > 100 (letter has pixels)
    A3: future_ref(step=0) != future_ref(step=10) (time dynamics)
    A4: guided_mean > random_mean (reward sensitivity, advisory)

Outputs: artifacts/env_sanity_w1.stdout
Exit code 0 if A1+A2+A3 PASS; exit code 1 on any hard failure.
"""
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Resolve package root for module-level import (handles both `python -m` and direct invocation)
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
_REPO_ROOT = _PKG_ROOT.parent.parent
for _p in [str(_PKG_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from gymnasium import spaces

_ARTIFACTS = _PKG_ROOT / "artifacts"
_ARTIFACTS.mkdir(parents=True, exist_ok=True)
STDOUT_PATH = _ARTIFACTS / "env_sanity_w1.stdout"

_log_lines: list = []


def _log(msg: str) -> None:
    """Print and buffer a timestamped log line."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


def _save_log() -> None:
    """Write all buffered log lines to STDOUT_PATH."""
    with open(str(STDOUT_PATH), "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


def run_sanity_checks() -> bool:
    """
    Run all checks and return True if all required asserts pass.

    Writes log to artifacts/env_sanity_w1.stdout.
    Exits with code 1 if any hard assert fails.
    """
    _log("env_sanity_w1.py - LightPaint environment check")
    _log(f"Python: {sys.version}")
    _log(f"PKG_ROOT: {_PKG_ROOT}")

    # --- Import env (PyBullet) ---
    _log("Importing LightPaintAviaryPyB...")
    try:
        from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB
        _log("Import OK")
    except Exception as exc:
        _log(f"FATAL: Cannot import LightPaintAviaryPyB: {exc}")
        import traceback
        _log(traceback.format_exc())
        _save_log()
        sys.exit(1)

    # --- Create env (Phase A, M0) ---
    _log("Creating LightPaintAviaryPyB(label='L', phase='A', wind_mode='M0')...")
    try:
        env = LightPaintAviaryPyB(label="L", phase="A", wind_mode="M0", gui=False,
                                   init_box_size=0.05, led_always_on=False)
        _log("Env created OK")
    except Exception as exc:
        _log(f"FATAL: Env creation failed: {exc}")
        import traceback
        _log(traceback.format_exc())
        _save_log()
        sys.exit(1)

    # --- Reset ---
    _log("Calling env.reset(seed=42)...")
    try:
        obs, info = env.reset(seed=42)
        _log(f"Reset OK. obs type={type(obs).__name__}")
    except Exception as exc:
        _log(f"FATAL: env.reset() failed: {exc}")
        import traceback
        _log(traceback.format_exc())
        env.close()
        _save_log()
        sys.exit(1)

    # ==========================================================
    # A1: Dict obs structure
    # ==========================================================
    _log("\n--- A1: Dict obs structure ---")
    a1_pass = True
    try:
        assert isinstance(env.observation_space, spaces.Dict), (
            f"obs_space type={type(env.observation_space).__name__} expected Dict"
        )
        _log("  observation_space type: Dict - PASS")

        expected_keys = {"drone_state", "future_ref", "target_mask", "progress_mask"}
        assert set(obs.keys()) == expected_keys, (
            f"obs.keys()={set(obs.keys())} expected {expected_keys}"
        )
        _log(f"  obs.keys() = {set(obs.keys())} - PASS")

        assert obs["drone_state"].shape == (12,), (
            f"drone_state.shape={obs['drone_state'].shape} expected (12,)"
        )
        _log(f"  drone_state.shape = {obs['drone_state'].shape} - PASS")

        assert obs["future_ref"].shape == (45,), (
            f"future_ref.shape={obs['future_ref'].shape} expected (45,)"
        )
        _log(f"  future_ref.shape = {obs['future_ref'].shape} - PASS")

        assert obs["target_mask"].shape == (1, 64, 64), (
            f"target_mask.shape={obs['target_mask'].shape} expected (1,64,64)"
        )
        _log(f"  target_mask.shape = {obs['target_mask'].shape} - PASS")

        assert obs["progress_mask"].shape == (1, 64, 64), (
            f"progress_mask.shape={obs['progress_mask'].shape} expected (1,64,64)"
        )
        _log(f"  progress_mask.shape = {obs['progress_mask'].shape} - PASS")

        _log("A1: PASS - Dict obs structure correct")
    except AssertionError as exc:
        _log(f"A1: FAIL - {exc}")
        a1_pass = False

    # ==========================================================
    # A2: target_mask non-zero
    # ==========================================================
    _log("\n--- A2: target_mask non-zero ---")
    a2_pass = True
    try:
        mask_sum = float(obs["target_mask"].sum())
        _log(f"  target_mask.sum() = {mask_sum:.1f}")
        assert mask_sum > 100, (
            f"target_mask too sparse: {mask_sum} pixels (expected >100)"
        )
        _log(f"A2: PASS - target_mask has {mask_sum:.0f} non-zero pixels (>100 threshold)")
    except AssertionError as exc:
        _log(f"A2: FAIL - {exc}")
        a2_pass = False

    # ==========================================================
    # A3: future_ref time dynamics
    # ==========================================================
    _log("\n--- A3: future_ref time dynamics ---")
    a3_pass = True
    try:
        ref_t0 = env._get_future_ref_15pts(0).flatten()
        ref_t10 = env._get_future_ref_15pts(10).flatten()
        max_diff = float(np.max(np.abs(ref_t0 - ref_t10)))
        _log(f"  future_ref(step=0)  first 6 vals: {ref_t0[:6]}")
        _log(f"  future_ref(step=10) first 6 vals: {ref_t10[:6]}")
        _log(f"  max_diff = {max_diff:.6f}")
        assert not np.allclose(ref_t0, ref_t10, atol=1e-6), (
            "future_ref is static - time dynamics broken"
        )
        _log(f"A3: PASS - future_ref changes with step (max_diff={max_diff:.6f})")
    except AssertionError as exc:
        _log(f"A3: FAIL - {exc}")
        a3_pass = False

    # ==========================================================
    # A4: Phase A reward signal (PID convergence + scripted LED)
    # ==========================================================
    _log("\n--- A4: Phase A reward signal (PID convergence) ---")
    a4_pass = True
    env.reset(seed=42)
    rewards = []
    initial_err = None
    final_err = None
    for k in range(150):
        _, r, term, trunc, info = env.step(np.zeros(0, dtype=np.float32))
        rewards.append(r)
        if k == 0:
            initial_err = info["tracking_err"]
        if k == 149:
            final_err = info["tracking_err"]
        if term or trunc:
            break
    mean_reward = float(np.mean(rewards))
    _log(f"  initial tracking_err = {initial_err:.4f} m")
    _log(f"  final   tracking_err = {final_err:.4f} m")
    _log(f"  mean reward over {len(rewards)} steps = {mean_reward:.4f}")
    try:
        assert final_err < 0.30, (
            f"PID failed to converge: final tracking_err={final_err:.4f} m (>0.30)"
        )
        _log(f"A4: PASS - PID converges (final_err {final_err:.4f} < 0.30)")
    except AssertionError as exc:
        _log(f"A4: FAIL - {exc}")
        a4_pass = False

    # ==========================================================
    # A5: Phase A action_space is empty Box(0,)
    # ==========================================================
    _log("\n--- A5: Phase A action_space.shape == (0,) ---")
    a5_pass = True
    try:
        assert env.action_space.shape == (0,), (
            f"Phase A action_space.shape={env.action_space.shape} expected (0,)"
        )
        _log(f"A5: PASS - action_space={env.action_space}")
    except AssertionError as exc:
        _log(f"A5: FAIL - {exc}")
        a5_pass = False

    # ==========================================================
    # A6: env.step(np.zeros(0)) returns valid obs/reward
    # ==========================================================
    _log("\n--- A6: env.step(zeros(0)) returns valid result ---")
    a6_pass = True
    try:
        env.reset(seed=42)
        out = env.step(np.zeros(0, dtype=np.float32))
        assert len(out) == 5, "step must return 5-tuple"
        obs_a6, r_a6, term_a6, trunc_a6, info_a6 = out
        assert isinstance(obs_a6, dict)
        assert "tracking_err" in info_a6
        assert isinstance(r_a6, float)
        _log(f"A6: PASS - step returned reward={r_a6:.4f} tracking_err={info_a6['tracking_err']:.4f}")
    except AssertionError as exc:
        _log(f"A6: FAIL - {exc}")
        a6_pass = False

    env.close()

    # ==========================================================
    # A7: M0Wind returns zero force
    # ==========================================================
    _log("\n--- A7: M0Wind.step(t) == zeros(3) ---")
    a7_pass = True
    try:
        from src.env.wind_modes import M0Wind
        wind = M0Wind()
        wind.reset(np.random.default_rng(0))
        f = wind.step(0.0)
        assert f.shape == (3,) and np.allclose(f, 0.0), f"M0Wind.step={f}"
        _log("A7: PASS - M0Wind emits zero force")
    except AssertionError as exc:
        _log(f"A7: FAIL - {exc}")
        a7_pass = False

    # ==========================================================
    # A8: VelocityPID zero state returns zero command
    # ==========================================================
    _log("\n--- A8: VelocityPID(zero state) == zeros(3) ---")
    a8_pass = True
    try:
        from src.env.pid_controller import VelocityPID
        pid = VelocityPID()
        u = pid.compute(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        assert u.shape == (3,) and np.allclose(u, 0.0), f"PID.compute zero state={u}"
        _log("A8: PASS - VelocityPID(zero state) returns zero command")
    except AssertionError as exc:
        _log(f"A8: FAIL - {exc}")
        a8_pass = False

    # ==========================================================
    # A9: PyBullet runtime hooks (drone URDF loaded + 20-dim state)
    # ==========================================================
    _log("\n--- A9: PyBullet runtime (CF2X URDF + state_20) ---")
    a9_pass = True
    try:
        # Build a fresh env to inspect (the earlier `env` was closed at end of A4)
        env_a9 = LightPaintAviaryPyB(label="L", phase="A", wind_mode="M0")
        env_a9.reset(seed=0)
        drone_id = int(env_a9.DRONE_IDS[0])
        client_id = int(env_a9.CLIENT)
        state_20 = env_a9._getDroneStateVector(0)
        _log(f"  DRONE_IDS[0]={drone_id}, CLIENT={client_id}, state_20.shape={state_20.shape}")
        assert drone_id >= 0, f"drone_id={drone_id} must be >= 0"
        assert client_id >= 0, f"client_id={client_id} must be >= 0"
        assert state_20.shape == (20,), f"state_20 shape={state_20.shape} expected (20,)"
        # DSLPIDControl wired
        assert env_a9.dsl_pid is not None
        env_a9.close()
        _log("A9: PASS - CF2X URDF loaded, state_20 OK, DSLPIDControl wired")
    except AssertionError as exc:
        _log(f"A9: FAIL - {exc}")
        a9_pass = False
    except Exception as exc:
        _log(f"A9: FAIL (exception) - {exc}")
        a9_pass = False

    # ==========================================================
    # Summary
    # ==========================================================
    _log("\n=== SUMMARY ===")
    all_hard_pass = (a1_pass and a2_pass and a3_pass and a4_pass and a5_pass
                     and a6_pass and a7_pass and a8_pass and a9_pass)
    _log(f"  A1 (Dict structure):              {'PASS' if a1_pass else 'FAIL'}")
    _log(f"  A2 (mask non-zero):               {'PASS' if a2_pass else 'FAIL'}")
    _log(f"  A3 (future_ref time dynamics):    {'PASS' if a3_pass else 'FAIL'}")
    _log(f"  A4 (Phase A PID convergence):     {'PASS' if a4_pass else 'FAIL'}")
    _log(f"  A5 (Phase A action_space empty):  {'PASS' if a5_pass else 'FAIL'}")
    _log(f"  A6 (env.step(zeros(0)) OK):       {'PASS' if a6_pass else 'FAIL'}")
    _log(f"  A7 (M0Wind zero force):           {'PASS' if a7_pass else 'FAIL'}")
    _log(f"  A8 (VelocityPID zero state):      {'PASS' if a8_pass else 'FAIL'}")
    _log(f"  A9 (PyBullet URDF + state_20):    {'PASS' if a9_pass else 'FAIL'}")

    if all_hard_pass:
        _log("\nALL REQUIRED ASSERTS PASS - proceed to training")
    else:
        _log("\nFAILURE - one or more required asserts FAILED")
        _log("DO NOT proceed to training until all asserts pass.")
        _save_log()
        sys.exit(1)

    _save_log()
    return all_hard_pass


if __name__ == "__main__":
    passed = run_sanity_checks()
    sys.exit(0 if passed else 1)
