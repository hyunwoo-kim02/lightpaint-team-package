"""
light_paint_aviary_pyb.py — LightPaintAviaryPyB (Phase A PyBullet).
Plan reference: joyful-painting-leaf.md CP-2.

Inherits gym_pybullet_drones BaseRLAviary so PyBullet CF2X dynamics + URDF
mesh are first-class. PID is `DSLPIDControl.computeControlFromState` (D7);
wind disturbance is injected via `_physics` override using
`p.applyExternalForce(WORLD_FRAME)` (D2/D3).

Phase A runs pure PID plus scripted LED. Phase B runs PID target velocity and
LED command residuals on top of the same baseline.

Composition (engine-agnostic, reused from CP-1):
- self.wind          : WindMode (M0/M1/M2 active)
- self.led_strategy  : LEDStrategy (ScriptedLED for A/B)
- self.G_letter      : binary target mask (case-preserved)
- self._ref_waypoints, self._ref_cumlen : skeleton-walk reference path

The shared standalone env (`light_paint_aviary_standalone.py`) remains for
unit-test fallback only; production runs go through this PyBullet env.
"""
from __future__ import annotations

# === PYBULLET-PATH-FIX-1 (must run before any pybullet/gpd import) ===
from src.env.pybullet_setup import apply_korean_path_fix as _apply_fix
_apply_fix()

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from gymnasium import spaces

import pybullet as p
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# CP-1 modules (engine-agnostic)
from src.env.wind_modes import make_wind_mode
from src.env.led_strategy import make_led_strategy
from src.env.lightpaint_ref import LightPaintRef, smooth_waypoint_path


# ----------------------------------------------------------------------
# Constants (mirrored from light_paint_aviary_standalone.py for parity)
# ----------------------------------------------------------------------
POS_NORM = 1.0
VEL_NORM = 1.5
ATT_NORM = math.pi
ANG_NORM = math.pi

V_REF = 0.5          # m/s reference speed (SRS §5.5)
DT = 1.0 / 30.0      # 30 Hz control frequency (matches BaseRLAviary ctrl_freq)
N_FUTURE = 15
RESIDUAL_DELTA_MAX = 0.5
LED_RESIDUAL_SCALE = 1.0

X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.5, 2.5
Y_CANVAS = 0.0

LED_STAMP_RADIUS_PX = 1
LAMBDA_SMOOTH = 0.4

MAX_EPISODE_STEPS_DEFAULT = 2000

_LEGACY_WIND_TO_MODE = {"W0": "M0", "W1": "M1", "W2": "M2", "W3": "M3"}


def world_to_pixel(x: float, z: float, size: int = 64) -> Tuple[int, int]:
    col = int(np.clip((x - X_MIN) / (X_MAX - X_MIN) * (size - 1), 0, size - 1))
    row = int(np.clip((Z_MAX - z) / (Z_MAX - Z_MIN) * (size - 1), 0, size - 1))
    return col, row


def pixel_to_world(col: int, row: int, size: int = 64) -> Tuple[float, float]:
    x = X_MIN + (col / (size - 1)) * (X_MAX - X_MIN)
    z = Z_MAX - (row / (size - 1)) * (Z_MAX - Z_MIN)
    return x, z


# ----------------------------------------------------------------------
class LightPaintAviaryPyB(BaseRLAviary):
    """PyBullet-backed Phase A light-painting env."""

    def __init__(
        self,
        label: Optional[str] = None,
        wind_mode: str = "M0",
        phase: Literal["A", "B", "C_discrete", "C_continuous"] = "A",
        gui: bool = False,
        init_box_size: float = 0.05,
        led_always_on: bool = False,
        max_episode_steps: Optional[int] = None,
        reference: Optional[LightPaintRef] = None,
        ctrl_freq: int = 30,
        pyb_freq: int = 240,
        # Backwards-compat aliases (mirror standalone API)
        letter: Optional[str] = None,
        wind: Optional[str] = None,
    ) -> None:
        # --- Resolve label / wind aliases (case preserved) ---
        if label is None and letter is None:
            label = "L"
        elif label is None:
            label = letter  # type: ignore[assignment]
        self.label = str(label)

        if wind is not None and wind in _LEGACY_WIND_TO_MODE:
            wind_mode = _LEGACY_WIND_TO_MODE[wind]
        self.wind_mode = wind_mode

        if phase == "A" and self.wind_mode != "M0":
            raise ValueError(
                f"Phase A enforces wind_mode='M0', got {self.wind_mode!r}."
            )

        self.phase = phase
        self.init_box_size = float(init_box_size)
        self.led_always_on = bool(led_always_on)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else MAX_EPISODE_STEPS_DEFAULT
        self.reference = reference
        self._ctrl_dt = 1.0 / float(ctrl_freq)

        # --- RNG (will be reseeded in reset()) ---
        self._rng = np.random.default_rng(42)

        # --- Load mask, build reference (must be ready BEFORE super().__init__) ---
        if self.reference is None:
            self._load_letter_mask()
            self._ref_waypoints, self._ref_cumlen = self._build_ref_trajectory()
        else:
            self._ref_waypoints = self.reference.waypoints.copy()
            self._ref_cumlen = self.reference.cumlen.copy()
            self.G_letter = self._build_reference_mask(
                self._ref_waypoints,
                segment_led=self.reference.segment_led,
            )

        # --- Composition (engine-agnostic) ---
        self.wind = make_wind_mode(self.wind_mode, self._rng)
        self.led_strategy = make_led_strategy(self.phase, self.G_letter)

        # --- Episode state (must exist before super().reset() / _computeObs is called) ---
        self.cumulative = np.zeros((64, 64), dtype=np.float32)
        self._latched_wind = np.zeros(3, dtype=np.float32)
        self._episode_step = 0
        self._prev_rpm = None
        self._last_p_ref = np.zeros(3, dtype=np.float32)
        self._last_v_ref = np.zeros(3, dtype=np.float32)
        self._last_delta_v = np.zeros(3, dtype=np.float32)
        self._last_target_vel = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(0, dtype=np.float32)
        self._last_yaw_ref = 0.0
        self._last_rpm = np.zeros(4, dtype=np.float32)
        self._last_led_ref = 0.0
        self._last_delta_led = 0.0
        self._last_brightness = 0.0
        self._last_tracking_err = 0.0
        self._last_out_of_bounds = False
        self._last_reward_components: Dict[str, float] = {}

        # --- Initial position at top of reference + noise ---
        start = self._ref_waypoints[0] if len(self._ref_waypoints) else np.array([0, 0, 1.5])
        noise = self._rng.uniform(-init_box_size, init_box_size, size=3).astype(np.float32)
        init_xyz = np.array([[start[0] + noise[0], 0.0, start[2] + noise[2]]], dtype=np.float32)

        # --- super().__init__ (PyBullet client + URDF) ---
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=init_xyz,
            initial_rpys=np.zeros((1, 3), dtype=np.float32),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obs=ObservationType.KIN,   # we override _computeObs to return Dict
            act=ActionType.RPM,
        )

        # --- DSLPIDControl (after super to make sure DroneModel is set up) ---
        self.dsl_pid = DSLPIDControl(drone_model=DroneModel.CF2X)

        # --- Override action/observation spaces post super() ---
        self.observation_space = spaces.Dict({
            "drone_state":   spaces.Box(low=-np.inf, high=np.inf, shape=(12,),       dtype=np.float32),
            "future_ref":    spaces.Box(low=-np.inf, high=np.inf, shape=(45,),       dtype=np.float32),
            "target_mask":   spaces.Box(low=0.0,     high=1.0,    shape=(1, 64, 64), dtype=np.float32),
            "progress_mask": spaces.Box(low=0.0,     high=1.0,    shape=(1, 64, 64), dtype=np.float32),
        })
        self.action_space = self._build_action_space(self.phase)

    # ------------------------------------------------------------------
    # Phase dispatch hooks
    # ------------------------------------------------------------------
    def _build_action_space(self, phase: str) -> spaces.Space:
        if phase == "A":
            return spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)
        if phase == "B":
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        if phase == "C_discrete":
            return spaces.Discrete(2)
        if phase == "C_continuous":
            return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        raise ValueError(f"Unknown phase: {phase!r}")

    # ------------------------------------------------------------------
    # Letter mask + reference path (CP-1 transcribe)
    # ------------------------------------------------------------------
    def _load_letter_mask(self) -> None:
        _here = Path(__file__).resolve().parent
        _pkg_root = _here.parent.parent  # .../lightpaint-team-package
        data_dir = _pkg_root / "data" / "letters"
        data_dir.mkdir(parents=True, exist_ok=True)
        png_path = data_dir / f"{self.label}.png"
        if png_path.is_file():
            from PIL import Image
            arr = np.array(Image.open(str(png_path)).convert("L"), dtype=np.uint8)
            self.G_letter = (arr > 127).astype(np.float32)
        else:
            pkg_str = str(_pkg_root)
            if pkg_str not in sys.path:
                sys.path.insert(0, pkg_str)
            from src.env.letter_masks import render_letter, save_mask
            self.G_letter = render_letter(self.label, size=64)
            save_mask(self.G_letter, png_path)

    def _build_ref_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Skeleton-walk reference path (transcribe of standalone CP-1 algorithm)."""
        binary = (self.G_letter > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            pts = np.array(
                [[0.0, 0.0, z] for z in np.linspace(Z_MIN, Z_MAX, 20)],
                dtype=np.float32,
            )
            cumlen = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            ).astype(np.float32)
            return pts, cumlen

        try:
            from skimage.morphology import skeletonize
            from skimage.measure import label as cc_label
            skel = skeletonize(binary, method="lee").astype(np.uint8)
            if skel.sum() == 0:
                skel = binary
            labels = cc_label(skel, connectivity=2)
            num_components = int(labels.max())
        except Exception:
            skel = binary
            labels = binary.astype(np.int32)
            num_components = 1
        if num_components == 0:
            num_components = 1
            labels = (skel > 0).astype(np.int32)

        if num_components > 1:
            centroid_xs = []
            for cid in range(1, num_components + 1):
                ys_c, xs_c = np.where(labels == cid)
                cx = float(xs_c.mean()) if len(xs_c) else 0.0
                centroid_xs.append((cx, cid))
            centroid_xs.sort()
            ordered_ids = [cid for _, cid in centroid_xs]
        else:
            ordered_ids = [1]

        pts_list: List[List[float]] = []
        for cid in ordered_ids:
            if num_components > 1:
                ys_cc, xs_cc = np.where(labels == cid)
            else:
                ys_cc, xs_cc = np.where(skel > 0)
            if len(xs_cc) == 0:
                continue
            for (px, py) in self._walk_skeleton_pixels(xs_cc, ys_cc):
                wx, wz = pixel_to_world(int(px), int(py), size=64)
                pts_list.append([wx, Y_CANVAS, wz])

        pts = np.array(pts_list, dtype=np.float32)
        if len(pts) > 3:
            pts = smooth_waypoint_path(
                pts,
                max_waypoints=240,
                smooth_window_m=0.08,
                samples_per_meter=120.0,
            )
        elif len(pts) > 400:
            idx = np.linspace(0, len(pts) - 1, 400, dtype=int)
            pts = pts[idx]
        cumlen = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        ).astype(np.float32)
        return pts, cumlen

    @staticmethod
    def _build_reference_mask(
        waypoints: np.ndarray,
        size: int = 64,
        segment_led: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Rasterize a waypoint reference into the existing 64x64 canvas mask."""
        pts = np.asarray(waypoints, dtype=np.float32)
        led = None if segment_led is None else np.asarray(segment_led, dtype=np.float32).reshape(-1)
        mask = np.zeros((size, size), dtype=np.float32)
        if len(pts) == 0:
            return mask
        for i in range(max(1, len(pts) - 1)):
            if led is not None and i < len(led) and float(led[i]) <= 0.0:
                continue
            p0 = pts[i]
            p1 = pts[min(i + 1, len(pts) - 1)]
            seg_len = float(np.linalg.norm(p1 - p0))
            n = max(2, int(np.ceil(seg_len * 96.0)))
            for alpha in np.linspace(0.0, 1.0, n, dtype=np.float32):
                p_ref = p0 * (1.0 - alpha) + p1 * alpha
                col, row = world_to_pixel(float(p_ref[0]), float(p_ref[2]), size=size)
                r = LED_STAMP_RADIUS_PX
                for dr in range(-r, r + 1):
                    for dc in range(-r, r + 1):
                        rr = int(np.clip(row + dr, 0, size - 1))
                        cc = int(np.clip(col + dc, 0, size - 1))
                        mask[rr, cc] = 1.0
        return mask

    @staticmethod
    def _walk_skeleton_pixels(xs_cc, ys_cc) -> List[Tuple[int, int]]:
        pixels = set(zip([int(v) for v in xs_cc], [int(v) for v in ys_cc]))
        if not pixels:
            return []

        def degree(px: int, py: int) -> int:
            return sum(
                1
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and (px + dx, py + dy) in pixels
            )

        endpoints = [pt for pt in pixels if degree(*pt) == 1]
        if endpoints:
            endpoints.sort(key=lambda pt: (pt[1], pt[0]))
            start = endpoints[0]
        else:
            start = min(pixels, key=lambda pt: (pt[1], pt[0]))

        ordered: List[Tuple[int, int]] = [start]
        visited = {start}
        current = start
        while True:
            best, best_d = None, 10.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (current[0] + dx, current[1] + dy)
                    if cand in pixels and cand not in visited:
                        d = (dx * dx + dy * dy) ** 0.5
                        if d < best_d:
                            best_d = d
                            best = cand
            if best is None:
                remaining = [p_ for p_ in pixels if p_ not in visited]
                if not remaining:
                    break
                cx, cy = current
                best = min(remaining, key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)
            ordered.append(best)
            visited.add(best)
            current = best
        return ordered

    # ------------------------------------------------------------------
    # Reference interpolation
    # ------------------------------------------------------------------
    def _interpolate_traj(self, s: float) -> np.ndarray:
        pts = self._ref_waypoints
        cumlen = self._ref_cumlen
        total_len = float(cumlen[-1])
        if total_len < 1e-6:
            return pts[0].copy()
        if s >= total_len:
            return pts[-1].copy()
        s = max(0.0, float(s))
        idx = int(np.searchsorted(cumlen, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(pts) - 2))
        seg_len = float(cumlen[idx + 1] - cumlen[idx])
        if seg_len < 1e-9:
            return pts[idx].copy()
        alpha = float(np.clip((s - float(cumlen[idx])) / seg_len, 0.0, 1.0))
        return (pts[idx] * (1.0 - alpha) + pts[idx + 1] * alpha).astype(np.float32)

    def _interpolate_traj_velocity(self, s: float) -> np.ndarray:
        pts = self._ref_waypoints
        cumlen = self._ref_cumlen
        total_len = float(cumlen[-1])
        if total_len < 1e-6 or s >= total_len:
            return np.zeros(3, dtype=np.float32)
        s = max(0.0, float(s))
        idx = int(np.searchsorted(cumlen, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(pts) - 2))
        tangent = pts[idx + 1] - pts[idx]
        n = float(np.linalg.norm(tangent))
        if n < 1e-9:
            return np.zeros(3, dtype=np.float32)
        return (tangent / n * V_REF).astype(np.float32)

    def _reference_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return position, velocity, yaw for the active reference at time t."""
        if self.reference is not None:
            p_ref = np.asarray(self.reference.pos(t), dtype=np.float32).reshape(3)
            v_ref = np.asarray(self.reference.vel(t), dtype=np.float32).reshape(3)
            yaw_ref = float(self.reference.yaw(t))
            return p_ref, v_ref, yaw_ref

        s_now = float(t) * V_REF
        return (
            self._interpolate_traj(s_now),
            self._interpolate_traj_velocity(s_now),
            0.0,
        )

    # ------------------------------------------------------------------
    # Future-ref obs helper (mirrors v4 reference)
    # ------------------------------------------------------------------
    def _get_future_ref_15pts(self, ep_step: int) -> np.ndarray:
        cur_pos = self._get_world_pos()
        if self.reference is not None:
            t_now = float(ep_step) * self._ctrl_dt
            world_pts = self.reference.future_positions(
                t_now,
                count=N_FUTURE,
                dt=3.0 * self._ctrl_dt,
            )
            return (world_pts - cur_pos[None, :]).astype(np.float32)

        total_len = float(self._ref_cumlen[-1])
        if total_len < 1e-6:
            return np.zeros((N_FUTURE, 3), dtype=np.float32)
        s_now = (float(ep_step) * DT * V_REF) % total_len
        pts = []
        for k in range(N_FUTURE):
            s_k = (s_now + 3.0 * k * DT * V_REF) % total_len
            world_pt = self._interpolate_traj(s_k)
            pts.append(world_pt - cur_pos)
        return np.array(pts, dtype=np.float32)

    def _get_world_pos(self) -> np.ndarray:
        try:
            state_20 = self._getDroneStateVector(0)
            return np.asarray(state_20[0:3], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # BaseRLAviary required overrides
    # ------------------------------------------------------------------
    def _observationSpace(self):
        """Return Dict obs. BaseRLAviary calls this in super().__init__."""
        return spaces.Dict({
            "drone_state":   spaces.Box(low=-np.inf, high=np.inf, shape=(12,),       dtype=np.float32),
            "future_ref":    spaces.Box(low=-np.inf, high=np.inf, shape=(45,),       dtype=np.float32),
            "target_mask":   spaces.Box(low=0.0,     high=1.0,    shape=(1, 64, 64), dtype=np.float32),
            "progress_mask": spaces.Box(low=0.0,     high=1.0,    shape=(1, 64, 64), dtype=np.float32),
        })

    def _actionSpace(self):
        return self._build_action_space(self.phase)

    def _computeObs(self):
        try:
            state_20 = self._getDroneStateVector(0)
        except Exception:
            state_20 = np.zeros(20, dtype=np.float32)
        state_20 = np.asarray(state_20, dtype=np.float32).flatten()
        if len(state_20) < 16:
            state_20 = np.pad(state_20, (0, 20 - len(state_20)))
        pos = state_20[0:3] / POS_NORM
        rpy = state_20[7:10] / ATT_NORM
        vel = state_20[10:13] / VEL_NORM
        ang_vel = state_20[13:16] / ANG_NORM
        drone_state = np.concatenate([pos, vel, rpy, ang_vel]).astype(np.float32)

        future_ref = self._get_future_ref_15pts(self._episode_step).flatten().astype(np.float32)
        target_mask = self.G_letter[None, :, :].astype(np.float32)
        progress_mask = self.cumulative[None, :, :].astype(np.float32)
        return {
            "drone_state":   drone_state,
            "future_ref":    future_ref,
            "target_mask":   target_mask,
            "progress_mask": progress_mask,
        }

    def _coerce_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).flatten()
        expected_shape = self.action_space.shape
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(
                f"action shape {arr.shape} does not match phase {self.phase!r} "
                f"action_space shape {expected_shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("action contains NaN or inf.")
        return np.clip(arr, self.action_space.low, self.action_space.high).astype(np.float32)

    def _preprocessAction(self, action):
        """Compute DSLPID RPM. Phase B action adds velocity and LED residuals."""
        if self.phase not in ("A", "B"):
            raise NotImplementedError(
                f"_preprocessAction for phase {self.phase!r} is not part of the active A/B path."
            )
        if self.phase == "A":
            action_arr = np.zeros(0, dtype=np.float32)
        else:
            action_arr = self._coerce_action(action)

        # 1. reference at the current control time
        t = self._episode_step * self._ctrl_dt
        p_ref, v_ref, yaw_ref = self._reference_state(t)
        if self.phase == "B":
            delta_v = (action_arr[0:3] * RESIDUAL_DELTA_MAX).astype(np.float32)
            delta_led = float(action_arr[3] * LED_RESIDUAL_SCALE)
        else:
            delta_v = np.zeros(3, dtype=np.float32)
            delta_led = 0.0
        target_vel = (v_ref + delta_v).astype(np.float32)

        # 2. drone state → DSLPID RPM
        state_20 = self._getDroneStateVector(0)
        try:
            rpm, _, _ = self.dsl_pid.computeControlFromState(
                control_timestep=self._ctrl_dt,
                state=state_20,
                target_pos=p_ref,
                target_rpy=np.array([0.0, 0.0, yaw_ref], dtype=np.float32),
                target_vel=target_vel,
            )
        except Exception as exc:
            raise RuntimeError("DSLPIDControl.computeControlFromState failed") from exc

        rpm = np.asarray(rpm, dtype=np.float32).flatten()
        if rpm.shape[0] != 4:
            raise ValueError(f"DSLPIDControl returned rpm shape {rpm.shape}, expected (4,).")
        if not np.all(np.isfinite(rpm)):
            raise ValueError("DSLPIDControl returned non-finite RPM values.")

        # 3. latch wind force for this control tick
        wind_force = np.asarray(self.wind.step(t), dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(wind_force)):
            raise ValueError("wind mode returned NaN or inf force.")
        self._latched_wind = wind_force

        # 4. cache for reward + info
        self._last_p_ref = p_ref.copy()
        self._last_v_ref = v_ref.copy()
        self._last_delta_v = delta_v.copy()
        self._last_target_vel = target_vel.copy()
        self._last_action = action_arr.copy()
        self._last_yaw_ref = float(yaw_ref)
        self._last_rpm = rpm.copy()
        self._last_delta_led = delta_led
        return rpm.reshape(self.NUM_DRONES, 4)

    def _physics(self, rpm, nth_drone):
        """Apply rotor thrust plus world-frame external disturbance force."""
        super()._physics(rpm, nth_drone)
        if np.linalg.norm(self._latched_wind) > 1e-9:
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                -1,
                forceObj=self._latched_wind.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )

    def _compute_reward_phase_a(
        self,
        pos: np.ndarray,
        p_ref: np.ndarray,
        brightness: float,
        rpm: np.ndarray,
        prev_rpm: Optional[np.ndarray],
        out_of_bounds: bool,
    ) -> Tuple[float, Dict[str, float]]:
        r_track = -float(np.linalg.norm(pos - p_ref))
        col, row = world_to_pixel(float(pos[0]), float(pos[2]))
        on_target = bool(self.G_letter[row, col] > 0.5)
        r_led_scripted = 1.0 if (brightness > 0.5 and on_target) else 0.0
        if prev_rpm is not None:
            du = (rpm - prev_rpm) / 5000.0  # normalize RPM delta
            r_smooth = float(np.exp(-float(np.linalg.norm(du)))) * LAMBDA_SMOOTH
        else:
            r_smooth = LAMBDA_SMOOTH
        r_terminal = -100.0 if out_of_bounds else 0.0
        total = r_track + r_led_scripted + r_smooth + r_terminal
        return total, {
            "r_track": r_track,
            "r_led_scripted": r_led_scripted,
            "r_smooth": r_smooth,
            "r_terminal": r_terminal,
        }

    def _stamp_led_progress(self, pos: np.ndarray, brightness: float = 1.0) -> None:
        if brightness <= 0.0:
            return
        col, row = world_to_pixel(float(pos[0]), float(pos[2]))
        r = LED_STAMP_RADIUS_PX
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr = int(np.clip(row + dr, 0, 63))
                cc = int(np.clip(col + dc, 0, 63))
                self.cumulative[rr, cc] = max(float(self.cumulative[rr, cc]), float(brightness))

    def _scripted_led_ref(self, pos: np.ndarray, t_led: float) -> float:
        if self.led_always_on:
            return 1.0
        if self.reference is not None:
            return float(self.reference.led(t_led))
        return float(self.led_strategy.decide(pos, np.zeros(0, dtype=np.float32), self.G_letter))

    def _computeReward(self) -> float:
        pos = self._get_world_pos()
        t_led = float(self._episode_step) * self._ctrl_dt
        led_ref = self._scripted_led_ref(pos, t_led)
        delta_led = self._last_delta_led if self.phase == "B" else 0.0
        brightness = float(np.clip(led_ref + delta_led, 0.0, 1.0))
        if brightness > 0.0:
            self._stamp_led_progress(pos, brightness)
        # Out-of-bounds check (xz arena + small y latitude)
        out_of_bounds = bool(
            pos[2] < Z_MIN - 0.5
            or pos[2] > Z_MAX + 0.5
            or pos[0] < X_MIN - 0.8
            or pos[0] > X_MAX + 0.8
        )
        reward, components = self._compute_reward_phase_a(
            pos=pos,
            p_ref=self._last_p_ref,
            brightness=brightness,
            rpm=self._last_rpm,
            prev_rpm=self._prev_rpm,
            out_of_bounds=out_of_bounds,
        )
        self._prev_rpm = self._last_rpm.copy()
        self._last_led_ref = float(led_ref)
        self._last_brightness = brightness
        self._last_tracking_err = float(np.linalg.norm(pos - self._last_p_ref))
        self._last_out_of_bounds = out_of_bounds
        self._last_reward_components = components
        self._episode_step += 1
        return float(reward)

    def _computeTerminated(self) -> bool:
        return bool(self._last_out_of_bounds)

    def _computeTruncated(self) -> bool:
        return bool(self._episode_step >= self.max_episode_steps)

    def _computeInfo(self) -> Dict[str, Any]:
        pos = self._get_world_pos()
        t = float(self._episode_step) * self._ctrl_dt
        reference_done = bool(self.reference is not None and t >= self.reference.duration)
        return {
            "t": t,
            "ref_name": self.reference.name if self.reference is not None else "letter_skeleton",
            "pos": pos.tolist(),
            "p_ref": self._last_p_ref.tolist(),
            "v_ref": self._last_v_ref.tolist(),
            "delta_v": self._last_delta_v.tolist(),
            "target_vel": self._last_target_vel.tolist(),
            "yaw_ref": float(self._last_yaw_ref),
            "reference_done": reference_done,
            "action": self._last_action.tolist(),
            "u_pid_rpm": self._last_rpm.tolist(),
            "wind_force": self._latched_wind.tolist(),
            "wind_frame": "WORLD_FRAME",
            "led_ref": float(self._last_led_ref),
            "delta_led": float(self._last_delta_led),
            "brightness": float(self._last_brightness),
            "led_on": bool(self._last_brightness > 0.5),
            "painted_px": int(self.cumulative.sum()),
            "tracking_err": float(self._last_tracking_err),
            "r": float(sum(self._last_reward_components.values())) if self._last_reward_components else 0.0,
            **self._last_reward_components,
        }

    # ------------------------------------------------------------------
    # Reset (wraps super to reset our composition)
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Re-sample wind / reset PID
        self.wind.reset(self._rng)
        try:
            self.dsl_pid.reset()
        except Exception:
            pass

        # Reset paint + episode state BEFORE super so any _computeObs is well-defined
        self.cumulative = np.zeros((64, 64), dtype=np.float32)
        self._latched_wind = np.zeros(3, dtype=np.float32)
        self._episode_step = 0
        self._prev_rpm = None
        p0, v0, yaw0 = self._reference_state(0.0)
        self._last_p_ref = p0.copy()
        self._last_v_ref = v0.copy()
        self._last_delta_v = np.zeros(3, dtype=np.float32)
        self._last_target_vel = v0.copy()
        action_shape = self._build_action_space(self.phase).shape or (0,)
        self._last_action = np.zeros(action_shape, dtype=np.float32)
        self._last_yaw_ref = float(yaw0)
        self._last_rpm = np.zeros(4, dtype=np.float32)
        self._last_led_ref = 0.0
        self._last_delta_led = 0.0
        self._last_brightness = 0.0
        self._last_tracking_err = 0.0
        self._last_out_of_bounds = False
        self._last_reward_components = {}

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    # ------------------------------------------------------------------
    # User-facing step
    # ------------------------------------------------------------------
    def step(self, action):
        if self.phase == "A":
            action = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        elif self.phase == "B":
            action = self._coerce_action(action)
        else:
            raise NotImplementedError(
                f"step for phase {self.phase!r} is not part of the active A/B path."
            )
        return super().step(action)
