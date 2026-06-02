"""
light_paint_aviary_standalone.py - Standalone numpy-only LightPaint env.

This module mirrors the LightPaint control interface without requiring
pybullet/gym-pybullet-drones. The PyBullet implementation remains the main
simulation environment for final experiments.

Class name `LightPaintAviaryW1` is preserved for existing imports.

Phase dispatch:
- 'A'           : PID-only check, action_space = Box(0,0,(0,)), wind_mode='M0' enforced.
- 'B'           : PID + velocity/LED residual, action_space = Box(-1,1,(4,)).
- 'C_discrete'  : frozen B + Discrete(2) LED interface.
- 'C_continuous': frozen B + Box(0,1,(1,)) brightness interface.

Composition:
- self.pid          : VelocityPID
- self.wind         : WindMode (M0/M1/M2)
- self.led_strategy : LEDStrategy (ScriptedLED for A/B; Learned* for C)

Accepted aliases:
- `letter=` arg accepted as alias for `label=`.
- `wind='W0'..'W2'` accepted as aliases for wind_mode='M0'..'M2'.
"""
import os
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.lightpaint_ref import corner_window_radius_m, effective_corner_indices
from src.env.reward_config import (
    CORNER_WINDOW_M,
    LED_RESIDUAL_SCALE,
    RESIDUAL_DELTA_MAX,
    W_ACTION_MAG,
    W_ACTION_RATE,
    W_COMPLETION,
    W_CORNER_ACCEL,
    W_CORNER_DECEL,
    W_CORNER_LATERAL,
    W_CORNER_PATH,
    W_CORNER_SPEED,
    W_CORNER_TRACK,
    W_FLICKER,
    W_INCOMPLETE,
    W_LED_FLICKER,
    W_LED_MISS,
    W_NEW_TARGET,
    W_OFF_TARGET,
    W_PATH,
    W_REPAINT,
    W_SCHEDULE,
)
from src.env.reward_terms import compute_lightpaint_reward, is_led_on, led_brightness_from_ref

# --- Observation normalization constants ---
POS_NORM = 1.0   # divide position by 1.0 m
VEL_NORM = 1.5   # divide velocity by 1.5 m/s
ATT_NORM = math.pi  # divide attitude (euler) by pi
ANG_NORM = math.pi  # divide angular velocity by pi

# --- Future reference constants ---
V_REF = 0.5      # m/s reference speed
DT = 1.0 / 30.0  # 30 Hz control frequency
N_FUTURE = 15    # future reference points (N_FUTURE × 3 = 45-dim future_ref)

# --- World bounds for pixel mapping ---
X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.5, 2.5
Y_CANVAS = 0.0   # canvas is in XZ plane at Y=0
Y_BOUND_M = 0.45

# --- Physics simulation constants ---
GRAVITY = 9.81         # m/s^2
MASS = 0.027           # kg (Crazyflie 2.1 class)
MAX_VEL = 2.0          # m/s maximum commanded velocity per axis
MAX_ANG_VEL = math.pi  # rad/s maximum commanded yaw rate
VEL_DAMP = 0.85        # velocity damping per step (air resistance approximation)
ANG_DAMP = 0.80        # angular velocity damping per step
MAX_EPISODE_STEPS = 2000  # default episode horizon (overridable via env init)

# --- Phase B residual scaling ---

_WIND_ALIAS_TO_MODE = {"W0": "M0", "W1": "M1", "W2": "M2"}

# --- Reward weights ---
W_LED_ASYM_POS = 1.0   # reward for LED ON + on target pixel
W_LED_ASYM_NEG = 0.1   # penalty for LED ON + off target pixel
W_PROGRESS = 2.0       # weight for r_progress (anti-hover, alpha=1.0)
GAMMA_SHAPE = 0.99     # Ng-1999 potential-based shaping discount

# --- LED stamp constants ---
LED_STAMP_RADIUS_PX = 1  # pixel radius for LED stamp in progress_mask


def world_to_pixel(x: float, z: float, size: int = 64) -> Tuple[int, int]:
    """
    Convert world (x, z) to pixel (col, row) in 64x64 grid.

    World X_MIN..X_MAX maps to col 0..63 (left to right).
    World Z_MAX..Z_MIN maps to row 0..63 (top to bottom).
    """
    col = int(np.clip((x - X_MIN) / (X_MAX - X_MIN) * (size - 1), 0, size - 1))
    row = int(np.clip((Z_MAX - z) / (Z_MAX - Z_MIN) * (size - 1), 0, size - 1))
    return col, row


def pixel_to_world(col: int, row: int, size: int = 64) -> Tuple[float, float]:
    """Convert pixel (col, row) to world (x, z) coordinates."""
    x = X_MIN + (col / (size - 1)) * (X_MAX - X_MIN)
    z = Z_MAX - (row / (size - 1)) * (Z_MAX - Z_MIN)
    return x, z


class LightPaintAviaryW1(gym.Env):
    """
    Single-drone LED light-painting environment.

    Standalone gymnasium.Env (no pybullet dependency) with simplified
    double-integrator physics for CPU/GPU-agnostic smoke-train on Windows.

    Implements:
    - Dict observation with drone state, future reference, target mask, and progress mask.
    - Residual velocity and LED action interface.
    - Light-painting reward terms.
    - Wind disturbance modes from src.env.wind_modes (M0/M1/M2)
    - 15-point future reference sequence.
    - Built-in letter mask pool.
    - Progress mask updated every step.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        label: Optional[str] = None,
        wind_mode: str = "M0",
        phase: Literal["A", "B", "C_discrete", "C_continuous"] = "A",
        gui: bool = False,
        init_box_size: float = 0.05,
        led_always_on: bool = False,
        max_episode_steps: Optional[int] = None,
        # Input aliases
        letter: Optional[str] = None,
        wind: Optional[str] = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            label: Text label to paint (e.g. 'L', 'RL', 'Pig'). Case-sensitive.
            wind_mode: Wind disturbance mode. M0/M1/M2 are active.
            phase: 'A' (PID check) | 'B' (wind robust) | 'C_discrete' | 'C_continuous'.
            gui: Ignored.
            init_box_size: Initial position noise box half-size in meters.
            led_always_on: If True, LED forced ON every step.
            letter: alias for `label`.
            wind: optional 'W0'..'W2' alias mapped to wind_mode.
        """
        super().__init__()

        # Resolve label / letter alias (case-preserved, no .upper())
        if label is None and letter is None:
            label = "L"
        elif label is None:
            label = letter  # type: ignore[assignment]
        self.label = str(label)
        self.letter = self.label

        # Resolve wind / wind_mode alias
        if wind is not None and wind in _WIND_ALIAS_TO_MODE:
            wind_mode = _WIND_ALIAS_TO_MODE[wind]
        self.wind_mode = wind_mode
        self.wind = wind_mode

        # Phase A enforces M0.
        if phase == "A" and self.wind_mode != "M0":
            raise ValueError(
                f"Phase A enforces wind_mode='M0', got {self.wind_mode!r}."
            )

        self.phase = phase
        self.gui = gui
        self.init_box_size = init_box_size
        self.led_always_on = led_always_on
        self.max_episode_steps = (
            int(max_episode_steps) if max_episode_steps is not None else MAX_EPISODE_STEPS
        )

        self._rng = np.random.default_rng(42)

        # Load label mask (case-sensitive)
        self._load_letter_mask()

        # Build reference trajectory using connected-component ordering
        self._ref_waypoints, self._ref_cumlen = self._build_ref_trajectory()
        self._corner_indices = self._resolve_corner_indices()
        self._corner_sharpness_lookup = self._resolve_corner_sharpness_lookup()

        # --- Composition: PID controller, wind generator, LED strategy ---
        from src.env.pid_controller import VelocityPID
        from src.env.wind_modes import make_wind_mode
        from src.env.led_strategy import make_led_strategy

        self.pid = VelocityPID(Kp=2.0, Kd=0.4, max_vel=MAX_VEL)
        self.wind = make_wind_mode(self.wind_mode, self._rng)
        self.led_strategy = make_led_strategy(self.phase, self.G_letter)

        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            "drone_state":   spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(12,), dtype=np.float32),
            "future_ref":    spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(45,), dtype=np.float32),
            "target_mask":   spaces.Box(low=0.0, high=1.0,
                                        shape=(1, 64, 64), dtype=np.float32),
            "progress_mask": spaces.Box(low=0.0, high=1.0,
                                        shape=(1, 64, 64), dtype=np.float32),
        })
        self.action_space = self._build_action_space(self.phase)

        # Episode state (initialized in reset())
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._rpy = np.zeros(3, dtype=np.float32)  # roll, pitch, yaw
        self._ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_u_final = np.zeros(3, dtype=np.float32)
        self._prev_delta_u_rl = None
        self._prev_brightness = 0.0
        self._prev_led = 0
        self._completion_reward_given = False
        self._episode_step = 0
        self.cumulative = np.zeros((64, 64), dtype=np.float32)
        self._prev_pos = np.zeros(3, dtype=np.float32)

    def _resolve_corner_indices(self) -> np.ndarray:
        return effective_corner_indices(self._ref_waypoints, self._ref_cumlen, window_m=CORNER_WINDOW_M)

    def _sparsify_corner_indices(self, indices: np.ndarray) -> np.ndarray:
        return effective_corner_indices(
            self._ref_waypoints,
            self._ref_cumlen,
            corner_indices=np.asarray(indices, dtype=np.int32),
            window_m=CORNER_WINDOW_M,
        )

    def _resolve_corner_sharpness_lookup(self) -> dict[int, float]:
        lookup: dict[int, float] = {}
        for idx in np.asarray(getattr(self, "_corner_indices", []), dtype=np.int32).reshape(-1):
            lookup[int(idx)] = self._corner_sharpness_at_waypoint(int(idx))
        return lookup

    def _corner_sharpness_for_index(self, corner_idx: int) -> float:
        default_sharpness = self._corner_sharpness_at_waypoint(corner_idx)
        return float(np.clip(self._corner_sharpness_lookup.get(int(corner_idx), default_sharpness), 0.0, 1.0))

    def _load_letter_mask(self) -> None:
        """
        Load (or render) the target label mask from data/letters/{label}.png.

        Case-sensitive: 'Pig' and 'PIG' use distinct PNG paths and cache keys.
        Auto-creates the PNG file on miss so subsequent loads share the artifact.
        """
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
            # Render inline and persist for reproducibility
            pkg_str = str(_pkg_root)
            if pkg_str not in sys.path:
                sys.path.insert(0, pkg_str)
            from src.env.letter_masks import render_letter, save_mask
            self.G_letter = render_letter(self.label, size=64)
            save_mask(self.G_letter, png_path)

    def _build_ref_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an arc-length-parameterized reference trajectory.

        Pipeline:
          1) Skeletonize the filled label mask to get a 1-pixel-wide curve.
          2) Connected-component label the skeleton; sort components by centroid x.
          3) For each component, walk pixels endpoint-to-endpoint (greedy
             nearest neighbor) so the path is locally continuous instead of
             jumping at row boundaries.
          4) Project to world (x, y=0, z) and arc-length parameterize.

        Multi-character labels (e.g. "RL") trace 'R' fully before 'L'.
        """
        binary = (self.G_letter > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            pts = np.array(
                [[0.0, 0.0, z] for z in np.linspace(Z_MIN, Z_MAX, 20)],
                dtype=np.float32,
            )
            diffs = np.diff(pts, axis=0)
            cumlen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))]).astype(np.float32)
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

        # Sort components left → right by centroid x
        component_ids = list(range(1, num_components + 1)) if num_components > 1 else [1]
        if num_components > 1:
            centroid_xs = []
            for cid in component_ids:
                ys_c, xs_c = np.where(labels == cid)
                cx = float(xs_c.mean()) if len(xs_c) > 0 else 0.0
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
            ordered = self._walk_skeleton_pixels(xs_cc, ys_cc)
            for (px, py) in ordered:
                wx, wz = pixel_to_world(int(px), int(py), size=64)
                pts_list.append([wx, Y_CANVAS, wz])

        pts = np.array(pts_list, dtype=np.float32)

        if len(pts) > 400:
            idx = np.linspace(0, len(pts) - 1, 400, dtype=int)
            pts = pts[idx]

        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)]).astype(np.float32)
        return pts, cumlen

    @staticmethod
    def _walk_skeleton_pixels(
        xs_cc: np.ndarray, ys_cc: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Greedy endpoint-to-endpoint walk over a connected skeleton component.

        Starts from a degree-1 pixel (an endpoint) if any exists, otherwise the
        top-left pixel. At each step, picks the unvisited neighbor with the
        smallest 8-connectivity distance.
        """
        pixels = set(zip([int(v) for v in xs_cc], [int(v) for v in ys_cc]))
        if not pixels:
            return []

        def degree(px: int, py: int) -> int:
            d = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (px + dx, py + dy) in pixels:
                        d += 1
            return d

        endpoints = [pt for pt in pixels if degree(*pt) == 1]
        if endpoints:
            # Pick top-most endpoint (smallest y), break ties by smallest x
            endpoints.sort(key=lambda pt: (pt[1], pt[0]))
            start = endpoints[0]
        else:
            start = min(pixels, key=lambda pt: (pt[1], pt[0]))

        ordered: List[Tuple[int, int]] = [start]
        visited = {start}
        current = start
        while True:
            best = None
            best_d = 10.0
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
                # Try to jump to nearest unvisited pixel anywhere in component
                remaining = [p for p in pixels if p not in visited]
                if not remaining:
                    break
                cx, cy = current
                best = min(remaining, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            ordered.append(best)
            visited.add(best)
            current = best
        return ordered

    def _interpolate_traj(self, s: float) -> np.ndarray:
        """Linear interpolation of 3D position at arc-length s along the trajectory."""
        pts = self._ref_waypoints
        cumlen = self._ref_cumlen
        total_len = float(cumlen[-1])
        if total_len < 1e-6:
            return pts[0].copy()
        # Clamp at the end (do not wrap) so PID can settle at the final waypoint
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
        """Tangent velocity vector at arc-length s, magnitude = V_REF (m/s)."""
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

    def _get_future_ref(self, step: int) -> np.ndarray:
        """
        Get N_FUTURE=15 future reference trajectory points in body frame.

        Computes arc-length position at t=step*DT with V_REF=0.5 m/s,
        samples N_FUTURE points spaced DT*V_REF apart, converts to body frame
        (relative to current drone pos). Returns shape (15, 3).
        """
        total_len = float(self._ref_cumlen[-1])
        if total_len < 1e-6:
            return np.zeros((N_FUTURE, 3), dtype=np.float32)

        s_now = (float(step) * DT * V_REF) % total_len
        future_pts = []
        for k in range(N_FUTURE):
            s_k = (s_now + k * DT * V_REF) % total_len
            world_pt = self._interpolate_traj(s_k)
            # Body frame: relative to current drone position
            body_pt = world_pt - self._pos
            future_pts.append(body_pt)
        return np.array(future_pts, dtype=np.float32)  # (15, 3)

    def _potential(self, pos: np.ndarray) -> float:
        """
        Ng-1999 potential function: distance to nearest on-target pixel.

        phi(s) = -min_distance_to_target_pixel, so moving closer increases potential.
        """
        ys, xs = np.where(self.G_letter > 0.5)
        if len(xs) == 0:
            return 0.0
        # Convert drone XZ to pixel
        col, row = world_to_pixel(float(pos[0]), float(pos[2]))
        # Nearest target pixel in pixel space
        dists = np.sqrt((xs - col) ** 2 + (ys - row) ** 2)
        min_dist_px = float(dists.min())
        # Scale: 1 pixel ~ (X_MAX-X_MIN)/63 m
        px_scale = (X_MAX - X_MIN) / 63.0
        min_dist_m = min_dist_px * px_scale
        return -min_dist_m  # higher = closer to target

    def _stamp_led_progress(self, pos: np.ndarray, brightness: float = 1.0) -> None:
        """
        Stamp LED paint into cumulative progress_mask weighted by brightness ∈ [0,1].

        Phase A uses brightness in {0.0, 1.0}; residual phases may use
        continuous brightness for soft accumulation.
        """
        if brightness <= 0.0:
            return
        col, row = world_to_pixel(float(pos[0]), float(pos[2]))
        r = LED_STAMP_RADIUS_PX
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr = int(np.clip(row + dr, 0, 63))
                cc = int(np.clip(col + dc, 0, 63))
                # max-merge so re-painting the same pixel does not overflow the [0,1] range
                self.cumulative[rr, cc] = max(float(self.cumulative[rr, cc]), float(brightness))

    def _nearest_path_stats(self, pos: np.ndarray) -> Tuple[float, int, float]:
        pts = np.asarray(self._ref_waypoints, dtype=np.float32)
        p0 = np.asarray(pos, dtype=np.float32).reshape(3)
        if len(pts) < 2:
            return float(np.linalg.norm(p0 - pts[0])), 0, 0.0
        seg = pts[1:] - pts[:-1]
        rel = p0[None, :] - pts[:-1]
        seg_len2 = np.sum(seg * seg, axis=1)
        alpha = np.clip(np.sum(rel * seg, axis=1) / np.maximum(seg_len2, 1e-9), 0.0, 1.0)
        closest = pts[:-1] + alpha[:, None] * seg
        dists = np.linalg.norm(closest - p0[None, :], axis=1)
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx, float(alpha[idx])

    def _corner_sharpness(self, segment_idx: int) -> float:
        pts = np.asarray(self._ref_waypoints, dtype=np.float32)
        if len(pts) < 3 or segment_idx >= len(pts) - 2:
            return 0.0
        return self._corner_sharpness_at_waypoint(segment_idx + 1)

    def _corner_sharpness_at_waypoint(self, corner_idx: int) -> float:
        pts = np.asarray(self._ref_waypoints, dtype=np.float32)
        if len(pts) < 3 or corner_idx <= 0 or corner_idx >= len(pts) - 1:
            return 0.0
        a = pts[corner_idx] - pts[corner_idx - 1]
        b = pts[corner_idx + 1] - pts[corner_idx]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(np.clip((1.0 - cosang) * 0.5, 0.0, 1.0))

    def _corner_context_for_distance(self, s_now: float) -> Tuple[float, float, float]:
        pts = np.asarray(self._ref_waypoints, dtype=np.float32)
        cum = np.asarray(self._ref_cumlen, dtype=np.float32)
        if len(pts) < 3 or len(cum) != len(pts):
            return 0.0, 0.0, float("inf")
        corner_indices = np.asarray(getattr(self, "_corner_indices", []), dtype=np.int32)
        if corner_indices.size == 0:
            return 0.0, 0.0, float("inf")
        s_now = float(np.clip(float(s_now), 0.0, float(cum[-1])))
        corner_s = cum[corner_indices]
        nearest_rel = int(np.argmin(np.abs(corner_s - s_now)))
        corner_idx = int(corner_indices[nearest_rel])
        corner_dist = float(abs(float(cum[corner_idx]) - s_now))
        radius_m = corner_window_radius_m(cum, corner_indices, CORNER_WINDOW_M)
        if radius_m <= 0.0 or corner_dist > radius_m:
            return 0.0, 0.0, corner_dist
        influence = (1.0 - corner_dist / max(radius_m, 1e-6)) ** 2
        sharpness = self._corner_sharpness_for_index(corner_idx)
        return float(influence * sharpness), float(sharpness), corner_dist

    def _corner_context(self, segment_idx: int, alpha: float) -> Tuple[float, float, float]:
        pts = np.asarray(self._ref_waypoints, dtype=np.float32)
        cum = np.asarray(self._ref_cumlen, dtype=np.float32)
        if len(pts) < 3 or len(cum) != len(pts) or segment_idx >= len(pts) - 1:
            return 0.0, 0.0, float("inf")
        seg_len = float(np.linalg.norm(pts[segment_idx + 1] - pts[segment_idx]))
        s_now = float(cum[segment_idx]) + float(np.clip(alpha, 0.0, 1.0)) * seg_len
        return self._corner_context_for_distance(s_now)

    def _scheduled_corner_context(self, t: float) -> Tuple[float, float, float]:
        return self._corner_context_for_distance(float(t) * V_REF)

    def _paint_stats(self, pos: np.ndarray, brightness: float) -> Tuple[float, float, float]:
        if brightness <= 0.0:
            return 0.0, 0.0, 0.0
        col, row = world_to_pixel(float(pos[0]), float(pos[2]))
        r = LED_STAMP_RADIUS_PX
        new_target = 0
        off_target = 0
        repaint = 0
        total = 0
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr = int(np.clip(row + dr, 0, 63))
                cc = int(np.clip(col + dc, 0, 63))
                total += 1
                if float(self.G_letter[rr, cc]) > 0.5:
                    if float(self.cumulative[rr, cc]) <= 1e-6:
                        new_target += 1
                    else:
                        repaint += 1
                else:
                    off_target += 1
        denom = float(max(total, 1))
        return new_target / denom, off_target / denom, repaint / denom

    def _paint_coverage(self) -> float:
        target = np.asarray(self.G_letter, dtype=np.float32) > 0.5
        target_px = int(target.sum())
        if target_px <= 0:
            return 0.0
        painted = np.asarray(self.cumulative, dtype=np.float32) > 0.3
        return float(np.logical_and(painted, target).sum()) / float(target_px)

    def _reference_done_for_reward(self, t: float) -> bool:
        if len(self._ref_cumlen) <= 0:
            return False
        return bool(float(t) >= float(self._ref_cumlen[-1]) / max(V_REF, 1e-6))

    def _is_out_of_bounds(self, pos: np.ndarray) -> bool:
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
        return bool(
            pos_arr[2] < Z_MIN - 0.5
            or pos_arr[2] > Z_MAX + 0.5
            or pos_arr[0] < X_MIN - 0.8
            or pos_arr[0] > X_MAX + 0.8
            or abs(float(pos_arr[1]) - Y_CANVAS) > Y_BOUND_M
        )

    def _compute_reward_phase_a(
        self,
        pos: np.ndarray,
        p_ref: np.ndarray,
        led_ref: float,
        brightness: float,
        u_final: np.ndarray,
        prev_u_final: np.ndarray,
        out_of_bounds: bool,
        delta_u_rl: np.ndarray,
        prev_delta_u_rl: Optional[np.ndarray],
        new_target: float,
        off_target: float,
        repaint: float,
        prev_brightness: float,
        coverage: float,
        completion_due: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Path/paint/smoothness reward shared by Phase A and B."""
        schedule_err = float(np.linalg.norm(pos - p_ref))
        path_dist, _, _ = self._nearest_path_stats(pos)
        t_ref = float(self._episode_step) * DT
        corner_influence, corner_sharpness, corner_dist = self._scheduled_corner_context(t_ref)
        speed = float(np.linalg.norm(self._vel))
        ref_speed = V_REF
        v_ref = self._interpolate_traj_velocity(float(self._episode_step) * DT * V_REF)
        return compute_lightpaint_reward(
            path_dist=path_dist,
            schedule_err=schedule_err,
            led_ref=led_ref,
            brightness=brightness,
            new_target=float(new_target),
            off_target=float(off_target),
            repaint=float(repaint),
            command=np.asarray(u_final, dtype=np.float32),
            prev_command=np.asarray(prev_u_final, dtype=np.float32),
            delta_v=delta_u_rl,
            prev_delta_v=prev_delta_u_rl,
            prev_brightness=prev_brightness,
            corner_influence=corner_influence,
            corner_sharpness=corner_sharpness,
            corner_dist=corner_dist,
            speed=speed,
            ref_speed=ref_speed,
            v_ref=v_ref,
            coverage=coverage,
            completion_due=completion_due,
            out_of_bounds=out_of_bounds,
        )

    def _apply_physics(self, u_final: np.ndarray, wind_force: np.ndarray) -> None:
        """
        Apply double-integrator physics with explicit velocity command + wind force.

        Args:
            u_final: 3-axis commanded velocity in m/s (already in physical units).
            wind_force: 3-axis world-frame disturbance force in N (acceleration ~ F/m·DT).
        """
        u = np.asarray(u_final, dtype=np.float32).reshape(3)
        wf = np.asarray(wind_force, dtype=np.float32).reshape(3)

        # Convert wind force to per-step velocity perturbation
        wind_dv = (wf / MASS) * DT

        # First-order lag for velocity command following (tau ~ 3 steps)
        tau = 3.0
        alpha_vel = 1.0 / tau
        self._vel[0] = self._vel[0] * VEL_DAMP + alpha_vel * (u[0] - self._vel[0]) + wind_dv[0]
        self._vel[1] = self._vel[1] * VEL_DAMP + alpha_vel * (u[1] - self._vel[1]) + wind_dv[1]
        self._vel[2] = self._vel[2] * VEL_DAMP + alpha_vel * (u[2] - self._vel[2]) + wind_dv[2]
        self._vel = np.clip(self._vel, -MAX_VEL, MAX_VEL)

        self._pos = self._pos + self._vel * DT

        # Clip position to arena bounds (with slight padding)
        self._pos[0] = float(np.clip(self._pos[0], X_MIN - 0.3, X_MAX + 0.3))
        self._pos[1] = float(np.clip(self._pos[1], -0.5, 0.5))
        self._pos[2] = float(np.clip(self._pos[2], Z_MIN - 0.2, Z_MAX + 0.2))

        # Simplified attitude update
        self._rpy[0] = float(np.clip(self._vel[1] * 0.1, -0.3, 0.3))
        self._rpy[1] = float(np.clip(-self._vel[0] * 0.1, -0.3, 0.3))
        self._ang_vel[0] = self._ang_vel[0] * ANG_DAMP
        self._ang_vel[1] = self._ang_vel[1] * ANG_DAMP
        self._ang_vel[2] = self._ang_vel[2] * ANG_DAMP

    # ------------------------------------------------------------------
    # Phase dispatch hooks
    # ------------------------------------------------------------------

    def _build_action_space(self, phase: str) -> spaces.Space:
        """Phase-dependent action space declaration."""
        if phase == "A":
            # No RL action; train_phase_a_pid passes np.zeros(0).
            return spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)
        if phase == "B":
            # Δu_RL residual on velocity, scaled by RESIDUAL_DELTA_MAX inside step()
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        if phase == "C_discrete":
            return spaces.Discrete(2)
        if phase == "C_continuous":
            return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        raise ValueError(f"Unknown phase: {phase!r}")

    def _extract_residual(self, action: np.ndarray) -> np.ndarray:
        """Extract Δu_RL (3,) from raw action. Phase A returns zeros."""
        if self.phase == "A":
            return np.zeros(3, dtype=np.float32)
        if self.phase == "B":
            arr = np.asarray(action, dtype=np.float32).flatten()
            if arr.shape != (4,):
                raise ValueError(f"Phase B action shape {arr.shape} != (4,).")
            if not np.all(np.isfinite(arr)):
                raise ValueError("Phase B action contains NaN or inf.")
            arr = np.clip(arr, -1.0, 1.0)
            return (arr[0:3] * RESIDUAL_DELTA_MAX).astype(np.float32)
        if self.phase in ("C_discrete", "C_continuous"):
            raise NotImplementedError(
                "_extract_residual supports only Phase B in the active environment."
            )
        raise ValueError(f"Unknown phase: {self.phase!r}")

    def _extract_led_residual(self, action: np.ndarray) -> float:
        if self.phase != "B":
            return 0.0
        arr = np.asarray(action, dtype=np.float32).flatten()
        if arr.shape != (4,):
            raise ValueError(f"Phase B action shape {arr.shape} != (4,).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Phase B action contains NaN or inf.")
        return float(np.clip(arr[3], -1.0, 1.0) * LED_RESIDUAL_SCALE)

    def _compute_obs(self) -> Dict[str, np.ndarray]:
        """
        Compute the Dict observation used by the policy.

        Returns dict with keys: drone_state (12,), future_ref (45,),
        target_mask (1,64,64), progress_mask (1,64,64).
        """
        # Normalized 12-dim drone state: pos(3) + vel(3) + rpy(3) + ang_vel(3)
        pos_norm = self._pos / POS_NORM
        vel_norm = self._vel / VEL_NORM
        rpy_norm = self._rpy / ATT_NORM
        ang_norm = self._ang_vel / ANG_NORM
        drone_state = np.concatenate([pos_norm, vel_norm, rpy_norm, ang_norm]).astype(np.float32)

        # 45-dim future reference (15 pts × 3 axes, flattened)
        future_ref = self._get_future_ref(self._episode_step).flatten().astype(np.float32)

        # Target mask: static (1, 64, 64)
        target_mask = self.G_letter[None, :, :].astype(np.float32)

        # Progress mask: cumulative painted (1, 64, 64)
        progress_mask = self.cumulative[None, :, :].astype(np.float32)

        return {
            "drone_state":   drone_state,
            "future_ref":    future_ref,
            "target_mask":   target_mask,
            "progress_mask": progress_mask,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Re-sample wind parameters for the new episode (per-episode constant)
        self.wind.reset(self._rng)
        # Reset PID derivative state
        self.pid.reset()

        # Spawn near start of reference trajectory with small noise
        start_pt = self._ref_waypoints[0] if len(self._ref_waypoints) > 0 else np.array(
            [0.0, 0.0, 1.5], dtype=np.float32
        )
        noise = self._rng.uniform(-self.init_box_size, self.init_box_size, size=3).astype(np.float32)
        self._pos = start_pt.copy() + noise
        self._pos[1] = 0.0  # keep on canvas plane
        self._vel = np.zeros(3, dtype=np.float32)
        self._rpy = np.zeros(3, dtype=np.float32)
        self._ang_vel = np.zeros(3, dtype=np.float32)
        self._prev_u_final = np.zeros(3, dtype=np.float32)
        self._prev_delta_u_rl = None
        self._prev_brightness = 0.0
        self._prev_led = 0
        self._completion_reward_given = False
        self._episode_step = 0
        self.cumulative = np.zeros((64, 64), dtype=np.float32)
        self._prev_pos = self._pos.copy()

        obs = self._compute_obs()
        info: Dict[str, Any] = {
            "pos": self._pos.tolist(),
            "brightness": 0.0,
            "led_on": False,
            "painted_px": 0,
            "tracking_err": 0.0,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step for Phase A or Phase B.

        Phase A: action.shape = (0,). Phase B: action.shape = (4,).
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        self._prev_pos = self._pos.copy()
        t = float(self._episode_step) * DT

        # 1. Reference position + velocity at the current arc-length
        s_now = t * V_REF
        p_ref = self._interpolate_traj(s_now)
        v_ref = self._interpolate_traj_velocity(s_now)

        # 2. RL residual (Phase A: zeros)
        delta_u_rl = self._extract_residual(action)
        delta_led = self._extract_led_residual(action)
        target_vel = (v_ref + delta_u_rl).astype(np.float32)

        # 3. PID command. Phase B residual modifies the PID target velocity.
        u_pid = self.pid.compute(p_ref, target_vel, self._pos, self._vel)
        u_final = u_pid.copy()

        # 4. Wind disturbance + physics
        wind_force = self.wind.step(t)
        self._apply_physics(u_final, wind_force)

        # 5. LED decision
        if self.led_always_on:
            led_ref = 1.0
        else:
            led_ref = float(self.led_strategy.decide(self._pos, np.zeros(0, dtype=np.float32), self.G_letter))
        prev_brightness = float(self._prev_brightness)
        brightness = led_brightness_from_ref(led_ref, delta_led)
        led_on = is_led_on(brightness)

        # 6. Stamp paint into cumulative mask
        paint_brightness = brightness if led_on else 0.0
        new_target, off_target, repaint = self._paint_stats(self._pos, paint_brightness)
        if led_on:
            self._stamp_led_progress(self._pos, brightness=brightness)
        reference_done = self._reference_done_for_reward(t)
        coverage = self._paint_coverage()
        completion_due = bool(reference_done and not self._completion_reward_given)

        # 7. Termination check
        out_of_bounds = self._is_out_of_bounds(self._pos)

        # 8. Reward (phase-specific)
        if self.phase in ("A", "B"):
            reward, reward_components = self._compute_reward_phase_a(
                self._pos,
                p_ref,
                led_ref,
                brightness,
                u_final,
                self._prev_u_final,
                out_of_bounds,
                delta_u_rl,
                self._prev_delta_u_rl,
                new_target,
                off_target,
                repaint,
                prev_brightness,
                coverage,
                completion_due,
            )
        else:
            raise NotImplementedError(
                f"step() reward path for phase={self.phase!r} is not implemented."
            )

        # 9. Advance counters
        self._episode_step += 1
        if completion_due:
            self._completion_reward_given = True
        self._prev_u_final = u_final.copy()
        self._prev_delta_u_rl = delta_u_rl.copy()
        self._prev_brightness = float(brightness)
        self._prev_led = int(led_on)

        terminated = bool(out_of_bounds)
        truncated = bool(self._episode_step >= self.max_episode_steps)

        obs = self._compute_obs()
        tracking_err = float(np.linalg.norm(self._pos - p_ref))
        info: Dict[str, Any] = {
            "pos": self._pos.tolist(),
            "p_ref": p_ref.tolist(),
            "v_ref": v_ref.tolist(),
            "delta_v": delta_u_rl.tolist(),
            "target_vel": target_vel.tolist(),
            "u_pid": u_pid.tolist(),
            "u_final": u_final.tolist(),
            "wind_force": np.asarray(wind_force, dtype=np.float32).tolist(),
            "wind_frame": "WORLD_FRAME",
            "led_ref": float(led_ref),
            "delta_led": float(delta_led),
            "brightness": float(brightness),
            "led_on": bool(led_on),
            "painted_px": int(self.cumulative.sum()),
            "tracking_err": tracking_err,
            "out_of_bounds": bool(out_of_bounds),
            "r": reward,  # for ep_info_buffer via Monitor
            **reward_components,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_pos(self) -> np.ndarray:
        """Return current drone position (3,). Used by environment checks."""
        return self._pos.copy()

    def close(self) -> None:
        """Cleanup (no pybullet physics server to disconnect)."""
        pass

    def render(self) -> Optional[np.ndarray]:
        """Return RGB array of target+progress masks composited (for visualization)."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :, 1] = (self.G_letter * 200).astype(np.uint8)      # Green = target
        rgb[:, :, 0] = (self.cumulative * 255).astype(np.uint8)     # Red = painted
        return rgb
