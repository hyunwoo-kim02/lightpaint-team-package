"""
pid_controller.py - VelocityPID for LightPaintAviaryW1.

Pure-numpy PD-velocity controller used by Phase A.
Returns a velocity command in m/s clipped to +-max_vel per axis.
"""
from __future__ import annotations

import numpy as np


class VelocityPID:
    """PD-velocity controller producing a 3-axis velocity command in m/s."""

    def __init__(self, Kp: float = 2.0, Kd: float = 0.4, max_vel: float = 2.0) -> None:
        self.Kp = float(Kp)
        self.Kd = float(Kd)
        self.max_vel = float(max_vel)
        self._prev_err = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        """Clear any latched derivative state. Call once per episode."""
        self._prev_err = np.zeros(3, dtype=np.float32)

    def compute(
        self,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Return velocity command u = clip(Kp * (p_ref - p) + Kd * (v_ref - v))."""
        p_ref = np.asarray(p_ref, dtype=np.float32).reshape(3)
        v_ref = np.asarray(v_ref, dtype=np.float32).reshape(3)
        p = np.asarray(p, dtype=np.float32).reshape(3)
        v = np.asarray(v, dtype=np.float32).reshape(3)

        err_p = p_ref - p
        err_v = v_ref - v
        u = self.Kp * err_p + self.Kd * err_v
        self._prev_err = err_p
        return np.clip(u, -self.max_vel, self.max_vel).astype(np.float32)
