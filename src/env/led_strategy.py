"""
led_strategy.py — LED on/off (or brightness) decision strategies.
Plan reference: joyful-painting-leaf.md §"신규 — led_strategy.py".

Phase scope:
- ScriptedLED: functional (Phase A/B) — looks up target_mask at the drone's
  current pixel and emits brightness 1.0 if on-target, else 0.0.
- LearnedDiscreteLED / LearnedContinuousLED: skeleton with NotImplementedError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# Re-import the world_to_pixel function locally to avoid a circular dependency
# with light_paint_aviary_w1.py. Constants are duplicated here from SRS §4.6.
_X_MIN, _X_MAX = -1.0, 1.0
_Z_MIN, _Z_MAX = 0.5, 2.5


def _world_to_pixel(x: float, z: float, size: int = 64) -> tuple[int, int]:
    col = int(np.clip((x - _X_MIN) / (_X_MAX - _X_MIN) * (size - 1), 0, size - 1))
    row = int(np.clip((_Z_MAX - z) / (_Z_MAX - _Z_MIN) * (size - 1), 0, size - 1))
    return col, row


class LEDStrategy(ABC):
    """Decide LED brightness in [0, 1] given drone pose, action, and target mask."""

    @abstractmethod
    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        """Return brightness in [0.0, 1.0]."""


class ScriptedLED(LEDStrategy):
    """LED ON whenever the drone hovers over an on-target pixel."""

    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        col, row = _world_to_pixel(float(pos[0]), float(pos[2]))
        return 1.0 if float(target_mask[row, col]) > 0.5 else 0.0


class LearnedDiscreteLED(LEDStrategy):
    """[next plan] action[0] > 0 → ON, else OFF."""

    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        raise NotImplementedError(
            "LearnedDiscreteLED is scheduled for Phase C; only ScriptedLED is wired in Checkpoint 1."
        )


class LearnedContinuousLED(LEDStrategy):
    """[next plan] brightness = clip(action[0], 0, 1)."""

    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        raise NotImplementedError(
            "LearnedContinuousLED is scheduled for Phase C; only ScriptedLED is wired in Checkpoint 1."
        )


def make_led_strategy(phase: str, target_mask: Optional[np.ndarray] = None) -> LEDStrategy:
    """Factory: phase ∈ {'A', 'B', 'C_discrete', 'C_continuous'}."""
    if phase in ("A", "B"):
        return ScriptedLED()
    if phase == "C_discrete":
        return LearnedDiscreteLED()
    if phase == "C_continuous":
        return LearnedContinuousLED()
    raise ValueError(f"Unknown phase: {phase!r}")
