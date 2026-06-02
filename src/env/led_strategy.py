"""
led_strategy.py - LED on/off (or brightness) decision strategies.

Phase scope:
- ScriptedLED: functional (Phase A/B) - looks up target_mask at the drone's
  current pixel and emits brightness 1.0 if on-target, else 0.0.
- LearnedDiscreteLED / LearnedContinuousLED: reserved command interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# Re-import the world_to_pixel function locally to avoid a circular dependency.
# These bounds match the XZ canvas used by the light-painting environments.
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
    """Discrete LED command interface: action[0] > 0 means ON, else OFF."""

    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        raise NotImplementedError(
            "LearnedDiscreteLED is not wired into the active environment."
        )


class LearnedContinuousLED(LEDStrategy):
    """Continuous LED command interface: brightness = clip(action[0], 0, 1)."""

    def decide(
        self,
        pos: np.ndarray,
        action: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        raise NotImplementedError(
            "LearnedContinuousLED is not wired into the active environment."
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
