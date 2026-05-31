"""Wind disturbance modes for LightPaint environments.

Each mode supports:
    reset(rng)        : sample fresh per-episode parameters
    step(t) -> (3,)   : world-frame force vector in N at episode time t
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


M1_FORCE_MIN_N = 0.00003
M1_FORCE_MAX_N = 0.00008
M2_FORCE_BASE_MIN_N = 0.00002
M2_FORCE_BASE_MAX_N = 0.00005
M2_FORCE_GUST_MIN_N = 0.00002
M2_FORCE_GUST_MAX_N = 0.00006
M2_GUST_FREQ_MIN_HZ = 0.2
M2_GUST_FREQ_MAX_HZ = 1.0


def _sample_horizontal_unit(rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)


class WindMode(ABC):
    """Abstract base for an episode-scoped wind disturbance generator."""

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Sample a fresh parameter set for a new episode."""

    @abstractmethod
    def step(self, t: float) -> np.ndarray:
        """Return the (3,) world-frame force vector in N at episode time t."""


class M0Wind(WindMode):
    """No disturbance, used for PID sanity."""

    def reset(self, rng: np.random.Generator) -> None:
        return None

    def step(self, t: float) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class M1Wind(WindMode):
    """Constant world-frame force with sampled horizontal direction and magnitude."""

    def __init__(self) -> None:
        self.force = np.zeros(3, dtype=np.float32)

    def reset(self, rng: np.random.Generator) -> None:
        direction = _sample_horizontal_unit(rng)
        magnitude = float(rng.uniform(M1_FORCE_MIN_N, M1_FORCE_MAX_N))
        self.force = (direction * magnitude).astype(np.float32)

    def step(self, t: float) -> np.ndarray:
        return self.force.copy()


class M2Wind(WindMode):
    """Constant world-frame direction with sinusoidal nonzero gust magnitude."""

    def __init__(self) -> None:
        self.direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.base_n = 0.03
        self.gust_n = 0.03
        self.freq_hz = 0.5
        self.phase_rad = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        self.direction = _sample_horizontal_unit(rng)
        self.base_n = float(rng.uniform(M2_FORCE_BASE_MIN_N, M2_FORCE_BASE_MAX_N))
        self.gust_n = float(rng.uniform(M2_FORCE_GUST_MIN_N, M2_FORCE_GUST_MAX_N))
        self.freq_hz = float(rng.uniform(M2_GUST_FREQ_MIN_HZ, M2_GUST_FREQ_MAX_HZ))
        self.phase_rad = float(rng.uniform(0.0, 2.0 * np.pi))

    def step(self, t: float) -> np.ndarray:
        wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * self.freq_hz * float(t) + self.phase_rad))
        magnitude = self.base_n + self.gust_n * float(wave)
        return (self.direction * magnitude).astype(np.float32)


_REGISTRY = {"M0": M0Wind, "M1": M1Wind, "M2": M2Wind}


def make_wind_mode(name: str, rng: np.random.Generator) -> WindMode:
    """Construct a WindMode by name and reset it with the supplied rng."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown wind mode: {name!r}. Choose from {list(_REGISTRY)}.")
    mode = _REGISTRY[name]()
    mode.reset(rng)
    return mode
