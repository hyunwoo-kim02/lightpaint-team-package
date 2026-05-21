"""PyBullet-free geometry constants and pixel/world conversion helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np

V_REF = 0.5
DT = 1.0 / 30.0

X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.5, 2.5
Y_CANVAS = 0.0
Y_BOUND_M = 0.45

LED_STAMP_RADIUS_PX = 1


def world_to_pixel(x: float, z: float, size: int = 64) -> Tuple[int, int]:
    col = int(np.clip((x - X_MIN) / (X_MAX - X_MIN) * (size - 1), 0, size - 1))
    row = int(np.clip((Z_MAX - z) / (Z_MAX - Z_MIN) * (size - 1), 0, size - 1))
    return col, row


def pixel_to_world(col: int, row: int, size: int = 64) -> Tuple[float, float]:
    x = X_MIN + (col / (size - 1)) * (X_MAX - X_MIN)
    z = Z_MAX - (row / (size - 1)) * (Z_MAX - Z_MIN)
    return x, z
