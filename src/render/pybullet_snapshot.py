"""
pybullet_snapshot.py — External-camera snapshot for PyBullet DIRECT-mode runs.

Transcribes mini_script/teams/integration/code/render_pybullet_snapshots.py:75-94
with the same view/proj defaults so the camera framing matches the published
mini_script visualizations.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


SNAP_W, SNAP_H = 320, 320

DEFAULT_CAM_TARGET = (0.0, 0.0, 1.5)   # center of xz canvas
DEFAULT_DISTANCE = 2.0
DEFAULT_YAW = 30
DEFAULT_PITCH = -10
DEFAULT_FOV = 55


def capture_pybullet_snapshot(
    client_id: int,
    width: int = SNAP_W,
    height: int = SNAP_H,
    cam_target: tuple = DEFAULT_CAM_TARGET,
    distance: float = DEFAULT_DISTANCE,
    yaw: float = DEFAULT_YAW,
    pitch: float = DEFAULT_PITCH,
    roll: float = 0,
    fov: float = DEFAULT_FOV,
) -> np.ndarray:
    """
    Capture an external-camera RGB snapshot using ER_TINY_RENDERER (works in DIRECT mode).

    Returns:
        np.ndarray of shape (height, width, 3), dtype=uint8 (RGB).
    """
    import pybullet as p

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(cam_target),
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=float(width) / float(height), nearVal=0.1, farVal=10.0,
    )
    w, h, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    img = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    return img
