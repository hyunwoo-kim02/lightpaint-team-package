from __future__ import annotations

import numpy as np
import pytest

from src.env import lightpaint_geometry as geom
from src.env import light_paint_aviary_standalone as standalone


def test_standalone_geometry_matches_shared_helpers():
    assert standalone.X_MIN == geom.X_MIN
    assert standalone.X_MAX == geom.X_MAX
    assert standalone.Z_MIN == geom.Z_MIN
    assert standalone.Z_MAX == geom.Z_MAX
    assert standalone.Y_CANVAS == geom.Y_CANVAS
    assert standalone.Y_BOUND_M == geom.Y_BOUND_M
    assert standalone.LED_STAMP_RADIUS_PX == geom.LED_STAMP_RADIUS_PX
    assert standalone.world_to_pixel(0.0, 1.5) == geom.world_to_pixel(0.0, 1.5)
    np.testing.assert_allclose(standalone.pixel_to_world(32, 32), geom.pixel_to_world(32, 32))


def test_pybullet_geometry_matches_shared_helpers():
    pytest.importorskip("pybullet")
    pytest.importorskip("pybullet_data")
    pytest.importorskip("gym_pybullet_drones")
    from src.env import light_paint_aviary_pyb as pyb

    assert pyb.X_MIN == geom.X_MIN
    assert pyb.X_MAX == geom.X_MAX
    assert pyb.Z_MIN == geom.Z_MIN
    assert pyb.Z_MAX == geom.Z_MAX
    assert pyb.Y_CANVAS == geom.Y_CANVAS
    assert pyb.Y_BOUND_M == geom.Y_BOUND_M
    assert pyb.LED_STAMP_RADIUS_PX == geom.LED_STAMP_RADIUS_PX
    assert pyb.world_to_pixel(0.0, 1.5) == geom.world_to_pixel(0.0, 1.5)
    np.testing.assert_allclose(pyb.pixel_to_world(32, 32), geom.pixel_to_world(32, 32))
