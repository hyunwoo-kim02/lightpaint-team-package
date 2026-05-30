"""Tests for wind_modes (Checkpoint 1 V0)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.wind_modes import (
    M1_FORCE_MAX_N,
    M1_FORCE_MIN_N,
    M2_FORCE_BASE_MAX_N,
    M2_FORCE_BASE_MIN_N,
    M2_FORCE_GUST_MAX_N,
    M2_FORCE_GUST_MIN_N,
    M0Wind,
    M1Wind,
    M2Wind,
    M3Wind,
    WindMode,
    make_wind_mode,
)


def test_m0_step_returns_zeros():
    rng = np.random.default_rng(0)
    wind = M0Wind()
    wind.reset(rng)
    for t in (0.0, 0.5, 5.0):
        f = wind.step(t)
        assert f.shape == (3,)
        assert f.dtype == np.float32
        assert np.allclose(f, 0.0)


def test_m1_is_nonzero_constant_and_seed_reproducible():
    rng = np.random.default_rng(0)
    wind = M1Wind()
    wind.reset(rng)
    f0 = wind.step(0.0)
    f1 = wind.step(5.0)
    assert f0.shape == (3,)
    assert f0.dtype == np.float32
    assert np.linalg.norm(f0) > 0.0
    assert M1_FORCE_MIN_N <= np.linalg.norm(f0) <= M1_FORCE_MAX_N
    np.testing.assert_allclose(f0, f1, atol=1e-7)

    wind_same = M1Wind()
    wind_same.reset(np.random.default_rng(0))
    np.testing.assert_allclose(f0, wind_same.step(0.0), atol=1e-7)


def test_m2_is_nonzero_time_varying_and_seed_reproducible():
    wind = M2Wind()
    wind.reset(np.random.default_rng(1))
    f0 = wind.step(0.0)
    f1 = wind.step(1.0)
    assert f0.shape == (3,)
    assert f0.dtype == np.float32
    assert np.linalg.norm(f0) > 0.0
    assert np.linalg.norm(f1) > 0.0
    assert np.linalg.norm(f0) <= M2_FORCE_BASE_MAX_N + M2_FORCE_GUST_MAX_N
    assert np.linalg.norm(f1) >= M2_FORCE_BASE_MIN_N
    assert not np.allclose(f0, f1)

    wind_same = M2Wind()
    wind_same.reset(np.random.default_rng(1))
    np.testing.assert_allclose(f0, wind_same.step(0.0), atol=1e-7)


def test_m3_raises_on_reset_and_step():
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError):
        M3Wind().reset(rng)
    with pytest.raises(NotImplementedError):
        M3Wind().step(0.0)


def test_wind_force_shapes_are_consistent():
    rng = np.random.default_rng(2)
    for cls in (M0Wind, M1Wind, M2Wind):
        wind = cls()
        wind.reset(rng)
        force = wind.step(0.25)
        assert force.shape == (3,)
        assert force.dtype == np.float32
        assert np.all(np.isfinite(force))
    for cls in (M3Wind,):
        with pytest.raises(NotImplementedError):
            cls().step(0.0)


def test_factory_returns_correct_subclass():
    rng = np.random.default_rng(0)
    assert isinstance(make_wind_mode("M0", rng), M0Wind)
    assert isinstance(make_wind_mode("M1", rng), M1Wind)
    assert isinstance(make_wind_mode("M2", rng), M2Wind)
    assert isinstance(make_wind_mode("M3", rng), M3Wind)


def test_factory_unknown_name_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_wind_mode("M9", rng)


def test_windmode_is_abstract():
    with pytest.raises(TypeError):
        WindMode()  # type: ignore[abstract]
