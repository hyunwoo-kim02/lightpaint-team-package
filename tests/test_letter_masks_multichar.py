"""Tests for letter_masks multichar + case-preservation (Checkpoint 1 V0)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.letter_masks import (
    _MASK_CACHE,
    _WAYPOINT_CACHE,
    auto_fit_font_size,
    get_mask,
    render_letter,
)


def _clear_caches():
    _MASK_CACHE.clear()
    _WAYPOINT_CACHE.clear()


def test_multichar_renders_nonzero():
    _clear_caches()
    for label in ("RL", "DG", "CAT", "Pig"):
        mask = render_letter(label, size=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert mask.sum() > 50, f"{label!r} mask should have nonzero pixels"


def test_case_sensitivity_distinct_masks():
    _clear_caches()
    pig_lower = render_letter("Pig", size=64)
    pig_upper = render_letter("PIG", size=64)
    assert not np.array_equal(pig_lower, pig_upper), \
        "'Pig' and 'PIG' must produce different masks (case preserved)"


def test_get_mask_caches_case_sensitive():
    _clear_caches()
    pig_lower = get_mask("Pig")
    pig_upper = get_mask("PIG")
    assert "Pig" in _MASK_CACHE
    assert "PIG" in _MASK_CACHE
    assert _MASK_CACHE["Pig"] is pig_lower
    assert _MASK_CACHE["PIG"] is pig_upper


def test_auto_fit_picks_smaller_for_longer_text():
    """Longer text should pick a smaller font size (or equal at extreme)."""
    short_font = auto_fit_font_size("L", canvas_size=64, target_fill=0.85)
    long_font = auto_fit_font_size("CAT", canvas_size=64, target_fill=0.85)
    assert long_font.size <= short_font.size, \
        f"Expected smaller font for longer text, got short={short_font.size} long={long_font.size}"


def test_auto_fit_text_fits_within_canvas():
    """Auto-fitted text must produce an in-canvas bounding box."""
    from PIL import Image, ImageDraw
    canvas_size = 64
    target_fill = 0.85
    for label in ("RL", "DG", "CAT", "Pig"):
        font = auto_fit_font_size(label, canvas_size=canvas_size, target_fill=target_fill)
        img = Image.new("L", (canvas_size, canvas_size))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), label, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        assert w <= canvas_size * target_fill, f"{label!r} width {w} exceeds fit"
        assert h <= canvas_size * target_fill, f"{label!r} height {h} exceeds fit"


def test_single_char_still_works():
    """Backwards compatibility: single letters still render correctly."""
    _clear_caches()
    for letter in ("L", "T", "R", "I"):
        mask = render_letter(letter, size=64)
        assert mask.sum() > 50
