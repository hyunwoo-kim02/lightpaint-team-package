from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.train.visualize_flight import save_phase_a_visualization


def test_save_phase_a_visualization_removes_stale_frames(tmp_path):
    frames_dir = tmp_path / "phase_a_L_M0_frames"
    frames_dir.mkdir()
    stale = frames_dir / "frame_9999.png"
    stale.write_bytes(b"stale")

    positions = [np.array([-0.2 + 0.02 * i, 0.0, 1.4], dtype=np.float32) for i in range(5)]
    refs = [np.array([-0.15 + 0.02 * i, 0.0, 1.45], dtype=np.float32) for i in range(5)]
    brightness = [1.0] * len(positions)
    target_mask = np.ones((64, 64), dtype=np.float32)

    save_phase_a_visualization(
        pos_list=positions,
        brightness_list=brightness,
        target_mask=target_mask,
        out_dir=tmp_path,
        label="L",
        wind_mode="M0",
        fps=10,
        ref_pos_list=refs,
    )

    assert not stale.exists()
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    assert len(frame_files) == len(positions)
