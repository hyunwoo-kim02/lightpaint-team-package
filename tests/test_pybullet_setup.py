from __future__ import annotations

import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env import pybullet_setup


def test_base_temp_dir_prefers_ascii_candidate(monkeypatch):
    ascii_dir = os.path.abspath(r"C:\Users\Codex\AppData\Local")
    monkeypatch.setattr(pybullet_setup.tempfile, "gettempdir", lambda: "E:/nonascii/\uD55C")
    monkeypatch.setenv("LOCALAPPDATA", ascii_dir)
    monkeypatch.setenv("ProgramData", r"Z:\missing-programdata")
    monkeypatch.setattr(
        pybullet_setup.os.path,
        "isdir",
        lambda path: os.path.abspath(path) == ascii_dir,
    )

    assert pybullet_setup._base_temp_dir() == ascii_dir


def test_copy_asset_mirror_replaces_partial_directory(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "mirror"
    src.mkdir()
    dst.mkdir()
    (src / "plane.urdf").write_text("plane", encoding="utf-8")
    (dst / "partial.txt").write_text("partial", encoding="utf-8")

    pybullet_setup._copy_asset_mirror(str(src), str(dst))

    assert (dst / "plane.urdf").read_text(encoding="utf-8") == "plane"
    assert not (dst / "partial.txt").exists()
