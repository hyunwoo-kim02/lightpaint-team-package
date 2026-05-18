"""
pybullet_setup.py — PYBULLET-PATH-FIX-1 module.
Plan reference: joyful-painting-leaf.md CP-2 (D7/D11).

Korean characters in the project path break PyBullet's URDF importer because
`pkg_resources.resource_filename` returns a path that PyBullet later
re-encodes through code that assumes ASCII. We mirror pybullet_data + the
gym-pybullet-drones assets directory into an ASCII path under %TEMP%, then
monkey-patch `pybullet_data.getDataPath` and `pkg_resources.resource_filename`
to redirect non-ASCII paths to that mirror.

Pattern transcribed from:
  mini_script/teams/integration/code/render_pybullet_snapshots.py:19-51
  lightpaint-team-package/reference/env_sanity_v4.py:26-59

Call apply_korean_path_fix() ONCE at module import time of any file that
imports gym_pybullet_drones. Idempotent: subsequent calls are no-ops.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid

_APPLIED = False


def _is_ascii_path(path: str) -> bool:
    try:
        os.fspath(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _base_temp_dir() -> str:
    candidates = [
        tempfile.gettempdir(),
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("ProgramData"),
        r"C:\Temp",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if _is_ascii_path(candidate) and os.path.isdir(candidate):
            return candidate
    fallback = os.path.join(os.getcwd(), ".pybullet_ascii_cache")
    os.makedirs(fallback, exist_ok=True)
    if not _is_ascii_path(fallback):
        raise RuntimeError(f"No ASCII-safe PyBullet cache directory is available: {fallback!r}")
    return fallback


def _acquire_lock(lock_path: str, timeout_s: float = 30.0):
    start = time.time()
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - start > timeout_s:
                try:
                    if time.time() - os.path.getmtime(lock_path) > timeout_s:
                        os.unlink(lock_path)
                        continue
                except FileNotFoundError:
                    continue
                raise TimeoutError(f"Timed out waiting for PyBullet asset mirror lock: {lock_path}")
            time.sleep(0.05)


def _copy_asset_mirror(src_dir: str, dst_dir: str) -> None:
    parent = os.path.dirname(dst_dir)
    tmp_dir = os.path.join(parent, f".{os.path.basename(dst_dir)}.{uuid.uuid4().hex}.tmp")
    shutil.copytree(src_dir, tmp_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.replace(tmp_dir, dst_dir)


def apply_korean_path_fix(mirror_dirname: str = "lightpaint_pybullet_data") -> str:
    """
    Mirror pybullet_data + gym_pybullet_drones/assets into %TEMP%/<mirror_dirname>
    and monkey-patch lookups to return the mirror path. Returns the mirror path.
    Idempotent — safe to call multiple times.
    """
    global _APPLIED
    ascii_dir = os.path.join(_base_temp_dir(), mirror_dirname)
    if not _is_ascii_path(ascii_dir):
        raise RuntimeError(f"PyBullet mirror path is not ASCII-safe: {ascii_dir!r}")

    # 1. Copy pybullet_data → mirror
    import pybullet_data as _pbd
    pbd_source = _pbd.getDataPath()
    lock_fd = _acquire_lock(ascii_dir + ".lock")
    try:
        if not os.path.isfile(os.path.join(ascii_dir, "plane.urdf")):
            _copy_asset_mirror(pbd_source, ascii_dir)
        elif os.path.abspath(pbd_source) != os.path.abspath(ascii_dir):
            shutil.copytree(pbd_source, ascii_dir, dirs_exist_ok=True)
        try:
            import gym_pybullet_drones as _gpd
            gpd_assets = os.path.join(os.path.dirname(_gpd.__file__), "assets")
            if os.path.isdir(gpd_assets):
                for fn in os.listdir(gpd_assets):
                    src = os.path.join(gpd_assets, fn)
                    dst = os.path.join(ascii_dir, fn)
                    if os.path.isfile(src) and not os.path.isfile(dst):
                        shutil.copy2(src, dst)
        except ImportError:
            pass
    finally:
        os.close(lock_fd)
        try:
            os.unlink(ascii_dir + ".lock")
        except FileNotFoundError:
            pass

    if _APPLIED:
        return ascii_dir

    # 3. Monkey-patch getDataPath → return ASCII mirror
    _pbd.getDataPath = lambda: ascii_dir  # type: ignore[method-assign]

    # 4. Monkey-patch pkg_resources.resource_filename to remap non-ASCII results
    try:
        import pkg_resources as _pkg
        _original = _pkg.resource_filename

        def _ascii_resource_filename(pkg_or_req, resource_name):
            result = _original(pkg_or_req, resource_name)
            try:
                result.encode("ascii")
                return result
            except UnicodeEncodeError:
                basename = os.path.basename(result)
                candidate = os.path.join(ascii_dir, basename)
                if os.path.exists(candidate):
                    return candidate
                return result

        _pkg.resource_filename = _ascii_resource_filename
    except ImportError:
        pass

    _APPLIED = True
    return ascii_dir
