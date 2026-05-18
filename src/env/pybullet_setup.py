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

_APPLIED = False


def apply_korean_path_fix(mirror_dirname: str = "lightpaint_pybullet_data") -> str:
    """
    Mirror pybullet_data + gym_pybullet_drones/assets into %TEMP%/<mirror_dirname>
    and monkey-patch lookups to return the mirror path. Returns the mirror path.
    Idempotent — safe to call multiple times.
    """
    global _APPLIED
    ascii_dir = os.path.join(tempfile.gettempdir(), mirror_dirname)

    # 1. Copy pybullet_data → mirror
    import pybullet_data as _pbd
    pbd_source = _pbd.getDataPath()
    if not os.path.isdir(ascii_dir):
        shutil.copytree(pbd_source, ascii_dir)
    elif os.path.abspath(pbd_source) != os.path.abspath(ascii_dir):
        # A stale mirror can exist from a previous failed/import-interrupted run.
        # Keep it, but refresh missing pybullet_data files such as plane.urdf.
        shutil.copytree(pbd_source, ascii_dir, dirs_exist_ok=True)

    # 2. Copy gym_pybullet_drones/assets/* → mirror (overlay)
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
