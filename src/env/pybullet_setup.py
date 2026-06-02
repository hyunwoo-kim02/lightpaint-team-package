from __future__ import annotations

import os
import json
import shutil
import tempfile
import time
import uuid

_APPLIED = False
_SENTINEL_NAME = ".lightpaint_assets_ready.json"
_BASE_REQUIRED_FILES = ("plane.urdf",)
_GPD_REQUIRED_FILES = ("cf2x.urdf", "cf2.dae")


def _is_runtime_path(path: str) -> bool:
    return all(ord(ch) < 128 for ch in os.fspath(path))


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
        if _is_runtime_path(candidate) and os.path.isdir(candidate):
            return candidate
    cache_dir = os.path.join(os.getcwd(), ".pybullet_runtime_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if not _is_runtime_path(cache_dir):
        raise RuntimeError("PyBullet setup failed.")
    return cache_dir


def _acquire_lock(lock_path: str, timeout_s: float = 180.0):
    start = time.time()
    while True:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - start > timeout_s:
                try:
                    if time.time() - os.path.getmtime(lock_path) > timeout_s:
                        try:
                            os.unlink(lock_path)
                        except PermissionError:
                            time.sleep(0.1)
                            continue
                        continue
                except FileNotFoundError:
                    continue
                raise TimeoutError("PyBullet setup timed out.")
            time.sleep(0.05)


def _copy_asset_tree(src_dir: str, dst_dir: str) -> None:
    parent = os.path.dirname(dst_dir)
    tmp_dir = os.path.join(parent, f".{os.path.basename(dst_dir)}.{uuid.uuid4().hex}.tmp")
    shutil.copytree(src_dir, tmp_dir)
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.replace(tmp_dir, dst_dir)


def _sentinel_path(asset_dir: str) -> str:
    return os.path.join(asset_dir, _SENTINEL_NAME)


def _read_sentinel(asset_dir: str) -> dict | None:
    try:
        with open(_sentinel_path(asset_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _write_sentinel(asset_dir: str, signature: dict) -> None:
    os.makedirs(asset_dir, exist_ok=True)
    tmp_path = _sentinel_path(asset_dir) + f".{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(signature, f, sort_keys=True)
    os.replace(tmp_path, _sentinel_path(asset_dir))


def _assets_ready(asset_dir: str, signature: dict) -> bool:
    for rel_path in signature.get("required_files", _BASE_REQUIRED_FILES):
        if not os.path.isfile(os.path.join(asset_dir, rel_path)):
            return False
    return _read_sentinel(asset_dir) == signature


def _asset_signature(pbd_source: str) -> tuple[dict, str | None]:
    required_files = list(_BASE_REQUIRED_FILES)
    signature = {"pybullet_data": os.path.abspath(pbd_source)}
    gpd_assets = None
    try:
        import gym_pybullet_drones as _gpd

        candidate = os.path.join(os.path.dirname(_gpd.__file__), "assets")
        if os.path.isdir(candidate):
            gpd_assets = candidate
            signature["gym_pybullet_drones_assets"] = os.path.abspath(candidate)
            required_files.extend(_GPD_REQUIRED_FILES)
    except ImportError:
        pass
    signature["required_files"] = required_files
    return signature, gpd_assets


def _copy_gpd_assets(gpd_assets: str | None, asset_dir: str) -> None:
    if not gpd_assets:
        return
    for fn in os.listdir(gpd_assets):
        src = os.path.join(gpd_assets, fn)
        dst = os.path.join(asset_dir, fn)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)


def prepare_pybullet_assets(asset_dirname: str = "lightpaint_pybullet_data") -> str:
    global _APPLIED
    asset_dir = os.path.join(_base_temp_dir(), asset_dirname)
    if not _is_runtime_path(asset_dir):
        raise RuntimeError("PyBullet setup failed.")

    import pybullet_data as _pbd
    pbd_source = _pbd.getDataPath()
    signature, gpd_assets = _asset_signature(pbd_source)
    if not _assets_ready(asset_dir, signature):
        lock_fd = _acquire_lock(asset_dir + ".lock")
        try:
            if not _assets_ready(asset_dir, signature):
                if not os.path.isfile(os.path.join(asset_dir, "plane.urdf")):
                    _copy_asset_tree(pbd_source, asset_dir)
                else:
                    shutil.copytree(pbd_source, asset_dir, dirs_exist_ok=True)
                _copy_gpd_assets(gpd_assets, asset_dir)
                _write_sentinel(asset_dir, signature)
        finally:
            os.close(lock_fd)
            try:
                os.unlink(asset_dir + ".lock")
            except FileNotFoundError:
                pass

    if _APPLIED:
        return asset_dir

    _pbd.getDataPath = lambda: asset_dir  # type: ignore[method-assign]

    try:
        import pkg_resources as _pkg
        _original = _pkg.resource_filename

        def _resource_filename(pkg_or_req, resource_name):
            result = _original(pkg_or_req, resource_name)
            if _is_runtime_path(result):
                return result
            basename = os.path.basename(result)
            candidate = os.path.join(asset_dir, basename)
            if os.path.exists(candidate):
                return candidate
            return result

        _pkg.resource_filename = _resource_filename
    except ImportError:
        pass

    _APPLIED = True
    return asset_dir
