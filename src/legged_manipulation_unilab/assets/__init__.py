"""Bundled robot assets; no network or Hugging Face access at runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from filelock import FileLock

ASSETS_ROOT_PATH = Path(__file__).resolve().parent


def cache_root() -> Path:
    override = os.environ.get("LEGGED_MANIPULATION_ASSET_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    digest = hashlib.sha256((ASSETS_ROOT_PATH / "manifest.json").read_bytes()).hexdigest()[:16]
    return root / "legged-manipulation-unilab" / digest


def ensure_assets(*, include_floor: bool = False) -> Path:
    """Copy bundled assets to a writable cache for XML materialization tools.

    All meshes, textures and XML ship in Git and the wheel. The local cache
    allows scene tools to create temporary XML even with read-only site-packages.
    Every listed file is checked for existence, so a partial cache is repaired.
    """
    manifest = json.loads((ASSETS_ROOT_PATH / "manifest.json").read_text())
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".assets.lock")):
        for relative in manifest["sha256"]:
            source, target = ASSETS_ROOT_PATH / relative, root / relative
            if (
                not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"][relative]
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
    return root


def resolve_scene(path: str | Path | None = None) -> str:
    """Materialize the bundled scene; preserve explicitly supplied custom scenes."""
    bundled = ASSETS_ROOT_PATH / "robots/go2_arm/scene_flat.xml"
    if path is not None and Path(path).resolve() != bundled:
        return str(path)
    return str(ensure_assets() / "robots/go2_arm/scene_flat.xml")


def main() -> None:
    print(ensure_assets())
