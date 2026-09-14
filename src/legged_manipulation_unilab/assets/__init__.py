"""Bundled robot assets; no network or Hugging Face access at runtime."""

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path

from filelock import FileLock

ASSETS_ROOT_PATH = Path(__file__).resolve().parent
_ROBOT_ASSETS_ROOT_PATH = ASSETS_ROOT_PATH / "robots"


def cache_root() -> Path:
    override = os.environ.get("LEGGED_MANIPULATION_ASSET_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "legged-manipulation-unilab"


def _cached_file_matches(source: Path, target: Path) -> bool:
    return target.is_file() and filecmp.cmp(source, target, shallow=False)


def ensure_assets(*, include_floor: bool = False) -> Path:
    """Copy packaged robot assets to a writable cache for XML materialization.

    The XML, meshes, and textures ship in Git and the wheel. A content-aware
    comparison repairs missing or corrupted cache files without a separate
    metadata file.
    """
    del include_floor  # The packaged robot tree includes the Motrix floor texture.
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".assets.lock")):
        for source in _ROBOT_ASSETS_ROOT_PATH.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(ASSETS_ROOT_PATH)
            target = root / relative
            if _cached_file_matches(source, target):
                continue
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
