"""Bundled-demo registry contract."""

from __future__ import annotations

from legged_manipulation_unilab.demo import (
    DEMO_REGISTRY,
    checkpoint_root,
    get_demo_spec,
    resolve_demo_checkpoint,
)


def test_demo_registry_specs() -> None:
    assert get_demo_spec("maniploco-ppo").algo == "ppo"
    assert get_demo_spec("maniploco-ppo").sim == "mujoco"
    assert get_demo_spec("maniploco-him").algo == "him_ppo"


def test_bundled_demo_checkpoints_exist() -> None:
    for name in DEMO_REGISTRY:
        demo_dir = resolve_demo_checkpoint(name)
        assert (demo_dir / "run_config.json").is_file(), name
        assert list(demo_dir.glob("model_*.pt")), name


def test_checkpoint_root_inside_package() -> None:
    assert checkpoint_root().is_dir()
