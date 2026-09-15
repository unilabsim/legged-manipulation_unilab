"""HIM-PPO playback-path contract tests: task-stage restore and overlays."""

from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf
from unisim.backend.base import DebugPrimitive

from legged_manipulation_unilab.training.him import (
    _restore_trained_arm_stage,
    build_play_debug_overlays,
)

_STAGE_ONE = {
    "freeze_arm_joints": True,
    "disable_ee_goal_trajectory": True,
    "fixed_ee_goal_cart": [0.3, 0.0, 0.25],
}
_FULL_MODE = {
    "freeze_arm_joints": False,
    "disable_ee_goal_trajectory": False,
    "fixed_ee_goal_cart": [0.3, 0.0, 0.25],
}


def _stage_one_cfg():
    return OmegaConf.create({"env": {"arm_stage": dict(_STAGE_ONE)}})


def _run_dir(tmp_path, arm_stage: dict):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"config": {"env": {"arm_stage": arm_stage}}})
    )
    return run_dir


def test_restore_trained_arm_stage_from_run_config(tmp_path):
    cfg = _stage_one_cfg()
    _restore_trained_arm_stage(cfg, _run_dir(tmp_path, _FULL_MODE))
    assert cfg.env.arm_stage.freeze_arm_joints is False
    assert cfg.env.arm_stage.disable_ee_goal_trajectory is False


def test_restore_without_run_config_keeps_cfg(tmp_path):
    cfg = _stage_one_cfg()
    _restore_trained_arm_stage(cfg, tmp_path)
    assert cfg.env.arm_stage.freeze_arm_joints is True


def test_restore_matching_stage_is_noop(tmp_path):
    cfg = _stage_one_cfg()
    _restore_trained_arm_stage(cfg, _run_dir(tmp_path, _STAGE_ONE))
    assert cfg.env.arm_stage.freeze_arm_joints is True


def test_build_play_overlays_mark_goal_and_command_arrow():
    goal = np.array([[1.0, 0.2, 0.3]])
    base = np.array([[0.0, 0.0, 0.278]])
    identity = np.array([[1.0, 0.0, 0.0, 0.0]])

    overlays = build_play_debug_overlays(
        vel_commands=np.array([[0.5, 0.0, 0.0]]),
        ee_goal_world=goal,
        base_pos_w=base,
        base_quat_w=identity,
    )
    assert [p.kind for p in overlays[0]] == ["sphere", "arrow"]
    sphere, arrow = overlays[0]
    assert isinstance(sphere, DebugPrimitive)
    assert sphere.pos == (1.0, 0.2, 0.3)
    w, x, y, z = arrow.quat
    assert w == pytest.approx(0.7071, abs=1e-3)
    assert y == pytest.approx(0.7071, abs=1e-3)
    assert abs(x) < 1e-6 and abs(z) < 1e-6

    # Yawed 90°: a body-frame +x command must draw as a world +y arrow.
    yawed = build_play_debug_overlays(
        vel_commands=np.array([[0.5, 0.0, 0.0]]),
        ee_goal_world=goal,
        base_pos_w=base,
        base_quat_w=np.array([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]]),
    )
    w, x, y, z = yawed[0][1].quat
    assert w == pytest.approx(0.7071, abs=1e-3)
    assert x == pytest.approx(-0.7071, abs=1e-3)
    assert abs(y) < 1e-6 and abs(z) < 1e-6

    standing = build_play_debug_overlays(
        vel_commands=np.array([[0.0, 0.0, 0.4]]),
        ee_goal_world=goal,
        base_pos_w=base,
        base_quat_w=identity,
    )
    assert [p.kind for p in standing[0]] == ["sphere"]
