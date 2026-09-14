"""Manager-Based contract tests for Go2ArmManipLoco."""

from __future__ import annotations

import importlib
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH
from legged_manipulation_unilab.cli import compose_config

_GO2_ARM_MANIP_LOCO_MODULE = "legged_manipulation_unilab.tasks.go2_arm.manip_loco"
_REGISTRY_MODULE = "unilab.base.registry"


def _registry_module():
    return importlib.import_module(_REGISTRY_MODULE)


def _ensure_registered() -> None:
    registry = _registry_module()
    registry.ensure_registries(packages=["legged_manipulation_unilab.tasks"])
    if not registry.contains("Go2ArmManipLoco"):
        module = sys.modules.get(_GO2_ARM_MANIP_LOCO_MODULE)
        if module is None:
            importlib.import_module(_GO2_ARM_MANIP_LOCO_MODULE)
        else:
            importlib.reload(module)


def _reward_override() -> dict:
    return {
        "tracking_lin_vel": {
            "func": "legged_manipulation_unilab.tasks.go2_arm.manager_env.Go2ArmReward",
            "weight": 1.0,
            "params": {"name": "tracking_lin_vel"},
        }
    }


def _make_env(num_envs: int = 2):
    _ensure_registered()
    registry = _registry_module()
    return registry.make(
        "Go2ArmManipLoco",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={
            "rewards": _reward_override(),
            "reward_parameters": {"tracking_sigma": 0.25, "base_height_target": 0.3},
            "domain_rand": {
                "randomize_base_mass": False,
                "randomize_body_mass": False,
                "random_com": False,
                "randomize_ground_friction": False,
                "randomize_dof_armature": False,
                "randomize_kp": False,
                "randomize_kd": False,
                "push_robots": False,
            },
        },
    )


def test_go2_arm_manip_loco_registered_manager_based():
    _ensure_registered()
    registry = _registry_module()
    assert registry.contains("Go2ArmManipLoco")
    meta = registry._envs["Go2ArmManipLoco"]
    assert meta.support_sim_backend("mujoco")
    assert meta.support_sim_backend("motrix")
    assert registry.resolve_reward_override_field("Go2ArmManipLoco") == "rewards"


def test_go2_arm_config_uses_manager_terms():
    _ensure_registered()
    registry = _registry_module()
    cfg = registry.materialize_env_config("Go2ArmManipLoco")
    assert hasattr(cfg, "observations")
    assert hasattr(cfg, "actions")
    assert hasattr(cfg, "events")
    assert cfg.policy_observation_group == "actor"
    assert cfg.critic_observation_group == "critic"

    cfg.domain_rand = {"randomize_body_mass": True, "body_mass_multiplier_range": [0.9, 1.1]}
    body_names = cfg._build_events()["body_mass"].params["asset_cfg"].body_names
    assert "arm_base_on_go2" not in body_names
    assert {"base", "FL_hip", "arm_base", "link1", "link6"} <= set(body_names)


def test_go2_arm_robot_actuators_have_unique_names():
    robot = ET.parse(
        ASSETS_ROOT_PATH / "robots/go2_arm/go2_with_arm_mjx_full_collision.xml"
    ).getroot()
    names = [element.attrib["name"] for element in robot.findall("./actuator/*")]
    assert len(names) == 18
    assert len(set(names)) == len(names)
    assert all(names)


@pytest.mark.slow
def test_go2_arm_playback_oracle_is_visual_model():
    """Discarded-visual physics models must not be used for video rendering."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unisim.backend.mujoco.playback import resolve_render_play_model_files

    env = _make_env(num_envs=1)
    try:
        with tempfile.TemporaryDirectory(prefix="go2-arm-playback-") as tmp_dir:
            model_file = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
        model_files = [model_file] if isinstance(model_file, str) else list(model_file)
        assert len(model_files) == 1
        assert model_files[0].endswith("scene_flat.xml")
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_mba_reset_step_contract():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2)
    try:
        assert env.obs_groups_spec == {"obs": 380, "critic": 79}
        assert env.action_space.shape == (18,)
        obs, _ = env.reset()
        assert obs["obs"].shape == (2, 380)
        state = env.step(np.zeros((2, 18), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert state.obs["obs"].shape == (2, 380)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_motrix_default_domain_randomization_reset_step_contract():
    pytest.importorskip("motrixsim", reason="Motrix backend not installed")
    from unilab.base.config_adapter import create_env
    from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override

    cfg = compose_config("ppo", "motrix", [])
    cfg.algo.num_envs = 2
    env = create_env(cfg, num_envs=2, env_cfg_override=build_ppo_env_cfg_override(cfg))
    try:
        obs, _ = env.reset()
        state = env.step(np.zeros((2, 18), dtype=np.float32))
        assert np.isfinite(obs["obs"]).all()
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.obs["obs"]).all()
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_him_effort_reward_terms_use_physics_state():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.config_adapter import create_env
    from unilab.scripts.train_rsl_rl import build_ppo_env_cfg_override

    cfg = compose_config("him_ppo", "mujoco", [])
    cfg.algo.num_envs = 2
    env = create_env(cfg, num_envs=2, env_cfg_override=build_ppo_env_cfg_override(cfg))
    try:
        env.reset()
        state = env.step(np.zeros((2, 18), dtype=np.float32))
        reward_log = state.info["log"]

        assert {
            "reward/dof_pos_limits",
            "reward/torques",
            "reward/energy",
            "reward/dof_acc",
        } <= set(reward_log)
        assert np.isfinite(state.reward).all()
        assert reward_log["reward/torques"] < 0.0
        assert reward_log["reward/energy"] < 0.0
        assert reward_log["reward/dof_acc"] < 0.0
    finally:
        env.close()
