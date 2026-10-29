"""Manager-Based contract tests for L1WArmManipLoco."""

from __future__ import annotations

import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH, resolve_scene
from legged_manipulation_unilab.cli import compose_config
from legged_manipulation_unilab.tasks.go2_arm.base import build_l1w_arm_position_gains
from legged_manipulation_unilab.tasks.go2_arm.manager_env import Go2ArmManipLocoCommand
from legged_manipulation_unilab.tasks.go2_arm.manip_loco import CommandsConfig
from legged_manipulation_unilab.tasks.l1_w_arm.manager_env import L1WArmReward
from legged_manipulation_unilab.tasks.l1_w_arm.manip_loco import L1WArmCommandsConfig

_L1W_MODULE = "legged_manipulation_unilab.tasks.l1_w_arm.manip_loco"
_REGISTRY_MODULE = "unilab.base.registry"


def _registry_module():
    return importlib.import_module(_REGISTRY_MODULE)


def _ensure_registered() -> None:
    registry = _registry_module()
    registry.ensure_registries(packages=["legged_manipulation_unilab.tasks"])
    if not registry.contains("L1WArmManipLoco"):
        module = sys.modules.get(_L1W_MODULE)
        if module is None:
            importlib.import_module(_L1W_MODULE)
        else:
            importlib.reload(module)


def _reward_override(reward_names: tuple[str, ...] = ("tracking_lin_vel",)) -> dict:
    return {
        name: {
            "func": "legged_manipulation_unilab.tasks.l1_w_arm.manager_env.L1WArmReward",
            "weight": 1.0,
            "params": {"name": name},
        }
        for name in reward_names
    }


_DISABLED_DOMAIN_RAND = {
    "randomize_base_mass": False,
    "randomize_body_mass": False,
    "random_com": False,
    "randomize_ground_friction": False,
    "randomize_dof_armature": False,
    "randomize_kp": False,
    "randomize_kd": False,
    "push_robots": False,
}


def _make_env(num_envs: int = 2, reward_names: tuple[str, ...] = ("tracking_lin_vel",)):
    _ensure_registered()
    registry = _registry_module()
    return registry.make(
        "L1WArmManipLoco",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={
            "rewards": _reward_override(reward_names),
            "reward_parameters": {"tracking_sigma": 0.25, "base_height_target": 0.38},
            "domain_rand": dict(_DISABLED_DOMAIN_RAND),
        },
    )


def test_l1_w_arm_manip_loco_registered_manager_based():
    _ensure_registered()
    registry = _registry_module()
    assert registry.contains("L1WArmManipLoco")
    meta = registry._envs["L1WArmManipLoco"]
    assert meta.support_sim_backend("mujoco")
    assert meta.support_sim_backend("motrix")
    assert registry.resolve_reward_override_field("L1WArmManipLoco") == "rewards"


def test_l1_w_arm_config_uses_manager_terms_and_22_dof_layout():
    _ensure_registered()
    registry = _registry_module()
    cfg = registry.materialize_env_config("L1WArmManipLoco")
    assert cfg.policy_observation_group == "actor"
    assert cfg.action_layout.dim == 22
    assert cfg.action_layout.wheel_slice == (12, 16)
    assert cfg.asset.base_name == "BASE_LINK"
    assert cfg.scene.default_keyframe_name == "home"
    gains = build_l1w_arm_position_gains(cfg.control_config)
    np.testing.assert_array_equal(gains["kp"][:12], 50.0)
    np.testing.assert_array_equal(gains["kd"][:12], 1.0)
    cfg.set_reward_terms({"tracking_lin_vel": 1.0})
    assert cfg.rewards["tracking_lin_vel"].func is L1WArmReward
    cfg.domain_rand = {"randomize_body_mass": True, "body_mass_multiplier_range": [0.9, 1.1]}
    body_names = cfg._build_events()["body_mass"].params["asset_cfg"].body_names
    assert "arm_base_on_l1" not in body_names
    assert {"BASE_LINK", "FL_HIP_LINK", "arm_base", "link1", "link6"} <= set(body_names)


def test_l1_w_arm_robot_actuators_have_unique_names_and_order():
    robot = ET.parse(ASSETS_ROOT_PATH / "robots/l1_w_arm/l1_w_with_arm.xml").getroot()
    names = [element.attrib["name"] for element in robot.findall("./actuator/*")]
    assert names[:12] == [
        "FBL_ABAD",
        "FBL_HIP",
        "FBL_KNEE",
        "FAR_ABAD",
        "FAR_HIP",
        "FAR_KNEE",
        "RAR_ABAD",
        "RAR_HIP",
        "RAR_KNEE",
        "RBL_ABAD",
        "RBL_HIP",
        "RBL_KNEE",
    ]
    assert names[12:16] == ["FBL_FOOT", "FAR_FOOT", "RAR_FOOT", "RBL_FOOT"]
    assert names[16:] == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert len(names) == 22
    assert len(set(names)) == len(names)


def test_l1_w_arm_resolve_scene_materializes_packaged_xml(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    packaged = ASSETS_ROOT_PATH / "robots/l1_w_arm/scene_flat.xml"
    resolved = resolve_scene(packaged)
    assert resolved.endswith("robots/l1_w_arm/scene_flat.xml")
    assert resolved != str(packaged)
    assert Path(resolved).is_file()


def test_l1_w_arm_owner_composition_identity():
    _ensure_registered()
    registry = _registry_module()
    cfg = compose_config("ppo", "mujoco", [], task="l1_w_arm_manip_loco")
    assert cfg.training.task_name == "L1WArmManipLoco"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.training.log_root == "logs/ppo-mujoco-l1w"
    cfg_him = compose_config("him_ppo", "mujoco", [], task="l1_w_arm_manip_loco")
    assert cfg_him.algo.num_one_step_obs == 88
    assert set(cfg_him.reward) == set(cfg.reward)
    assert len(cfg.reward) == 18
    assert cfg_him.env.command_config.gait_frequency == pytest.approx(1.8)
    assert cfg_him.env.arm_stage.freeze_arm_joints is False
    assert cfg_him.env.arm_stage.disable_ee_goal_trajectory is False
    him_task_cfg = registry.materialize_env_config("L1WArmManipLoco")
    him_task_cfg.domain_rand = dict(cfg_him.env.domain_rand)
    him_events = him_task_cfg._build_events()
    assert him_events["body_mass"].mode == "reset"
    assert him_events["body_mass"].params["asset_cfg"].body_names == (
        "arm_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
    )
    assert him_events["base_mass"].params["asset_cfg"].body_names == ("BASE_LINK",)
    for owner in (
        cfg,
        compose_config("ppo", "motrix", [], task="l1_w_arm_manip_loco"),
        cfg_him,
    ):
        assert owner.env.control_config.leg_kp == pytest.approx(50.0)
        assert owner.env.control_config.leg_kd == pytest.approx(1.0)
        assert owner.env.command_config.pure_x_command_prob == pytest.approx(0.35)
        assert owner.env.command_config.gait_x_threshold == pytest.approx(0.5)
        assert owner.env.command_config.gait_lateral_threshold == pytest.approx(0.08)
        assert owner.env.command_config.gait_yaw_threshold == pytest.approx(0.12)
        assert owner.reward
        assert "rolling_leg_motion" not in owner.reward
        assert "rolling_foot_lift" not in owner.reward
        assert all(
            term.func == "legged_manipulation_unilab.tasks.l1_w_arm.manager_env.L1WArmReward"
            for term in owner.reward.values()
        )
    with pytest.raises(ValueError, match="identity"):
        compose_config(
            "ppo", "mujoco", ["training={sim_backend:unknown}"], task="l1_w_arm_manip_loco"
        )


def test_l1_w_arm_samples_pure_forward_and_backward_commands():
    count = 20_000
    ids = np.arange(count)

    def sample(cfg):
        command = SimpleNamespace(
            _task_cfg=SimpleNamespace(command_config=cfg),
            _env=SimpleNamespace(rng=np.random.default_rng(7)),
            vel_command_b=np.zeros((count, 3), dtype=np.float32),
        )
        Go2ArmManipLocoCommand._sample_velocity(command, ids)
        return command.vel_command_b

    wheel_commands = sample(L1WArmCommandsConfig(zero_command_prob=0.15))
    pure_x = (np.abs(wheel_commands[:, 0]) > 0.1) & np.all(
        wheel_commands[:, 1:3] == 0.0, axis=1
    )
    assert 0.2 < np.mean(pure_x) < 0.35
    assert np.any(wheel_commands[pure_x, 0] > 0.0)
    assert np.any(wheel_commands[pure_x, 0] < 0.0)

    go2_commands = sample(CommandsConfig(zero_command_prob=0.15))
    go2_pure_x = (np.abs(go2_commands[:, 0]) > 0.1) & np.all(
        go2_commands[:, 1:3] == 0.0, axis=1
    )
    assert not np.any(go2_pure_x)


@pytest.mark.slow
def test_l1_w_arm_mba_reset_step_contract():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(
        num_envs=2,
        reward_names=(
            "tracking_lin_vel",
            "contact",
            "swing_feet_z",
        ),
    )
    try:
        assert env.obs_groups_spec == {"obs": 440, "critic": 91}
        assert env.action_space.shape == (22,)
        obs, _ = env.reset()
        assert obs["obs"].shape == (2, 440)
        state = env.step(np.zeros((2, 22), dtype=np.float32))
        assert state.reward.shape == (2,)
        assert np.isfinite(state.reward).all()
        assert state.obs["obs"].shape == (2, 440)
        default_z = float(np.asarray(env.scene["robot"].data.default_root_state)[0, 2])
        assert default_z == pytest.approx(0.4)
    finally:
        env.close()
