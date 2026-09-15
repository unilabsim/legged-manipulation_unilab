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


def _reward_override(reward_names: tuple[str, ...] = ("tracking_lin_vel",)) -> dict:
    return {
        name: {
            "func": "legged_manipulation_unilab.tasks.go2_arm.manager_env.Go2ArmReward",
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


def _make_env(
    num_envs: int = 2,
    reward_names: tuple[str, ...] = ("tracking_lin_vel",),
    extra: dict | None = None,
):
    _ensure_registered()
    registry = _registry_module()
    env_cfg_override = {
        "rewards": _reward_override(reward_names),
        "reward_parameters": {"tracking_sigma": 0.25, "base_height_target": 0.3},
        "domain_rand": dict(_DISABLED_DOMAIN_RAND),
    }
    if extra:
        env_cfg_override.update(extra)
    return registry.make(
        "Go2ArmManipLoco",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=env_cfg_override,
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

    # The owner yaml ships these terms at weight 0 (proven full-mode economics);
    # re-enable them here so the contract keeps verifying the physics wiring.
    cfg = compose_config(
        "him_ppo",
        "mujoco",
        [
            "reward.dof_pos_limits.weight=-10.0",
            "reward.torques.weight=-0.0002",
            "reward.energy.weight=-0.0002",
            "reward.dof_acc.weight=-2.5e-07",
        ],
    )
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


_ARM_TOUCH_SENSOR_NAMES = (
    "arm_touch_base",
    "arm_touch_link1",
    "arm_touch_link2",
    "arm_touch_link3",
    "arm_touch_link4",
    "arm_touch_link5",
    "arm_touch_link6",
    "arm_touch_eef",
    "arm_touch_g2base",
)


def _reward_terms(env) -> dict:
    manager = env.reward_manager
    return {
        name: manager._term_cfgs[manager.active_terms.index(name)].func
        for name in manager.active_terms
    }


def _home_keyframe_qpos() -> np.ndarray:
    root = ET.parse(ASSETS_ROOT_PATH / "robots/go2_arm/scene_flat.xml").getroot()
    for key in root.findall("./keyframe/key"):
        if key.attrib.get("name") == "home":
            return np.asarray([float(item) for item in key.attrib["qpos"].split()], dtype=np.float64)
    raise AssertionError("scene_flat.xml does not declare a 'home' keyframe")


def test_go2_arm_scene_declares_home_keyframe_default():
    _ensure_registered()
    registry = _registry_module()
    cfg = registry.materialize_env_config("Go2ArmManipLoco")
    assert cfg.scene.default_keyframe_name == "home"


@pytest.mark.slow
def test_go2_arm_default_state_uses_home_keyframe():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2)
    try:
        keyframe = _home_keyframe_qpos()
        entity = env.scene["robot"]
        default_joint_pos = np.asarray(entity.data.default_joint_pos, dtype=np.float64)
        default_joint_pos = default_joint_pos.reshape(env.num_envs, -1)
        expected_joint_pos = np.broadcast_to(keyframe[-18:], default_joint_pos.shape)
        np.testing.assert_allclose(default_joint_pos, expected_joint_pos, atol=1e-9)
        default_root_state = np.asarray(entity.data.default_root_state, dtype=np.float64)
        default_root_state = default_root_state.reshape(env.num_envs, -1)
        np.testing.assert_allclose(default_root_state[:, 2], keyframe[2], atol=1e-9)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_gait_reward_terms_read_matching_sensor_channels():
    """Gait rewards must read the same backend channels the formulas name.

    Guards the flat sensor block against width drift: foot-contact sensors are
    1-wide in the MJCF, so any offset that assumes 3-wide feet sensors silently
    repoints swing_feet_z / contact / foot_drag / arm_collision at other data.
    """
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(
        num_envs=2,
        reward_names=("contact", "swing_feet_z", "arm_collision", "foot_drag"),
    )
    try:
        terms = _reward_terms(env)
        command = env.command_manager.get_term("task")
        num_envs = env.num_envs

        def sensor(name: str, column: int) -> np.ndarray:
            data = np.asarray(env._backend.get_sensor_data(name), dtype=np.float64)
            return data.reshape(num_envs, -1)[:, column]

        env.reset()
        for _ in range(15):
            state = env.step(np.zeros((num_envs, 18), dtype=np.float32))
            assert np.isfinite(state.reward).all()

        forces = np.stack([sensor(name, -1) for name in env._cfg.sensor.feet_force], axis=1)
        heights = np.stack([sensor(name, 2) for name in env._cfg.sensor.feet_pos], axis=1)
        touch_total = np.zeros(num_envs, dtype=np.float64)
        for name in _ARM_TOUCH_SENSOR_NAMES:
            touch_total += sensor(name, 0)

        phase = np.asarray(command.feet_phase, dtype=np.float64)
        target = float(env._cfg.reward_parameters.target_foot_height)
        swing = phase >= 0.6
        stance = phase < 0.6
        expected_swing = (
            np.sum(np.exp(-np.square(heights - target) / 0.01) * swing, axis=1) / heights.shape[1]
        )
        expected_contact = np.sum((forces > 0.1) == stance, axis=1) / forces.shape[1]
        expected_drag = np.sum(
            np.square(np.clip(target / 2.0 - heights, 0.0, None)) * (forces < 0.5), axis=1
        )

        assert np.allclose(terms["swing_feet_z"](env, name="swing_feet_z"), expected_swing)
        assert np.allclose(terms["contact"](env, name="contact"), expected_contact)
        assert np.allclose(terms["foot_drag"](env, name="foot_drag"), expected_drag)
        assert np.allclose(terms["arm_collision"](env, name="arm_collision"), touch_total)
    finally:
        env.close()


_CURRICULUM_ON = {
    "curriculum_config": {
        "enable": True,
        "threshold": 0.8,
        "step_size": [0.1, 0.05, 0.1],
        "max_vel_limit": [1.0, 0.4, 0.8],
    }
}


def _vel_limit(command) -> tuple[np.ndarray, np.ndarray]:
    low, high = command._task_cfg.command_config.vel_limit
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


@pytest.mark.slow
def test_go2_arm_command_curriculum_expands_vel_limit_at_reset():
    """A well-tracked episode must expand vel_limit and clear its accumulators."""
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2, extra=dict(_CURRICULUM_ON))
    try:
        command = env.command_manager.get_term("task")
        env.reset()
        low_before, high_before = _vel_limit(command)
        command._episode_tracking_sum[:] = 100.0  # 1.0 per step for 100 steps
        command._episode_steps[:] = 100
        command.reset(np.arange(2, dtype=np.int32))
        low_after, high_after = _vel_limit(command)
        np.testing.assert_allclose(
            high_after, np.clip(high_before + [0.1, 0.05, 0.1], 0.0, [1.0, 0.4, 0.8])
        )
        np.testing.assert_allclose(
            low_after,
            np.clip(low_before - np.array([0.1, 0.05, 0.1]), -np.array([1.0, 0.4, 0.8]), 0.0),
        )
        assert np.all(command._episode_tracking_sum == 0.0)
        assert np.all(command._episode_steps == 0)
        assert command._expansion_count == 1
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_command_state_logged_each_step():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2)
    try:
        env.reset()
        state = env.step(np.zeros((2, 18), dtype=np.float32))
        log = state.info["log"]
        # Registry defaults (no him owner yaml here): [[-0.6,-0.4,-0.8],[1.0,0.4,0.8]].
        assert log["command/vel_high_x"] == pytest.approx(1.0)
        assert log["command/vel_low_z"] == pytest.approx(-0.8)
        assert log["command/curriculum_expansions"] == 0
        command = env.command_manager.get_term("task")
        assert command.gait_frequency == pytest.approx(
            env._cfg.command_config.gait_frequency
        )
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_command_curriculum_disabled_keeps_vel_limit():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2)
    try:
        command = env.command_manager.get_term("task")
        env.reset()
        low_before, high_before = _vel_limit(command)
        command._episode_tracking_sum[:] = 1.0
        command._episode_steps[:] = 100
        command.reset(np.arange(2, dtype=np.int32))
        np.testing.assert_allclose(_vel_limit(command)[0], low_before)
        np.testing.assert_allclose(_vel_limit(command)[1], high_before)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_tracking_reward_records_curriculum_progress():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2, extra=dict(_CURRICULUM_ON))
    try:
        command = env.command_manager.get_term("task")
        env.reset()
        assert np.all(command._episode_tracking_sum == 0.0)
        env.step(np.zeros((2, 18), dtype=np.float32))
        assert np.all(command._episode_tracking_sum > 0.0)
        assert np.all(command._episode_steps == 1)
    finally:
        env.close()


@pytest.mark.slow
def test_go2_arm_ee_goal_world_available_for_play_overlay():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    env = _make_env(num_envs=2)
    try:
        env.reset()
        state = env.step(np.zeros((2, 18), dtype=np.float32))
        goal_world = np.asarray(env.command_manager.get_term("task").curr_ee_goal_world)
        assert goal_world.shape == (2, 3)
        assert np.isfinite(goal_world).all()
        assert np.isfinite(state.obs["obs"]).all()
    finally:
        env.close()
