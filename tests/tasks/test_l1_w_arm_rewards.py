"""Checks for lateral-hip isolation and N-P3O diagonal joint symmetry."""

from types import SimpleNamespace

import numpy as np
import pytest

from legged_manipulation_unilab.tasks.go2_arm import manager_env
from legged_manipulation_unilab.tasks.go2_arm.manip_loco import Go2ArmManipLocoCfg
from legged_manipulation_unilab.tasks.l1_w_arm import manager_env as l1_manager_env
from legged_manipulation_unilab.tasks.l1_w_arm.manip_loco import L1WArmManipLocoCfg


def _reward_inputs(monkeypatch, name):
    cfg = L1WArmManipLocoCfg()
    sensors = cfg.sensor
    foot_force = tuple(sensors.feet_force)
    foot_pos = tuple(sensors.feet_pos)
    names = [
        sensors.local_linvel,
        sensors.gyro,
        "upvector",
        *foot_force,
        *foot_pos,
        *l1_manager_env._ARM_TOUCH_SENSORS,
    ]
    slices = {}
    width = 0
    for sensor in names:
        size = 3 if sensor in [sensors.local_linvel, sensors.gyro, "upvector", *foot_pos] else 1
        slices[sensor] = slice(width, width + size)
        width += size
    values = np.zeros((1, width))
    for sensor in foot_pos:
        values[0, slices[sensor]][2] = 0.071
    data = SimpleNamespace(
        joint_pos=np.zeros((1, 22)),
        default_joint_pos=np.zeros((1, 22)),
        joint_vel=np.zeros((1, 22)),
        root_link_pos_w=np.zeros((1, 3)),
    )
    reward = l1_manager_env.L1WArmReward.__new__(l1_manager_env.L1WArmReward)
    reward._name = name
    reward._task_cfg = cfg
    reward._feet_force_names = foot_force
    reward._feet_pos_names = foot_pos
    reward._slices = slices
    reward._views = SimpleNamespace(read=lambda: values)
    reward._ee_view = SimpleNamespace(read=lambda: np.zeros((1, 3)))
    reward._entity = SimpleNamespace(data=data)
    monkeypatch.setattr(
        l1_manager_env,
        "_command",
        lambda env: SimpleNamespace(vel_command_b=np.zeros((1, 3)), feet_phase=np.zeros((1, 4))),
    )
    return reward, data, values, slices


def test_hip_pos_excludes_pitch_knee_wheels_and_arm(monkeypatch):
    reward, data, _, _ = _reward_inputs(monkeypatch, "hip_pos")
    # Both pitch joints, a wheel and an arm joint must not incur ABAD cost.
    data.joint_pos[0, [1, 2, 12, 16]] = 0.4
    assert reward(None, name=reward._name)[0] == 0.0
    data.joint_pos[0, 9] = 0.2  # Left rear ABAD.
    assert reward(None, name=reward._name)[0] == pytest.approx(0.04)
    data.default_joint_pos[0, 9] = 0.2
    assert reward(None, name=reward._name)[0] == 0.0


def test_mirrorup_pairs_l1w_diagonal_joints_and_excludes_wheels_arm(monkeypatch):
    reward, data, values, slices = _reward_inputs(monkeypatch, "mirrorup")
    values[0, slices["upvector"]][2] = 1.0
    # L1W joint order: FL, FR, RR, RL; sensor order differs.
    data.joint_pos[0, :12] = [0.1, 0.8, -1.5, -0.2, 1.0, -1.3] * 2
    data.joint_pos[0, 12:] = 10.0
    assert reward(None, name="mirrorup")[0] == pytest.approx(0.0)
    # Equal foot heights do not hide a mismatch in any of the three angles.
    data.joint_pos[0, 6:9] += [0.2, 0.3, 0.4]
    data.joint_pos[0, 9:12] += [0.1, 0.2, 0.3]
    expected = 0.5 * (0.2**2 + 0.3**2 + 0.4**2 + 0.1**2 + 0.2**2 + 0.3**2)
    assert reward(None, name="mirrorup")[0] == pytest.approx(expected)
    # Compare raw angles, independently of the default pose and foot heights.
    data.default_joint_pos[:] = data.joint_pos
    values[0, slices[reward._feet_pos_names[0]]][2] = 0.3
    assert reward(None, name="mirrorup")[0] == pytest.approx(expected)


@pytest.mark.parametrize("up_z, expected", [(1.0, 0.5), (0.5, 0.25), (-1.0, 0.0)])
def test_mirrorup_uses_positive_body_up_z(monkeypatch, up_z, expected):
    reward, data, values, slices = _reward_inputs(monkeypatch, "mirrorup")
    data.joint_pos[0, 0] = 1.0
    values[0, slices["upvector"]][2] = up_z
    assert reward(None, name="mirrorup")[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "velocity_command, expected",
    [
        ([0, 0, 0], 30.0),
        ([0.001, 0.0, 0.0], 30.0),
        ([0.05, 0.0, 0.0], 0.0),
        ([0.1, -0.1, 0.1], 0.0),
        ([0.2, 0, 0], 0.0),
        ([0, -0.2, 0], 0.0),
        ([0, 0, 0.2], 0.0),
    ],
)
def test_wheel_stand_still_gates_all_command_axes_and_only_penalizes_wheels(
    monkeypatch, velocity_command, expected
):
    reward, data, _, _ = _reward_inputs(monkeypatch, "wheel_stand_still")
    data.joint_vel[:] = 100.0  # Moving legs and arm must not contribute.
    data.joint_vel[0, 12:16] = [1.0, -2.0, 3.0, -4.0]
    original_vel = data.joint_vel.copy()
    command = SimpleNamespace(
        vel_command_b=np.array([velocity_command]),
        feet_phase=np.zeros((1, 4)),
    )
    command._command_is_moving = lambda commands: (
        manager_env.Go2ArmManipLocoCommand._command_is_moving(command, commands)
    )
    monkeypatch.setattr(l1_manager_env, "_command", lambda env: command)
    assert reward(None, name="wheel_stand_still")[0] == pytest.approx(expected)
    np.testing.assert_array_equal(data.joint_vel, original_vel)


def test_l1w_gait_runs_only_for_lateral_or_yaw_commands(monkeypatch):
    command = manager_env.Go2ArmManipLocoCommand.__new__(manager_env.Go2ArmManipLocoCommand)
    command._task_cfg = L1WArmManipLocoCfg()
    command.vel_command_b = np.array(
        [
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, 0.0, -0.2],
            [0.0, 0.04, 0.0],
            [0.0, 0.08, 0.0],
        ]
    )
    command.phase = np.full(6, 0.2, dtype=np.float32)
    command.feet_phase = np.full((6, 4), 0.8, dtype=np.float32)
    command.gait_frequency = 2.0
    command._episode_steps = np.zeros(6, dtype=np.int32)
    monkeypatch.setattr(
        manager_env.Go2ArmManipLocoCommand, "_update_ee_trajectory", lambda self: None
    )
    monkeypatch.setattr(manager_env.Go2ArmManipLocoCommand, "_log_command_state", lambda self: None)

    command._update_command(None)

    np.testing.assert_array_equal(command.phase[[0, 1, 4]], 0.0)
    np.testing.assert_array_equal(command.feet_phase[[0, 1, 4]], 0.0)
    np.testing.assert_allclose(command.phase[[2, 3, 5]], 0.24)
    assert np.any(command.feet_phase[[2, 3, 5]] >= 0.6)
    assert command._command_is_moving(command.vel_command_b)[0]


@pytest.mark.parametrize(
    "velocity_command, expected_swing, expected_contact",
    [
        ([0.5, 0.0, 0.0], 0.0, 1.0),
        ([0.0, 0.08, 0.0], 0.5, 0.5),
        ([0.0, 0.0, -0.2], 0.5, 0.5),
    ],
)
def test_gait_rewards_follow_lateral_or_yaw_gate(
    monkeypatch, velocity_command, expected_swing, expected_contact
):
    reward, _, values, slices = _reward_inputs(monkeypatch, "swing_feet_z")
    reward._task_cfg.reward_parameters.target_foot_height = 0.10
    values[0, slices[reward._feet_pos_names[0]]][2] = 0.10
    values[0, slices[reward._feet_pos_names[3]]][2] = 0.10
    for sensor in reward._feet_force_names:
        values[0, slices[sensor]] = 1.0
    command = SimpleNamespace(
        vel_command_b=np.array([velocity_command]),
        feet_phase=np.array([[0.75, 0.25, 0.25, 0.75]]),
        gait_frequency=2.0,
    )
    command._gait_is_active = lambda commands: manager_env.Go2ArmManipLocoCommand._gait_is_active(
        SimpleNamespace(_task_cfg=reward._task_cfg), commands
    )
    monkeypatch.setattr(l1_manager_env, "_command", lambda env: command)
    assert reward(None, name="swing_feet_z")[0] == pytest.approx(expected_swing)
    reward._name = "contact"
    assert reward(None, name="contact")[0] == pytest.approx(expected_contact)


def test_go2_forward_command_keeps_gait_active():
    command = manager_env.Go2ArmManipLocoCommand.__new__(manager_env.Go2ArmManipLocoCommand)
    command._task_cfg = Go2ArmManipLocoCfg()
    np.testing.assert_array_equal(
        command._gait_is_active(np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        [True, False],
    )
