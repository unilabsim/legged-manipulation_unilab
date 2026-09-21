from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.utils.rotation import np_matrix_from_quat, np_quat_from_euler_xyz
from unilab.visualization.interactive_playback import PlaybackControls

from legged_manipulation_unilab.cli import compose_config
from legged_manipulation_unilab.training.pico_mapping import (
    PicoMapper,
    apply_dead_zone,
    joystick_to_velocity,
    pose_matrix,
    process_xr_pose,
    quat_wxyz_from_matrix,
    radial_square_normalize,
    retarget_ee_pose,
)
from legged_manipulation_unilab.training.pico_play import (
    apply_pico_command,
    apply_pico_viewer_key,
    bind_pico_ee_teleop,
    play_with_pico,
    push_pico_command,
    wants_pico_play,
)

_VEL_LIMIT = [[-1.0, -0.4, -0.8], [1.0, 0.4, 0.8]]


def _identity_pose7() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _pico_data(*, right_pose=None, head_pose=None, grip=0.0, trigger=0.0, stick=(0.0, 0.0)) -> dict:
    return {
        "right_pose": _identity_pose7() if right_pose is None else right_pose,
        "head_pose": _identity_pose7() if head_pose is None else head_pose,
        "right_grip": grip,
        "right_trigger": trigger,
        "left_joystick": list(stick),
    }


class _FakeCommand:
    def __init__(self):
        self.written = None
        self.ee_pose = None
        self.curr_ee_goal_cart = np.array([[0.30, 0.0, 0.25]], dtype=np.float64)
        self.ee_goal_orn_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.vel_command_b = np.zeros((1, 3), dtype=np.float64)
        self._task_cfg = SimpleNamespace(
            ik=SimpleNamespace(use_orientation=False, orientation_mode="zero_error"),
            arm_stage=SimpleNamespace(freeze_arm_joints=False, disable_ee_goal_trajectory=False),
            goal_ee=SimpleNamespace(
                init_ee_cart=[0.30, 0.0, 0.25],
                default_orn_roll=1.5708,
                arm_induced_pitch=0.78,
            ),
        )
        self._ee_hold = False

    def enable_ee_hold(self):
        self._ee_hold = True

    def set_ee_goal_cart(self, cart):
        self.curr_ee_goal_cart[:] = np.asarray(cart, dtype=np.float64).reshape(3)

    def set_velocity_command(self, command):
        self.written = np.asarray(command, dtype=np.float64)
        self.vel_command_b[:] = self.written.reshape(3)

    def set_ee_goal_pose(self, cart, quat):
        self.ee_pose = (
            np.asarray(cart, dtype=np.float64).reshape(3),
            np.asarray(quat, dtype=np.float64).reshape(4),
        )
        self.curr_ee_goal_cart[:] = self.ee_pose[0]
        self.ee_goal_orn_quat[:] = self.ee_pose[1]
        return True


def test_wants_pico_play_reads_interactive_flag():
    cfg = compose_config("ppo", "mujoco", ["interactive.pico=true"])
    assert wants_pico_play(cfg) is True
    cfg = compose_config("him_ppo", "mujoco", [])
    assert wants_pico_play(cfg) is False
    cfg = compose_config("him_ppo", "mujoco", ["interactive.pico=true"])
    assert wants_pico_play(cfg) is True


def test_composed_pico_defaults():
    for algo in ("ppo", "him_ppo"):
        cfg = compose_config(algo, "mujoco", ["interactive.pico=true"])
        assert float(cfg.interactive.pico_deadzone) == pytest.approx(0.1)
        assert float(cfg.interactive.pico_motion_scale) == pytest.approx(1.0)
        assert float(cfg.interactive.pico_grip_threshold) == pytest.approx(0.5)
        assert float(cfg.interactive.pico_home_duration) == pytest.approx(1.2)
        assert float(cfg.interactive.pico_home_lin_speed) == pytest.approx(0.15)


def test_pico_play_rejects_motrix():
    cfg = compose_config("ppo", "motrix", ["interactive.pico=true"])
    with pytest.raises(SystemExit, match="mujoco"):
        play_with_pico(cfg, algo="ppo")


def test_apply_dead_zone_renormalizes():
    assert apply_dead_zone(0.05, 0.1) == 0.0
    assert apply_dead_zone(1.0, 0.1) == pytest.approx(1.0)
    assert apply_dead_zone(-1.0, 0.1) == pytest.approx(-1.0)
    assert apply_dead_zone(0.1, 0.1) == 0.0
    assert apply_dead_zone(0.55, 0.1) == pytest.approx(0.5)


def test_joystick_scheme_a_maps_y_to_vx_and_x_to_yaw():
    forward = joystick_to_velocity([0.0, 1.0], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert forward == pytest.approx([1.0, 0.0, 0.0])
    turn = joystick_to_velocity([1.0, 0.0], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert turn[0] == pytest.approx(0.0)
    assert turn[1] == pytest.approx(0.0)
    assert turn[2] == pytest.approx(-0.8)
    idle = joystick_to_velocity([0.05, -0.02], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert idle.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_joystick_diagonal_applies_forward_and_yaw_together():
    diagonal = joystick_to_velocity([0.8, 0.8], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert diagonal[0] > 0.0
    assert diagonal[1] == pytest.approx(0.0)
    assert diagonal[2] < 0.0
    opposite = joystick_to_velocity([-0.8, 0.8], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert opposite[0] > 0.0
    assert opposite[2] > 0.0


def test_radial_square_normalize_lifts_circle_diagonal_to_full_axes():
    x, y = radial_square_normalize(0.70710678118, 0.70710678118)
    assert x == pytest.approx(1.0, abs=1e-5)
    assert y == pytest.approx(1.0, abs=1e-5)
    x, y = radial_square_normalize(0.0, 1.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(1.0)
    x, y = radial_square_normalize(0.35355339059, 0.35355339059)
    assert x == pytest.approx(0.5, abs=1e-5)
    assert y == pytest.approx(0.5, abs=1e-5)


def test_joystick_full_circle_diagonal_reaches_near_full_scale():
    # Circular gate full diagonal ≈ (0.707, 0.707); after radial remap ≈ (1, 1).
    cmd = joystick_to_velocity(
        [0.70710678118, 0.70710678118], vel_limit=_VEL_LIMIT, dead_zone=0.1
    )
    assert cmd[0] == pytest.approx(1.0, abs=1e-5)
    assert cmd[2] == pytest.approx(-0.8, abs=1e-5)
    pure = joystick_to_velocity([0.0, 1.0], vel_limit=_VEL_LIMIT, dead_zone=0.1)
    assert pure[0] == pytest.approx(1.0)
    assert pure[2] == pytest.approx(0.0)


def test_process_xr_pose_identity_and_y_up_to_z_up():
    identity = process_xr_pose(_identity_pose7(), _identity_pose7())
    np.testing.assert_allclose(identity, np.eye(4), atol=1e-6)

    controller = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    transform = process_xr_pose(controller, _identity_pose7())
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 1.0], atol=1e-6)


def test_quat_wxyz_from_matrix_roundtrip():
    quat = np_quat_from_euler_xyz(0.3, -0.4, 0.5)
    restored = quat_wxyz_from_matrix(np_matrix_from_quat(quat))
    aligned = restored if np.dot(restored, quat) >= 0.0 else -restored
    np.testing.assert_allclose(aligned, quat, atol=1e-6)


def test_clutch_captures_on_grip_and_freezes_on_release():
    mapper = PicoMapper(motion_scale=1.0, dead_zone=0.1, grip_threshold=0.5, vel_limit=_VEL_LIMIT)
    home = np.array([0.30, 0.0, 0.25], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    idle = mapper.update(_pico_data(grip=0.0), home, quat)
    assert idle.ee_cart is None
    assert idle.clutching is False

    held = mapper.update(_pico_data(grip=0.8), home, quat)
    assert held.clutching is True
    np.testing.assert_allclose(held.ee_cart, home, atol=1e-6)
    np.testing.assert_allclose(held.ee_quat, quat, atol=1e-6)

    moved_controller = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    moved = mapper.update(_pico_data(grip=0.8, right_pose=moved_controller), home, quat)
    assert moved.ee_cart is not None
    np.testing.assert_allclose(moved.ee_cart, home + np.array([0.0, 0.0, 0.1]), atol=1e-6)

    pitched = np_quat_from_euler_xyz(np.pi / 2.0, 0.78, 0.0)
    pitched_mapper = PicoMapper(motion_scale=1.0, vel_limit=_VEL_LIMIT)
    pitched_mapper.update(_pico_data(grip=0.8), home, pitched, dt=0.0)
    pitched_moved = pitched_mapper.update(
        _pico_data(grip=0.8, right_pose=moved_controller), home, pitched, dt=0.0
    )
    assert pitched_moved.ee_cart is not None
    np.testing.assert_allclose(pitched_moved.ee_cart, home + np.array([0.0, 0.0, 0.1]), atol=1e-6)
    np.testing.assert_allclose(pitched_moved.ee_cart[0], home[0], atol=1e-6)

    released = mapper.update(_pico_data(grip=0.0), moved.ee_cart, moved.ee_quat)
    assert released.ee_cart is None
    assert released.clutching is False


def test_retarget_keeps_operator_axes_uncoupled_with_pitched_ee():
    pitched = np_quat_from_euler_xyz(np.pi / 2.0, 0.78, 0.0)
    T_ee0 = pose_matrix([0.30, 0.0, 0.25], pitched)
    T_ctrl0 = np.eye(4)
    up = np.eye(4)
    up[:3, 3] = [0.0, 0.0, 0.1]
    forward = np.eye(4)
    forward[:3, 3] = [0.1, 0.0, 0.0]
    left = np.eye(4)
    left[:3, 3] = [0.0, 0.1, 0.0]

    ee_up = retarget_ee_pose(T_ctrl0, up, T_ee0, 1.0)
    np.testing.assert_allclose(ee_up[:3, 3], [0.30, 0.0, 0.35], atol=1e-6)
    np.testing.assert_allclose(ee_up[:3, :3], T_ee0[:3, :3], atol=1e-6)

    ee_fwd = retarget_ee_pose(T_ctrl0, forward, T_ee0, 1.0)
    np.testing.assert_allclose(ee_fwd[:3, 3], [0.40, 0.0, 0.25], atol=1e-6)

    ee_left = retarget_ee_pose(T_ctrl0, left, T_ee0, 1.0)
    np.testing.assert_allclose(ee_left[:3, 3], [0.30, 0.1, 0.25], atol=1e-6)


def test_retarget_uses_world_translation_even_if_controller_is_pitched():
    pitched_ctrl = pose_matrix([0.0, 0.0, 0.0], np_quat_from_euler_xyz(0.0, 0.7, 0.0))
    moved = pitched_ctrl.copy()
    moved[:3, 3] = [0.0, 0.0, 0.1]
    T_ee0 = pose_matrix([0.30, 0.0, 0.25], np_quat_from_euler_xyz(np.pi / 2.0, 0.78, 0.0))
    ee = retarget_ee_pose(pitched_ctrl, moved, T_ee0, 1.0)
    np.testing.assert_allclose(ee[:3, 3], [0.30, 0.0, 0.35], atol=1e-6)


def test_right_trigger_homes_ee_smoothly():
    home = np.array([0.30, 0.0, 0.25], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mapper = PicoMapper(
        vel_limit=_VEL_LIMIT,
        home_cart=home,
        home_quat=quat,
        home_duration=1.0,
        home_lin_speed=10.0,
        home_ang_speed=10.0,
    )
    current = home + np.array([0.05, 0.02, -0.04])
    first = mapper.update(_pico_data(trigger=0.9), current, quat, dt=0.25)
    assert first.home_started is True
    assert first.home_ee is True
    assert first.ee_cart is not None
    assert not np.allclose(first.ee_cart, home, atol=1e-4)
    assert not np.allclose(first.ee_cart, current, atol=1e-4)
    held = mapper.update(_pico_data(trigger=0.9), first.ee_cart, first.ee_quat, dt=0.25)
    assert held.home_started is False
    assert held.home_ee is True
    done = mapper.update(_pico_data(trigger=0.9), held.ee_cart, held.ee_quat, dt=1.0)
    np.testing.assert_allclose(done.ee_cart, home, atol=1e-6)
    idle = mapper.update(_pico_data(trigger=0.9), done.ee_cart, done.ee_quat, dt=0.02)
    assert idle.home_ee is False
    assert idle.ee_cart is None


def test_right_trigger_while_clutching_recaptures_after_smooth_home():
    home = np.array([0.30, 0.0, 0.25], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mapper = PicoMapper(
        vel_limit=_VEL_LIMIT,
        home_cart=home,
        home_quat=quat,
        home_duration=0.4,
        home_lin_speed=10.0,
        home_ang_speed=10.0,
    )
    mapper.update(_pico_data(grip=0.8), home, quat, dt=0.0)
    moved_controller = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    moved = mapper.update(_pico_data(grip=0.8, right_pose=moved_controller), home, quat, dt=0.0)
    assert moved.ee_cart is not None
    mid = mapper.update(
        _pico_data(grip=0.8, trigger=0.9, right_pose=moved_controller),
        moved.ee_cart,
        moved.ee_quat,
        dt=0.2,
    )
    assert mid.home_ee is True
    assert mid.clutching is False
    assert not np.allclose(mid.ee_cart, home, atol=1e-4)
    done = mapper.update(
        _pico_data(grip=0.8, trigger=0.9, right_pose=moved_controller),
        mid.ee_cart,
        mid.ee_quat,
        dt=0.4,
    )
    assert done.home_ee is True
    assert done.clutching is True
    np.testing.assert_allclose(done.ee_cart, home, atol=1e-6)
    np.testing.assert_allclose(mapper._T_ee0[:3, 3], home, atol=1e-6)


def test_bind_pico_ee_teleop_enables_orientation_ik():
    command = _FakeCommand()
    bind_pico_ee_teleop(command)  # type: ignore[arg-type]
    assert command._ee_hold is True
    assert command._task_cfg.ik.use_orientation is True
    assert command._task_cfg.ik.orientation_mode == "target"


def test_apply_pico_command_writes_velocity_and_pose():
    from legged_manipulation_unilab.training.pico_mapping import PicoCommand

    fake = _FakeCommand()
    apply_pico_command(
        fake,  # type: ignore[arg-type]
        PicoCommand(
            velocity=np.array([0.4, 0.0, -0.2], dtype=np.float64),
            ee_cart=np.array([0.32, 0.01, 0.26], dtype=np.float64),
            ee_quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
            clutching=True,
        ),
    )
    assert fake.written == pytest.approx([0.4, 0.0, -0.2])
    assert fake.ee_pose[0] == pytest.approx([0.32, 0.01, 0.26])
    assert fake.ee_pose[1] == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_push_pico_command_syncs_obs_from_left_stick():
    mapper = PicoMapper(vel_limit=_VEL_LIMIT)
    fake = _FakeCommand()
    history = 2
    one_step = 10
    actor = torch.zeros((1, history * one_step))
    session = SimpleNamespace(
        obs=actor,
        env=SimpleNamespace(cfg=SimpleNamespace(history=SimpleNamespace(num_actor_history=history))),
    )
    push_pico_command(
        session,
        fake,  # type: ignore[arg-type]
        mapper,
        _pico_data(stick=(0.0, 1.0)),
    )
    assert fake.written == pytest.approx([1.0, 0.0, 0.0])
    start = (history - 1) * one_step + 6
    assert actor[0, start : start + 3].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_backspace_resets_then_recaptures_if_still_gripping():
    from legged_manipulation_unilab.training.pico_play import _KEY_BACKSPACE

    mapper = PicoMapper(vel_limit=_VEL_LIMIT)
    home = np.array([0.30, 0.0, 0.25], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mapper.update(_pico_data(grip=0.8), home, quat)
    origin = mapper._T_ee0.copy()
    origin[:3, 3] = [0.40, 0.0, 0.25]
    mapper._T_ee0 = origin
    session = SimpleNamespace(reset_calls=0, obs=None)

    def _reset():
        session.reset_calls += 1

    session.reset = _reset
    apply_pico_viewer_key(
        _KEY_BACKSPACE,
        controls=PlaybackControls(),
        session=session,
        command=_FakeCommand(),  # type: ignore[arg-type]
        mapper=mapper,
        pico_data=_pico_data(grip=0.8),
    )
    assert session.reset_calls == 1
    assert mapper._clutch is True
    np.testing.assert_allclose(mapper._T_ee0[:3, 3], [0.30, 0.0, 0.25], atol=1e-6)


def test_pico_device_reads_injected_client():
    from legged_manipulation_unilab.training.pico_device import PicoDevice

    class _FakeXr:
        def __init__(self):
            self.closed = False

        def get_pose_by_name(self, name):
            return np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        def get_key_value_by_name(self, name):
            if name == "right_grip":
                return 0.7
            if name == "right_trigger":
                return 0.1
            return 0.0

        def get_joystick_state(self, controller):
            return [0.2, -0.3]

        def close(self):
            self.closed = True

    fake = _FakeXr()
    device = PicoDevice(client=fake, service_path=None)  # type: ignore[arg-type]
    data = device.get()
    np.testing.assert_allclose(data["right_pose"][:3], [0.1, 0.2, 0.3])
    assert data["right_grip"] == pytest.approx(0.7)
    assert data["right_trigger"] == pytest.approx(0.1)
    assert data["left_joystick"] == pytest.approx([0.2, -0.3])
    device.close()
    assert fake.closed is True
    assert device.client is None


def test_eval_pico_routes_to_pico_play(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from legged_manipulation_unilab import assets as assets_mod
    from legged_manipulation_unilab import cli
    from legged_manipulation_unilab.training import pico_play

    captured: dict[str, object] = {}

    monkeypatch.setattr(assets_mod, "ensure_assets", lambda **kwargs: tmp_path)

    def _fake_play(cfg, *, algo):
        captured["algo"] = algo
        captured["pico"] = bool(cfg.interactive.pico)
        captured["task"] = str(cfg.training.task_name)

    monkeypatch.setattr(pico_play, "play_with_pico", _fake_play)
    cli._main(
        play=True,
        argv=[
            "--task",
            "l1_w_arm_manip_loco",
            "interactive.pico=true",
        ],
    )
    assert captured == {
        "algo": "ppo",
        "pico": True,
        "task": "L1WArmManipLoco",
    }


def test_eval_pico_takes_priority_over_keyboard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from legged_manipulation_unilab import assets as assets_mod
    from legged_manipulation_unilab import cli
    from legged_manipulation_unilab.training import keyboard_play, pico_play

    captured: dict[str, str] = {}
    monkeypatch.setattr(assets_mod, "ensure_assets", lambda **kwargs: tmp_path)
    monkeypatch.setattr(
        pico_play, "play_with_pico", lambda cfg, *, algo: captured.update(mode="pico")
    )
    monkeypatch.setattr(
        keyboard_play, "play_with_keyboard", lambda cfg, *, algo: captured.update(mode="keyboard")
    )
    cli._main(
        play=True,
        argv=["interactive.pico=true", "interactive.keyboard=true"],
    )
    assert captured == {"mode": "pico"}
