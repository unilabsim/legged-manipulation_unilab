from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from unilab.visualization.interactive_playback import KeyboardCommander

from legged_manipulation_unilab.cli import compose_config
from legged_manipulation_unilab.tasks.go2_arm.manager_env import clip_ee_goal_cart
from legged_manipulation_unilab.tasks.go2_arm.manip_loco import EEGoalConfig
from legged_manipulation_unilab.training.keyboard_play import (
    EndEffectorCommander,
    _handle_command_key,
    _handle_ee_key,
    apply_keyboard_ee,
    apply_keyboard_velocity,
    disable_play_terminations,
    episode_requires_reset,
    play_with_keyboard,
    sync_velocity_into_session_obs,
    wants_keyboard_play,
)


class _FakeCommand:
    def __init__(self):
        self.written = None
        self.ee_written = None
        self.curr_ee_goal_cart = np.array([[0.30, 0.0, 0.25]], dtype=np.float64)

    def set_velocity_command(self, command):
        self.written = np.asarray(command, dtype=np.float64)

    def set_ee_goal_cart(self, command):
        self.ee_written = np.asarray(command, dtype=np.float64)
        self.curr_ee_goal_cart[:] = self.ee_written.reshape(3)


def test_wants_keyboard_play_reads_interactive_flag():
    cfg = compose_config("ppo", "mujoco", ["interactive.keyboard=true"])
    assert wants_keyboard_play(cfg) is True
    cfg = compose_config("him_ppo", "mujoco", [])
    assert wants_keyboard_play(cfg) is False
    cfg = compose_config("him_ppo", "mujoco", ["interactive.keyboard=true"])
    assert wants_keyboard_play(cfg) is True


def test_disable_play_terminations_clears_active_terms():
    from types import SimpleNamespace

    manager = SimpleNamespace(
        active_terms=["time_out", "fallen"],
        _term_names=["time_out", "fallen"],
        _term_cfgs=["a", "b"],
        _class_term_cfgs=["c"],
        _term_dones={"time_out": np.array([True]), "fallen": np.array([False])},
        _truncated_buf=np.array([True]),
        _terminated_buf=np.array([True]),
    )
    env = SimpleNamespace(
        termination_manager=manager,
        _manual_reset_pending=np.array([True]),
    )
    disable_play_terminations(env)
    assert manager._term_names == []
    assert manager._term_cfgs == []
    assert manager._class_term_cfgs == []
    assert manager._term_dones == {}
    assert manager._truncated_buf.tolist() == [False]
    assert manager._terminated_buf.tolist() == [False]
    assert env._manual_reset_pending.tolist() == [False]


def test_keyboard_play_rejects_motrix():
    cfg = compose_config("ppo", "motrix", ["interactive.keyboard=true"])
    with pytest.raises(SystemExit, match="mujoco"):
        play_with_keyboard(cfg, algo="ppo")


def test_handle_command_key_drive_and_strafe():
    commander = KeyboardCommander.from_vel_limit(
        [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]],
        step_lin=0.2,
        step_ang=0.2,
    )
    from legged_manipulation_unilab.training import keyboard_play as mod

    assert _handle_command_key(commander, mod._KEY_UP)
    assert _handle_command_key(commander, mod._KEY_LEFT)
    assert commander.command == pytest.approx([0.2, 0.0, 0.2])
    assert _handle_command_key(commander, ord("7"))
    assert commander.command[1] == pytest.approx(0.2)
    assert _handle_command_key(commander, ord("8"))
    assert commander.command[1] == pytest.approx(0.0)
    assert _handle_command_key(commander, ord("8"))
    assert commander.command[1] == pytest.approx(-0.2)
    for key in "aAdD12":
        assert _handle_command_key(commander, ord(key)) is False
    assert _handle_command_key(commander, mod._KEY_ENTER)
    assert commander.command.tolist() == [0.0, 0.0, 0.0]
    assert _handle_command_key(commander, ord("q")) is False
    assert _handle_command_key(commander, ord("i")) is False


def test_handle_ee_key_nudge_and_home():
    ee = EndEffectorCommander([0.30, 0.0, 0.25], step=0.02)
    assert _handle_ee_key(ee, ord("u"))
    assert ee.cart[0] == pytest.approx(0.32)
    assert _handle_ee_key(ee, ord("h"))
    assert ee.cart[1] == pytest.approx(0.02)
    assert _handle_ee_key(ee, ord("y"))
    assert ee.cart[2] == pytest.approx(0.27)
    assert _handle_ee_key(ee, ord("l"))
    assert ee.cart.tolist() == pytest.approx([0.30, 0.0, 0.25])
    assert _handle_ee_key(ee, ord("i")) is False
    assert _handle_ee_key(ee, ord("q")) is False


def test_clip_ee_goal_cart_stays_in_spherical_workspace():
    cfg = EEGoalConfig()
    clipped = clip_ee_goal_cart(np.array([2.0, 0.0, 0.0]), cfg)
    radius = float(np.linalg.norm(clipped))
    assert radius == pytest.approx(cfg.sphere_l_range[1])


def test_apply_keyboard_ee_writes_and_syncs_commander():
    ee = EndEffectorCommander([0.30, 0.0, 0.25], step=0.02)
    ee.nudge(ee.AXIS_X, +1.0)
    fake = _FakeCommand()
    apply_keyboard_ee(fake, ee)  # type: ignore[arg-type]
    assert fake.ee_written == pytest.approx([0.32, 0.0, 0.25])
    assert ee.cart == pytest.approx(fake.curr_ee_goal_cart[0])


def test_apply_keyboard_velocity_writes_commander_command():
    commander = KeyboardCommander.from_vel_limit(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        step_lin=0.2,
    )
    commander.nudge(commander.AXIS_VX, +1.0)
    fake = _FakeCommand()
    apply_keyboard_velocity(fake, commander)  # type: ignore[arg-type]
    assert fake.written == pytest.approx([0.2, 0.0, 0.0])


def test_composed_keyboard_linear_step_clears_stand_deadzone():
    for algo in ("ppo", "him_ppo"):
        cfg = compose_config(algo, "mujoco", ["interactive.keyboard=true"])
        assert float(cfg.interactive.keyboard_step_lin) > 0.1
        assert float(cfg.interactive.keyboard_step_ee) > 0.0


def test_pending_key_queue_drains_in_order():
    from legged_manipulation_unilab.training.keyboard_play import PendingKeyQueue

    queue = PendingKeyQueue()
    queue.push(65)
    queue.push(66)
    assert queue.drain() == [65, 66]
    assert queue.drain() == []


def test_apply_viewer_backspace_resets_on_play_loop():
    from types import SimpleNamespace

    from unilab.visualization.interactive_playback import PlaybackControls

    from legged_manipulation_unilab.training.keyboard_play import (
        _KEY_BACKSPACE,
        apply_viewer_key,
    )

    commander = KeyboardCommander.from_vel_limit(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        step_lin=0.2,
    )
    commander.nudge(commander.AXIS_VX, +1.0)
    ee = EndEffectorCommander([0.30, 0.0, 0.25], step=0.02)
    ee.nudge(ee.AXIS_Z, +1.0)
    command = _FakeCommand()
    session = SimpleNamespace(
        reset_calls=0,
        obs=None,
        env=SimpleNamespace(cfg=None, state=None),
    )

    def _reset():
        session.reset_calls += 1

    session.reset = _reset
    apply_viewer_key(
        _KEY_BACKSPACE,
        controls=PlaybackControls(),
        commander=commander,
        session=session,
        command=command,  # type: ignore[arg-type]
        ee_commander=ee,
    )
    assert session.reset_calls == 1
    assert commander.command.tolist() == [0.0, 0.0, 0.0]
    assert ee.cart.tolist() == pytest.approx([0.30, 0.0, 0.25])
    assert command.written == pytest.approx([0.0, 0.0, 0.0])
    assert command.ee_written == pytest.approx([0.30, 0.0, 0.25])


def test_reset_held_episode_keeps_keyboard_velocity_after_done():
    from types import SimpleNamespace

    from legged_manipulation_unilab.training.keyboard_play import reset_held_episode

    commander = KeyboardCommander.from_vel_limit(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        step_lin=0.2,
    )
    commander.nudge(commander.AXIS_VX, +1.0)
    command = _FakeCommand()
    env = SimpleNamespace(_manual_reset_pending=np.array([True]), cfg=None, state=None)
    session = SimpleNamespace(env=env, reset_calls=0, obs=None)

    def _reset():
        session.reset_calls += 1
        env._manual_reset_pending[:] = False

    session.reset = _reset
    assert reset_held_episode(session, command, commander) is True  # type: ignore[arg-type]
    assert session.reset_calls == 1
    assert command.written == pytest.approx([0.2, 0.0, 0.0])
    assert reset_held_episode(session, command, commander) is False  # type: ignore[arg-type]
    assert session.reset_calls == 1


def test_episode_requires_reset_ignores_latched_terminated_flags():
    from types import SimpleNamespace

    env = SimpleNamespace(
        _manual_reset_pending=np.array([False]),
        state=SimpleNamespace(terminated=np.array([True]), truncated=np.array([True])),
    )
    assert episode_requires_reset(env) is False
    env._manual_reset_pending[:] = True
    assert episode_requires_reset(env) is True


def test_sync_velocity_into_session_obs_patches_latest_actor_frame():
    from types import SimpleNamespace

    import torch

    history = 5
    one_step = 76
    actor = torch.zeros(1, history * one_step)
    env_obs = np.zeros((1, history * one_step), dtype=np.float32)
    command = SimpleNamespace(vel_command_b=np.array([[0.4, -0.2, 0.1]], dtype=np.float32))
    session = SimpleNamespace(
        obs={"actor": actor, "policy": actor.clone()},
        env=SimpleNamespace(
            cfg=SimpleNamespace(history=SimpleNamespace(num_actor_history=history)),
            state=SimpleNamespace(obs={"obs": env_obs}),
        ),
    )
    sync_velocity_into_session_obs(session, command)  # type: ignore[arg-type]
    start = (history - 1) * one_step + 6
    assert actor[0, start : start + 3].tolist() == pytest.approx([0.4, -0.2, 0.1])
    assert env_obs[0, start : start + 3] == pytest.approx([0.4, -0.2, 0.1])


def test_eval_keyboard_routes_to_keyboard_play(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from legged_manipulation_unilab import assets as assets_mod
    from legged_manipulation_unilab import cli
    from legged_manipulation_unilab.training import keyboard_play

    captured: dict[str, object] = {}

    monkeypatch.setattr(assets_mod, "ensure_assets", lambda **kwargs: tmp_path)

    def _fake_play(cfg, *, algo):
        captured["algo"] = algo
        captured["keyboard"] = bool(cfg.interactive.keyboard)
        captured["task"] = str(cfg.training.task_name)

    monkeypatch.setattr(keyboard_play, "play_with_keyboard", _fake_play)
    cli._main(
        play=True,
        argv=[
            "--task",
            "l1_w_arm_manip_loco",
            "interactive.keyboard=true",
        ],
    )
    assert captured == {
        "algo": "ppo",
        "keyboard": True,
        "task": "L1WArmManipLoco",
    }
