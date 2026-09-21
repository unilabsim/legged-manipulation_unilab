"""MuJoCo Pico teleop for Manip-Loco play: left stick chassis, right-grip EE pose."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.utils.rotation import np_quat_from_euler_xyz
from unilab.visualization.debug_primitives import append_debug_primitives_to_scene
from unilab.visualization.interactive_playback import PlaybackControls

from legged_manipulation_unilab.tasks.go2_arm.manager_env import Go2ArmManipLocoCommand
from legged_manipulation_unilab.training.him import build_play_debug_overlays
from legged_manipulation_unilab.training.keyboard_play import (
    _KEY_BACKSPACE,
    PendingKeyQueue,
    _load_viewer_model,
    _open_him_session,
    _open_ppo_session,
    _select_device,
    bind_ee_teleop,
    bind_velocity_teleop,
    episode_requires_reset,
    sync_velocity_into_session_obs,
)
from legged_manipulation_unilab.training.pico_device import PicoDevice
from legged_manipulation_unilab.training.pico_mapping import PicoMapper

_DEFAULT_DEADZONE = 0.1
_DEFAULT_MOTION_SCALE = 1.0
_DEFAULT_GRIP_THRESHOLD = 0.5
_DEFAULT_SERVICE = "/opt/apps/roboticsservice/runService.sh"


def wants_pico_play(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "interactive.pico", default=False))


def bind_pico_ee_teleop(command: Go2ArmManipLocoCommand) -> None:
    """Hold EE goals from Pico and enable IK position + orientation tracking."""
    bind_ee_teleop(command)
    ik = command._task_cfg.ik
    if not ik.use_orientation or ik.orientation_mode != "target":
        print(
            "[pico-play] enabling IK orientation tracking for pose teleop "
            f"(use_orientation={ik.use_orientation}, mode={ik.orientation_mode})"
        )
        ik.use_orientation = True
        ik.orientation_mode = "target"


def home_ee_pose(command: Go2ArmManipLocoCommand) -> tuple[np.ndarray, np.ndarray]:
    """Task init Cartesian goal plus the default EE orientation used at hold bind."""
    cfg = command._task_cfg.goal_ee
    cart = np.asarray(cfg.init_ee_cart, dtype=np.float64).reshape(3)
    quat = np.asarray(
        np_quat_from_euler_xyz(float(cfg.default_orn_roll), float(cfg.arm_induced_pitch), 0.0),
        dtype=np.float64,
    ).reshape(4)
    return cart, quat


def mapper_from_command(command: Go2ArmManipLocoCommand, cfg: DictConfig) -> PicoMapper:
    home_cart, home_quat = home_ee_pose(command)
    return PicoMapper(
        motion_scale=float(
            OmegaConf.select(cfg, "interactive.pico_motion_scale", default=_DEFAULT_MOTION_SCALE)
        ),
        dead_zone=float(
            OmegaConf.select(cfg, "interactive.pico_deadzone", default=_DEFAULT_DEADZONE)
        ),
        grip_threshold=float(
            OmegaConf.select(
                cfg, "interactive.pico_grip_threshold", default=_DEFAULT_GRIP_THRESHOLD
            )
        ),
        vel_limit=command._task_cfg.command_config.vel_limit,
        home_cart=home_cart,
        home_quat=home_quat,
        home_duration=float(
            OmegaConf.select(cfg, "interactive.pico_home_duration", default=1.2)
        ),
        home_lin_speed=float(
            OmegaConf.select(cfg, "interactive.pico_home_lin_speed", default=0.15)
        ),
        home_ang_speed=float(
            OmegaConf.select(cfg, "interactive.pico_home_ang_speed", default=1.0)
        ),
    )


def apply_pico_command(command: Go2ArmManipLocoCommand, pico_cmd) -> None:
    command.set_velocity_command(pico_cmd.velocity)
    if pico_cmd.ee_cart is not None and pico_cmd.ee_quat is not None:
        command.set_ee_goal_pose(pico_cmd.ee_cart, pico_cmd.ee_quat)


def push_pico_command(
    session: Any,
    command: Go2ArmManipLocoCommand,
    mapper: PicoMapper,
    pico_data: dict,
    *,
    dt: float = 0.02,
) -> None:
    pico_cmd = mapper.update(
        pico_data,
        np.asarray(command.curr_ee_goal_cart[0], dtype=np.float64),
        np.asarray(command.ee_goal_orn_quat[0], dtype=np.float64),
        dt=dt,
    )
    apply_pico_command(command, pico_cmd)
    if pico_cmd.home_started:
        print("[pico-play] EE home (right trigger)")
    sync_velocity_into_session_obs(session, command)


def reset_held_episode_pico(
    session: Any,
    command: Go2ArmManipLocoCommand,
    mapper: PicoMapper,
    pico_data: dict,
) -> bool:
    if not episode_requires_reset(session.env):
        return False
    session.reset()
    mapper.reset_homing()
    push_pico_command(session, command, mapper, pico_data, dt=0.0)
    return True


def apply_pico_viewer_key(
    keycode: int,
    *,
    controls: PlaybackControls,
    session: Any,
    command: Go2ArmManipLocoCommand,
    mapper: PicoMapper,
    pico_data: dict,
) -> None:
    if keycode == ord(" "):
        paused = controls.toggle_pause()
        print(f"[pico-play] {'paused' if paused else 'resumed'} (space)")
        return
    if keycode in (ord("N"), ord("n")):
        controls.request_single_step()
        if not controls.paused:
            controls.pause()
            print("[pico-play] paused for single-step mode (n)")
        print("[pico-play] single step requested (n)")
        return
    if keycode in (ord("+"), ord("=")):
        controls.set_speed(controls.speed * 1.25)
        print(f"[pico-play] speed={controls.speed:.2f}x")
        return
    if keycode in (ord("-"), ord("_")):
        controls.set_speed(controls.speed / 1.25)
        print(f"[pico-play] speed={controls.speed:.2f}x")
        return
    if keycode == _KEY_BACKSPACE:
        mapper.reset_clutch()
        mapper.reset_homing()
        session.reset()
        push_pico_command(session, command, mapper, pico_data, dt=0.0)
        print("[pico-play] reset (backspace); release and re-grip to recapture EE")


def _print_help(mapper: PicoMapper) -> None:
    print("[pico-play] Pico teleop ENABLED:")
    print("  Left stick Y  : chassis forward / backward (vx)")
    print("  Left stick X  : chassis turn left / right (vyaw)")
    print("  Right grip    : clutch EE pose (position + orientation)")
    print("  Right trigger : smooth-home EE (task init pose + default orientation)")
    print("  Left stick    : push diagonally for forward+turn at the same time")
    print("  Backspace     : reset episode (homes EE, zeros chassis until stick moves)")
    print("  Space         : pause / resume")
    print("  Timeout / fallen terminations are disabled; only Backspace resets.")
    print(
        f"  Stick deadzone={mapper.dead_zone:.2f}, motion_scale={mapper.motion_scale:.2f}, "
        f"grip_threshold={mapper.grip_threshold:.2f}, home_duration={mapper.home_duration:.2f}s."
    )


def run_pico_viewer(session: Any, cfg: DictConfig, device: PicoDevice) -> None:
    import mujoco
    import mujoco.viewer

    env = session.env
    command = bind_velocity_teleop(env)
    bind_pico_ee_teleop(command)
    mapper = mapper_from_command(command, cfg)
    controls = PlaybackControls(
        paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
    )
    mj_model = _load_viewer_model(env)
    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.02))

    session.reset()
    pico_data = device.get()
    push_pico_command(session, command, mapper, pico_data)
    pending_keys = PendingKeyQueue()

    def _on_key(keycode: int) -> None:
        pending_keys.push(keycode)

    _print_help(mapper)

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()
                pico_data = device.get()
                for keycode in pending_keys.drain():
                    apply_pico_viewer_key(
                        keycode,
                        controls=controls,
                        session=session,
                        command=command,
                        mapper=mapper,
                        pico_data=pico_data,
                    )
                dt = 0.0 if controls.paused else ctrl_dt
                push_pico_command(session, command, mapper, pico_data, dt=dt)
                session.advance(controls)
                phys = session.physics_state()[0].astype(np.float64)
                mujoco.mj_setState(mj_model, viz_data, phys, state_spec)
                mujoco.mj_forward(mj_model, viz_data)
                viewer.user_scn.ngeom = 0
                overlays = build_play_debug_overlays(
                    vel_commands=np.asarray(command.vel_command_b),
                    ee_goal_world=np.asarray(command.curr_ee_goal_world),
                    base_pos_w=np.asarray(env.scene["robot"].data.root_link_pos_w),
                    base_quat_w=np.asarray(env.scene["robot"].data.root_link_quat_w),
                )
                append_debug_primitives_to_scene(viewer.user_scn, overlays[0])
                viewer.sync()
                sleep_s = (ctrl_dt / max(float(controls.speed), 1e-6)) - (time.perf_counter() - t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)


def play_with_pico(cfg: DictConfig, *, algo: str) -> None:
    if str(cfg.training.sim_backend) != "mujoco":
        raise SystemExit("pico play requires --sim mujoco (MuJoCo viewer)")
    device_name = _select_device(cfg)
    print(f"[pico-play] Device: {device_name}")
    opener = _open_him_session if algo == "him_ppo" else _open_ppo_session
    opened = opener(cfg, device_name, num_envs=1)
    if opened is None:
        return
    resources, session = opened
    service = OmegaConf.select(cfg, "interactive.pico_service", default=_DEFAULT_SERVICE)
    pico = PicoDevice(service_path=None if service in (None, "", False) else str(service))
    try:
        with resources:
            run_pico_viewer(session, cfg, pico)
    finally:
        pico.close()
