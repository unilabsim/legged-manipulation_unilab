"""MuJoCo keyboard teleop for Manip-Loco play: chassis velocity and EE pose hold."""

from __future__ import annotations

import tempfile
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
from unilab.base.config_adapter import create_env
from unilab.training import (
    algo_config_dict,
    format_play_checkpoint_error,
    get_log_root,
    parse_checkpoint_path,
)
from unilab.utils.checkpoint import get_entrypoint_log_root
from unilab.visualization.debug_primitives import append_debug_primitives_to_scene
from unilab.visualization.interactive_playback import (
    KeyboardCommander,
    PlaybackControls,
    RslRlPlaybackConfig,
    create_rsl_rl_playback_session,
    infer_checkpoint_actor_input_dim,
    make_sim2sim_preflight,
    normalize_checkpoint_value,
)
from unisim.backend.mujoco.playback import resolve_render_play_model_files

from legged_manipulation_unilab.algos.him_ppo.runner import HIMOnPolicyRunner
from legged_manipulation_unilab.tasks.go2_arm.manager_env import Go2ArmManipLocoCommand
from legged_manipulation_unilab.training.him import (
    _backend_adapter as _him_backend_adapter,
)
from legged_manipulation_unilab.training.him import (
    _restore_trained_arm_stage,
    build_play_debug_overlays,
)

ROOT_DIR = Path.cwd()

_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259
_KEY_RIGHT, _KEY_LEFT, _KEY_DOWN, _KEY_UP = 262, 263, 264, 265
# Match Go2ArmManipLocoCommand sampling: |cmd| <= 0.1 is quantized to stand, and
# gait phase only advances when abs(cmd) > 0.1. A 0.1 keyboard step is therefore
# a stand command (and OOD for the actor). First nudge must clear that deadzone.
_VELOCITY_STAND_DEADZONE = 0.1
_DEFAULT_KEYBOARD_STEP_LIN = 0.2
_DEFAULT_KEYBOARD_STEP_ANG = 0.2
_DEFAULT_KEYBOARD_STEP_EE = 0.02
# Actor observation drops linvel, so body-frame vel_command starts at index 6.
_ACTOR_VEL_COMMAND_OFFSET = 6


class PendingKeyQueue:
    """GLFW key callbacks may fire during ``viewer.sync()``; env work stays on the play loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._keys: deque[int] = deque()

    def push(self, keycode: int) -> None:
        with self._lock:
            self._keys.append(int(keycode))

    def drain(self) -> list[int]:
        with self._lock:
            keys = list(self._keys)
            self._keys.clear()
        return keys


def wants_keyboard_play(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "interactive.keyboard", default=False))


class EndEffectorCommander:
    """Mutable arm-base-frame EE Cartesian goal driven by keyboard nudges."""

    AXIS_X = 0
    AXIS_Y = 1
    AXIS_Z = 2

    def __init__(self, home: np.ndarray, *, step: float = _DEFAULT_KEYBOARD_STEP_EE) -> None:
        self.home = np.asarray(home, dtype=np.float64).reshape(3).copy()
        self.cart = self.home.copy()
        self.step = float(step)

    @classmethod
    def from_command(
        cls, command: Go2ArmManipLocoCommand, *, step: float = _DEFAULT_KEYBOARD_STEP_EE
    ) -> "EndEffectorCommander":
        home = np.asarray(command._task_cfg.goal_ee.init_ee_cart, dtype=np.float64)
        commander = cls(home, step=step)
        commander.cart[:] = np.asarray(command.curr_ee_goal_cart[0], dtype=np.float64)
        return commander

    def nudge(self, axis: int, sign: float) -> None:
        self.cart[int(axis)] += self.step * (1.0 if sign >= 0 else -1.0)

    def go_home(self) -> None:
        self.cart[:] = self.home

    def describe(self) -> str:
        return (
            f"ee x={self.cart[0]:+.3f} y={self.cart[1]:+.3f} z={self.cart[2]:+.3f} "
            f"(step={self.step:.3f} m)"
        )


def task_velocity_command(env: Any) -> Go2ArmManipLocoCommand:
    term = env.command_manager.get_term("task")
    if not isinstance(term, Go2ArmManipLocoCommand):
        raise TypeError(f"Expected Go2ArmManipLocoCommand, got {type(term).__name__}")
    return term


def disable_play_terminations(env: Any) -> None:
    """Drop timeout/fallen terms so teleop play does not auto-end the episode.

    Manual reset (Backspace) still works. Autoreset stays off so a future
    termination, if re-enabled, would not silently bounce the robot.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is not None:
        names = list(getattr(manager, "active_terms", []) or [])
        for attr in ("_term_names", "_term_cfgs", "_class_term_cfgs"):
            terms = getattr(manager, attr, None)
            if isinstance(terms, list):
                terms.clear()
        dones = getattr(manager, "_term_dones", None)
        if isinstance(dones, dict):
            dones.clear()
        for buf_name in ("_truncated_buf", "_terminated_buf"):
            buf = getattr(manager, buf_name, None)
            if isinstance(buf, np.ndarray):
                buf[:] = False
        if names:
            print(f"[play] disabled terminations: {', '.join(names)} (Backspace still resets)")
        else:
            print("[play] episode terminations already empty")
    pending = getattr(env, "_manual_reset_pending", None)
    if isinstance(pending, np.ndarray):
        pending[:] = False


def bind_velocity_teleop(env: Any) -> Go2ArmManipLocoCommand:
    """Make keyboard/Pico the velocity source; leave EE resampling untouched."""
    command = task_velocity_command(env)
    command.enable_velocity_hold()
    command.set_velocity_command(np.zeros(3, dtype=np.float64))
    env.set_autoreset(False)
    disable_play_terminations(env)
    return command


def bind_ee_teleop(command: Go2ArmManipLocoCommand) -> None:
    """Make keyboard the EE pose source and enable IK tracking during play."""
    stage = command._task_cfg.arm_stage
    if stage.freeze_arm_joints or stage.disable_ee_goal_trajectory:
        print(
            "[keyboard-play] enabling IK EE tracking for teleop "
            f"(freeze_arm={stage.freeze_arm_joints}, "
            f"fixed_ee={stage.disable_ee_goal_trajectory})"
        )
        stage.freeze_arm_joints = False
        stage.disable_ee_goal_trajectory = False
    command.enable_ee_hold()
    command.set_ee_goal_cart(np.asarray(command._task_cfg.goal_ee.init_ee_cart, dtype=np.float64))


def apply_keyboard_velocity(command: Go2ArmManipLocoCommand, commander: KeyboardCommander) -> None:
    command.set_velocity_command(commander.command)


def apply_keyboard_ee(command: Go2ArmManipLocoCommand, ee_commander: EndEffectorCommander) -> None:
    command.set_ee_goal_cart(ee_commander.cart)
    ee_commander.cart[:] = np.asarray(command.curr_ee_goal_cart[0], dtype=np.float64)


def _actor_history_length(env: Any) -> int:
    history = getattr(getattr(env, "cfg", None), "history", None)
    length = int(getattr(history, "num_actor_history", 1) or 1)
    return max(length, 1)


def _write_last_frame_command(
    obs: np.ndarray | torch.Tensor, command: np.ndarray, history: int
) -> None:
    """Overwrite vel_command in the newest actor-history frame (in-place)."""
    if obs.ndim != 2 or obs.shape[1] < _ACTOR_VEL_COMMAND_OFFSET + 3:
        return
    width = int(obs.shape[1])
    one_step = width // history if history > 0 and width % history == 0 else width
    start = width - one_step + _ACTOR_VEL_COMMAND_OFFSET
    end = start + 3
    if start < 0 or end > width:
        return
    value = np.asarray(command, dtype=np.float32).reshape(3)
    if isinstance(obs, torch.Tensor):
        obs[:, start:end] = torch.as_tensor(value, device=obs.device, dtype=obs.dtype)
    else:
        obs[:, start:end] = value.astype(obs.dtype, copy=False)


def sync_velocity_into_session_obs(session: Any, command: Go2ArmManipLocoCommand) -> None:
    """Make this step's policy obs see the keyboard command (not the previous frame)."""
    if session.obs is None:
        return
    history = _actor_history_length(session.env)
    vel = np.asarray(command.vel_command_b[0], dtype=np.float32)
    obs = session.obs
    if isinstance(obs, torch.Tensor):
        _write_last_frame_command(obs, vel, history)
    else:
        for key in ("actor", "policy"):
            tensor = obs.get(key) if hasattr(obs, "get") else None
            if isinstance(tensor, torch.Tensor):
                _write_last_frame_command(tensor, vel, history)
    state = getattr(session.env, "state", None)
    env_obs = getattr(state, "obs", None)
    mapped = env_obs.get("obs") if isinstance(env_obs, dict) else None
    if isinstance(mapped, np.ndarray):
        _write_last_frame_command(mapped, vel, history)


def push_keyboard_command(
    session: Any,
    command: Go2ArmManipLocoCommand,
    commander: KeyboardCommander,
    ee_commander: EndEffectorCommander | None = None,
) -> None:
    apply_keyboard_velocity(command, commander)
    if ee_commander is not None:
        apply_keyboard_ee(command, ee_commander)
    sync_velocity_into_session_obs(session, command)


def episode_requires_reset(env: Any) -> bool:
    """True after a done step when auto-reset is off and the next step would fail.

    Only ``_manual_reset_pending`` is authoritative. ``state.terminated`` stays
    True after ``init_state()``'s internal autoreset path, so treating it as
    done would reset every subsequent frame.
    """
    pending = getattr(env, "_manual_reset_pending", None)
    return isinstance(pending, np.ndarray) and bool(np.any(pending))


def reset_held_episode(
    session: Any,
    command: Go2ArmManipLocoCommand,
    commander: KeyboardCommander,
    ee_commander: EndEffectorCommander | None = None,
) -> bool:
    """Reset a finished episode and keep the current keyboard velocity and EE pose."""
    if not episode_requires_reset(session.env):
        return False
    session.reset()
    push_keyboard_command(session, command, commander, ee_commander)
    return True


def _select_device(cfg: DictConfig) -> str:
    if cfg.training.device is not None:
        return str(cfg.training.device)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _commander_from_env(env: Any, cfg: DictConfig) -> KeyboardCommander:
    command = task_velocity_command(env)
    return KeyboardCommander.from_vel_limit(
        command._task_cfg.command_config.vel_limit,
        step_lin=float(
            OmegaConf.select(
                cfg, "interactive.keyboard_step_lin", default=_DEFAULT_KEYBOARD_STEP_LIN
            )
        ),
        step_ang=float(
            OmegaConf.select(
                cfg, "interactive.keyboard_step_ang", default=_DEFAULT_KEYBOARD_STEP_ANG
            )
        ),
    )


def _ee_commander_from_env(
    command: Go2ArmManipLocoCommand, cfg: DictConfig
) -> EndEffectorCommander:
    return EndEffectorCommander.from_command(
        command,
        step=float(
            OmegaConf.select(cfg, "interactive.keyboard_step_ee", default=_DEFAULT_KEYBOARD_STEP_EE)
        ),
    )


def apply_viewer_key(
    keycode: int,
    *,
    controls: PlaybackControls,
    commander: KeyboardCommander,
    session: Any,
    command: Go2ArmManipLocoCommand,
    ee_commander: EndEffectorCommander | None = None,
) -> None:
    if keycode == ord(" "):
        paused = controls.toggle_pause()
        print(f"[keyboard-play] {'paused' if paused else 'resumed'} (space)")
        return
    if keycode in (ord("N"), ord("n")):
        controls.request_single_step()
        if not controls.paused:
            controls.pause()
            print("[keyboard-play] paused for single-step mode (n)")
        print("[keyboard-play] single step requested (n)")
        return
    if keycode in (ord("+"), ord("=")):
        controls.set_speed(controls.speed * 1.25)
        print(f"[keyboard-play] speed={controls.speed:.2f}x")
        return
    if keycode in (ord("-"), ord("_")):
        controls.set_speed(controls.speed / 1.25)
        print(f"[keyboard-play] speed={controls.speed:.2f}x")
        return
    if keycode == _KEY_BACKSPACE:
        commander.zero()
        if ee_commander is not None:
            ee_commander.go_home()
        session.reset()
        push_keyboard_command(session, command, commander, ee_commander)
        print("[keyboard-play] reset (backspace)")
        return
    if _handle_command_key(commander, keycode):
        return
    if ee_commander is not None:
        _handle_ee_key(ee_commander, keycode)


def _handle_command_key(commander: KeyboardCommander, keycode: int) -> bool:
    if keycode == _KEY_UP:
        commander.nudge(commander.AXIS_VX, +1.0)
    elif keycode == _KEY_DOWN:
        commander.nudge(commander.AXIS_VX, -1.0)
    elif keycode == _KEY_LEFT:
        commander.nudge(commander.AXIS_VYAW, +1.0)
    elif keycode == _KEY_RIGHT:
        commander.nudge(commander.AXIS_VYAW, -1.0)
    elif keycode in (ord("a"), ord("A")):
        commander.nudge(commander.AXIS_VY, +1.0)
    elif keycode in (ord("d"), ord("D")):
        commander.nudge(commander.AXIS_VY, -1.0)
    elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
        commander.zero()
    else:
        return False
    print(f"[keyboard-play] {commander.describe()}")
    return True


def _handle_ee_key(ee_commander: EndEffectorCommander, keycode: int) -> bool:
    if keycode in (ord("u"), ord("U")):
        ee_commander.nudge(ee_commander.AXIS_X, +1.0)
    elif keycode in (ord("j"), ord("J")):
        ee_commander.nudge(ee_commander.AXIS_X, -1.0)
    elif keycode in (ord("h"), ord("H")):
        ee_commander.nudge(ee_commander.AXIS_Y, +1.0)
    elif keycode in (ord("k"), ord("K")):
        ee_commander.nudge(ee_commander.AXIS_Y, -1.0)
    elif keycode in (ord("y"), ord("Y")):
        ee_commander.nudge(ee_commander.AXIS_Z, +1.0)
    elif keycode in (ord("o"), ord("O")):
        ee_commander.nudge(ee_commander.AXIS_Z, -1.0)
    elif keycode in (ord("l"), ord("L")):
        ee_commander.go_home()
    else:
        return False
    print(f"[keyboard-play] {ee_commander.describe()}")
    return True


def _load_viewer_model(env: Any):
    import mujoco

    try:
        with tempfile.TemporaryDirectory(prefix="legged-keyboard-play-") as tmp_dir:
            model_files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=tmp_dir)
            model_file = model_files[0] if isinstance(model_files, list) else model_files
            return mujoco.MjModel.from_xml_path(str(model_file))
    except Exception as exc:
        print(f"[keyboard-play] visual playback model unavailable ({exc}); using scene XML")
    scene = getattr(getattr(env, "cfg", None), "scene", None)
    model_file = None if scene is None else getattr(scene, "model_file", None)
    if model_file:
        return mujoco.MjModel.from_xml_path(str(model_file))
    playback_model = env.get_playback_model()
    if isinstance(playback_model, str):
        return mujoco.MjModel.from_xml_path(playback_model)
    return playback_model


def _open_ppo_session(cfg: DictConfig, device: str, *, num_envs: int):
    from unilab.scripts.train_rsl_rl import (
        apply_ppo_runtime_flags,
        build_ppo_play_env_cfg_override,
    )

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise RuntimeError("rsl_rl is required for PPO keyboard play") from exc

    rl_cfg = algo_config_dict(cfg)
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            format_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    def _normalize_play_train_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
        normalized = cast("dict[str, Any]", normalize_ppo_train_cfg(train_cfg))
        apply_ppo_runtime_flags(normalized, cfg, training_enabled=False)
        return normalized

    resources = ExitStack()
    play_env_cfg_override = build_ppo_play_env_cfg_override(cfg)

    def _create_env(n: int):
        created = create_env(cfg, num_envs=n, env_cfg_override=play_env_cfg_override)
        resources.callback(created.close)
        return created

    try:
        session, _mode, _path = create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task=str(cfg.training.task_name),
                load_run=str(cfg.algo.load_run),
                checkpoint=normalize_checkpoint_value(getattr(cfg.algo, "checkpoint", None)),
                action_mode="policy",
                policy_obs_mode="flat",
                algo_log_name=str(cfg.algo.algo_log_name),
                log_root=getattr(cfg.training, "log_root", None),
                num_envs=num_envs,
            ),
            env_factory=_create_env,
            algo_config=rl_cfg,
            root_dir=ROOT_DIR,
            device=device,
            checkpoint_resolver=lambda *_args: str(load_path),
            checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
            entrypoint_log_root=get_entrypoint_log_root,
            wrapper_cls=resolve_rsl_rl_ppo_runtime(
                rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper
            ).wrapper_cls
            or RslRlVecEnvWrapper,
            runner_cls=resolve_rsl_rl_ppo_runtime(
                rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper
            ).runner_cls
            or OnPolicyRunner,
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=_normalize_play_train_cfg,
            sim2sim_preflight=make_sim2sim_preflight(cfg, algo_name="ppo"),
            guard_algo_name="ppo",
        )
    except Exception:
        resources.close()
        raise
    return resources, session


def _open_him_session(cfg: DictConfig, device: str, *, num_envs: int):
    rl_cfg = algo_config_dict(cfg)
    task_log_root = get_log_root(ROOT_DIR, cfg) / str(cfg.training.task_name)
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            format_play_checkpoint_error(
                cfg,
                task_log_root=task_log_root,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    _ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "actor_state_dict" not in _ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not a HIM-PPO checkpoint "
            f"(found keys: {_ckpt_keys}). Aborting play."
        )
        return None

    _restore_trained_arm_stage(cfg, load_path_dir)
    preflight = make_sim2sim_preflight(cfg, algo_name="ppo")
    if preflight is not None:
        preflight(str(load_path_dir))
    expected = int(cfg.algo.num_one_step_obs) * int(cfg.algo.num_actor_history)
    state = torch.load(load_path, map_location="cpu", weights_only=True)["actor_state_dict"]
    weight = state.get("estimator.encoder.0.weight")
    if isinstance(weight, torch.Tensor) and weight.shape[1] != expected:
        raise ValueError(
            f"HIM checkpoint observation dimension mismatch: {weight.shape[1]} != {expected}"
        )

    resources = ExitStack()

    def _create_env(n: int):
        env_cfg_override = cast(
            dict[str, Any], _him_backend_adapter(cfg).build_play_env_cfg_override()
        )
        created = create_env(cfg, num_envs=n, env_cfg_override=env_cfg_override)
        resources.callback(created.close)
        return created

    try:
        session, _mode, _path = create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task=str(cfg.training.task_name),
                load_run=str(getattr(cfg.algo, "load_run", "-1")),
                checkpoint=normalize_checkpoint_value(getattr(cfg.algo, "checkpoint", None)),
                action_mode="policy",
                policy_obs_mode="flat",
                algo_log_name=str(cfg.algo.algo_log_name),
                log_root=getattr(cfg.training, "log_root", None),
                num_envs=num_envs,
            ),
            env_factory=_create_env,
            algo_config=rl_cfg,
            root_dir=ROOT_DIR,
            device=device,
            checkpoint_resolver=lambda *_args: str(load_path),
            checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
            entrypoint_log_root=get_entrypoint_log_root,
            wrapper_cls=RslRlVecEnvWrapper,
            runner_cls=HIMOnPolicyRunner,
            runner_loader=lambda runner, path: runner.load(path),
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=lambda train_cfg: train_cfg,
            sim2sim_preflight=lambda _path: None,
            guard_algo_name="him_ppo",
        )
        if isinstance(session.runner, HIMOnPolicyRunner):
            resources.callback(session.runner.close)
        him_policy = session.policy
        if him_policy is not None:
            session.policy = lambda obs: him_policy(obs["actor"])
    except Exception:
        resources.close()
        raise
    return resources, session


def run_keyboard_viewer(session: Any, cfg: DictConfig) -> None:
    import mujoco
    import mujoco.viewer

    env = session.env
    command = bind_velocity_teleop(env)
    bind_ee_teleop(command)
    commander = _commander_from_env(env, cfg)
    ee_commander = _ee_commander_from_env(command, cfg)
    apply_keyboard_velocity(command, commander)
    apply_keyboard_ee(command, ee_commander)
    controls = PlaybackControls(
        paused=bool(OmegaConf.select(cfg, "interactive.start_paused", default=False)),
        speed=float(OmegaConf.select(cfg, "interactive.speed", default=1.0)),
    )
    mj_model = _load_viewer_model(env)
    viz_data = mujoco.MjData(mj_model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.02))

    session.reset()
    push_keyboard_command(session, command, commander, ee_commander)
    pending_keys = PendingKeyQueue()

    def _on_key(keycode: int) -> None:
        pending_keys.push(keycode)

    print("[keyboard-play] Keyboard teleop ENABLED:")
    print("  Up / Down    : chassis forward / backward (vx)")
    print("  Left / Right : chassis turn left / right  (vyaw)")
    print("  A / D        : chassis strafe left / right (vy)")
    print("  Enter        : chassis full stop")
    print("  U / J        : EE +x / -x (arm-base frame)")
    print("  H / K        : EE +y / -y")
    print("  Y / O        : EE +z / -z")
    print("  L            : EE home (task init pose)")
    print("  Backspace    : reset episode (zeros velocity, homes EE)")
    print("  Space        : pause / resume")
    print("  Timeout / fallen terminations are disabled; only Backspace resets.")
    print(
        f"  Linear step is {commander.step_lin:.2f} m/s "
        f"(gait/policy treat |cmd| <= {_VELOCITY_STAND_DEADZONE} as stand)."
    )
    print(f"  EE step is {ee_commander.step:.3f} m, clipped to the spherical workspace.")

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        with torch.inference_mode():
            while viewer.is_running():
                t0 = time.perf_counter()
                for keycode in pending_keys.drain():
                    apply_viewer_key(
                        keycode,
                        controls=controls,
                        commander=commander,
                        session=session,
                        command=command,
                        ee_commander=ee_commander,
                    )
                push_keyboard_command(session, command, commander, ee_commander)
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


def play_with_keyboard(cfg: DictConfig, *, algo: str) -> None:
    if str(cfg.training.sim_backend) != "mujoco":
        raise SystemExit("keyboard play requires --sim mujoco (MuJoCo viewer)")
    device = _select_device(cfg)
    print(f"[keyboard-play] Device: {device}")
    opener = _open_him_session if algo == "him_ppo" else _open_ppo_session
    opened = opener(cfg, device, num_envs=1)
    if opened is None:
        return
    resources, session = opened
    with resources:
        run_keyboard_viewer(session, cfg)
