"""Manager-Based action, command, observation, reward, and termination terms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from unilab.base.entity import Entity
from unilab.dtype_config import get_global_dtype
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unilab.managers.command_manager import CommandTerm, CommandTermCfg
from unilab.managers.manager_base import ManagerTermBase
from unilab.utils.rotation import np_matrix_from_quat, np_quat_from_euler_xyz

from legged_manipulation_unilab.tasks.geometry import (
    np_cartesian_to_spherical as _cart2sphere,
)
from legged_manipulation_unilab.tasks.geometry import np_quat_orientation_error_local
from legged_manipulation_unilab.tasks.geometry import (
    np_spherical_to_cartesian as _sphere2cart,
)


def _task_cfg(env: ManagerBasedRlEnv) -> Any:
    cfg = env._cfg
    if not hasattr(cfg, "goal_ee"):
        raise TypeError("Go2Arm manager terms require Go2ArmManipLocoCfg")
    return cfg


def _command(env: Any) -> "Go2ArmManipLocoCommand":
    term = env.command_manager.get_term("task")
    if not isinstance(term, Go2ArmManipLocoCommand):
        raise TypeError(f"Expected Go2ArmManipLocoCommand, got {type(term).__name__}")
    return term


_UPVECTOR_SENSOR = "upvector"


def _sensor_quat_wxyz(backend: Any, quat: np.ndarray) -> np.ndarray:
    """Return a framequat sensor reading in the wxyz contract order.

    MotrixSim emits framequat sensor values in xyzw order.  unisim's motrix
    backend normalizes body-track and free-joint quaternions to wxyz but not
    named framequat sensors, while the mujoco path already emits wxyz; without
    this flip the arm IK rotates its Jacobian with a garbage base frame and
    saturates ``dq_clip`` every step (20+ rad/s arm thrashing).  Remove once
    the backend converts framequat sensors upstream.
    """
    if backend.backend_type == "motrix":
        quat = np.asarray(quat, dtype=get_global_dtype())[:, [3, 0, 1, 2]]
    return quat


_ARM_TOUCH_SENSORS = (
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


@dataclass(kw_only=True)
class Go2ArmIKActionCfg(ActionTermCfg):
    """Flat 18-DOF action: leg position targets plus Jacobian-guided arm targets."""

    def build(self, env: "ManagerBasedRlEnv") -> "Go2ArmIKAction":  # pyright: ignore[reportIncompatibleMethodOverride]
        return Go2ArmIKAction(self, env)


class Go2ArmIKAction(ActionTerm):
    cfg: Go2ArmIKActionCfg

    def __init__(self, cfg: Go2ArmIKActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        cfg_task = _task_cfg(env)
        self._task_cfg = cfg_task
        entity: Entity = env.scene[cfg.entity_name]
        joint_ids, joint_names = entity.find_joints_by_actuator_names((".*",))
        if len(joint_ids) != 18:
            raise ValueError(f"Go2Arm manager action expects 18 actuators, got {len(joint_ids)}")
        self._entity = entity
        self._joint_ids = np.asarray(joint_ids, dtype=np.intp)
        self._joint_ids.setflags(write=False)
        self._joint_names = tuple(joint_names)
        arm_names = set(cfg_task.asset.arm_joint_names)
        self._arm_rows = np.asarray(
            [index for index, name in enumerate(joint_names) if name in arm_names], dtype=np.intp
        )
        if self._arm_rows.size != 6:
            raise ValueError("Go2Arm manager action could not resolve all six arm actuators")
        self._site_id = int(env._backend.get_site_ids([cfg_task.asset.ee_site_name])[0])
        self._arm_dof_ids = np.asarray(
            env._backend.get_joint_dof_indices(cfg_task.asset.arm_joint_names), dtype=np.intp
        )
        self._ee_pos_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_pos,))
        self._ee_quat_view = env.scene.bind_sensor_data((cfg_task.sensor.ee_local_quat,))
        self._arm_ref_quat_view = env.scene.bind_sensor_data((cfg_task.sensor.arm_ref_world_quat,))
        dtype = get_global_dtype()
        self._raw_actions = np.zeros((env.num_envs, 18), dtype=dtype)
        self._target = np.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 18

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_actions

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        self._raw_actions[env_ids if env_ids is not None else slice(None)] = 0.0

    def process_actions(self, actions: np.ndarray) -> None:
        if actions.shape != (self.num_envs, 18):
            raise ValueError(f"Go2Arm action shape must be {(self.num_envs, 18)}")
        if not np.isfinite(actions).all():
            raise ValueError("Go2Arm action contains NaN or Inf")
        self._raw_actions[:] = actions
        if self._task_cfg.arm_stage.freeze_arm_joints:
            self._raw_actions[:, 12:18] = 0.0
        exec_actions = (
            self._env.action_manager.prev_action
            if self._task_cfg.control_config.simulate_action_latency
            else self._raw_actions
        )
        default = self._entity.data.default_joint_pos
        leg_target = exec_actions[:, :12] * self._task_cfg.control_config.action_scale
        leg_target += default[:, :12]
        if self._task_cfg.arm_stage.freeze_arm_joints:
            arm_target = np.broadcast_to(default[:, 12:18], (self.num_envs, 6)).copy()
        else:
            command = _command(self._env)
            backend = getattr(self._env, "_backend")
            ee_pos = np.asarray(self._ee_pos_view.read(), dtype=get_global_dtype())
            ee_quat = _sensor_quat_wxyz(backend, self._ee_quat_view.read())
            arm_pos = self._entity.data.joint_pos[:, self._joint_ids[self._arm_rows]]
            dq = self._compute_arm_ik_delta(
                command.curr_ee_goal_cart,
                ee_pos,
                command.ee_goal_orn_quat,
                ee_quat,
            )
            arm_target = (
                arm_pos
                + exec_actions[:, 12:18] * self._task_cfg.control_config.arm_action_scale
                + self._task_cfg.ik.gain * dq
            )
        self._target[:, :12] = leg_target
        self._target[:, 12:] = arm_target
        ranges = self._entity.data.actuator_ctrl_range
        np.clip(self._target, ranges[:, 0], ranges[:, 1], out=self._target)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def _compute_arm_ik_delta(
        self,
        goal_pos: np.ndarray,
        curr_pos: np.ndarray,
        goal_quat: np.ndarray,
        curr_quat: np.ndarray,
    ) -> np.ndarray:
        cfg = self._task_cfg.ik
        pos_err = goal_pos - curr_pos
        backend = getattr(self._env, "_backend")
        jacp_w, jacr_w = backend.get_site_jacobian_w(
            self._site_id,
            self._arm_dof_ids,
        )
        ref_rot_w = np_matrix_from_quat(
            _sensor_quat_wxyz(backend, self._arm_ref_quat_view.read())
        )
        rot_w_to_b = np.swapaxes(ref_rot_w, 1, 2)
        jacp_b = np.matmul(rot_w_to_b, jacp_w)
        if cfg.use_orientation:
            if cfg.orientation_mode == "target":
                orn_err = np_quat_orientation_error_local(goal_quat, curr_quat)
            elif cfg.orientation_mode == "zero_error":
                orn_err = np.zeros_like(pos_err)
            else:
                raise ValueError("ik.orientation_mode must be 'target' or 'zero_error'")
            jac = np.concatenate([jacp_b, np.matmul(rot_w_to_b, jacr_w)], axis=1)
            dpose = np.concatenate([pos_err, orn_err], axis=1)
        else:
            jac = jacp_b
            dpose = pos_err
        identity = np.eye(jac.shape[1], dtype=jac.dtype)[None, :, :]
        lhs = np.matmul(jac, np.swapaxes(jac, 1, 2)) + identity * (cfg.damping**2)
        solved = np.linalg.solve(lhs, dpose[:, :, None])
        dq = np.matmul(np.swapaxes(jac, 1, 2), solved)[:, :, 0]
        if cfg.dq_clip > 0.0:
            np.clip(dq, -cfg.dq_clip, cfg.dq_clip, out=dq)
        return dq


@dataclass(kw_only=True)
class Go2ArmManipLocoCommandCfg(CommandTermCfg):
    def build(self, env: "ManagerBasedRlEnv") -> "Go2ArmManipLocoCommand":  # pyright: ignore[reportIncompatibleMethodOverride]
        return Go2ArmManipLocoCommand(self, env)


class Go2ArmManipLocoCommand(CommandTerm):
    """Velocity command plus task-owned EE trajectory and gait phase state."""

    cfg: Go2ArmManipLocoCommandCfg

    def __init__(self, cfg: Go2ArmManipLocoCommandCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        cfg_task = _task_cfg(env)
        self._task_cfg = cfg_task
        n = env.num_envs
        dtype = get_global_dtype()
        self.vel_command_b = np.zeros((n, 3), dtype=dtype)
        self.phase = np.zeros(n, dtype=np.float32)
        self.feet_phase = np.zeros((n, len(cfg_task.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = float(cfg_task.command_config.gait_frequency)
        self.curr_ee_goal_cart = np.zeros((n, 3), dtype=dtype)
        self.curr_ee_goal_sphere = np.zeros((n, 3), dtype=dtype)
        self.curr_ee_goal_world = np.zeros((n, 3), dtype=dtype)
        self.ee_goal_orn_euler = np.zeros((n, 3), dtype=dtype)
        self.ee_goal_orn_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype), (n, 1))
        self._ee_start_sphere = np.zeros((n, 3), dtype=dtype)
        self._ee_goal_sphere = np.zeros((n, 3), dtype=dtype)
        self._arm_goal_timer = np.zeros(n, dtype=np.int32)
        self._traj_steps = np.ones(n, dtype=np.int32)
        self._traj_total_steps = np.ones(n, dtype=np.int32)
        self._episode_tracking_sum = np.zeros(n, dtype=np.float64)
        self._episode_steps = np.zeros(n, dtype=np.int32)
        self._expansion_count = 0
        self._ctrl_dt = float(env.step_dt)
        self._metrics = {
            "tracking_lin_vel": np.zeros(n, dtype=np.float32),
            "episode_steps": np.zeros(n, dtype=np.float32),
        }
        self._resetting_ids: np.ndarray | None = None
        self._goal_view = env.scene.bind_sensor_data(("armbasepoint_world_pos",))
        self._goal_quat_view = env.scene.bind_sensor_data(("armbasepoint_world_quat",))

    @property
    def command(self) -> np.ndarray:
        return self.vel_command_b

    def reset(self, env_ids: np.ndarray | slice | None) -> dict[str, float]:
        ids = env_ids if isinstance(env_ids, np.ndarray) else np.arange(self.num_envs)
        if not isinstance(ids, np.ndarray):  # narrows for type checkers
            raise TypeError("Command reset IDs must be an ndarray")
        self._resetting_ids = ids.copy()
        try:
            result = super().reset(ids)
        finally:
            self._resetting_ids = None
        # Update the curriculum from the finished episode after the reset
        # resample (legacy reset-plan ordering: the fresh commands still use
        # the pre-expansion range), then clear the episode accumulators.
        self._update_command_curriculum(ids)
        self._episode_tracking_sum[ids] = 0.0
        self._episode_steps[ids] = 0
        return result

    def record_tracking(self, value: np.ndarray) -> None:
        """Accumulate the unweighted tracking_lin_vel value for the curriculum."""
        self._episode_tracking_sum += np.asarray(value, dtype=np.float64)

    def _update_command_curriculum(self, env_ids: np.ndarray) -> None:
        """Expand vel_limit when the finished episodes tracked well (legacy rule).

        Mean per-step unweighted tracking_lin_vel above the threshold expands
        the sampling range by ``step_size``, clamped to ``max_vel_limit``.
        Without this the tiny initial range keeps standing near-optimal and
        velocity tracking is never trained.
        """
        curriculum = self._task_cfg.curriculum_config
        if not curriculum.enable:
            return
        ep_steps = np.maximum(self._episode_steps[env_ids], 1)
        mean_per_step = float(np.mean(self._episode_tracking_sum[env_ids] / ep_steps))
        if mean_per_step > float(curriculum.threshold):
            step = np.asarray(curriculum.step_size, dtype=np.float64)
            max_limit = np.asarray(curriculum.max_vel_limit, dtype=np.float64)
            low = np.asarray(self._task_cfg.command_config.vel_limit[0], dtype=np.float64)
            high = np.asarray(self._task_cfg.command_config.vel_limit[1], dtype=np.float64)
            self._task_cfg.command_config.vel_limit = [
                np.clip(low - step, -max_limit, 0.0).tolist(),
                np.clip(high + step, 0.0, max_limit).tolist(),
            ]
            self._expansion_count += 1
            print(
                f"[curriculum] step {self._env.common_step_counter}: vel_limit -> "
                f"low={self._task_cfg.command_config.vel_limit[0]} "
                f"high={self._task_cfg.command_config.vel_limit[1]} "
                f"(expansion #{self._expansion_count}, batch mean {mean_per_step:.3f})"
            )

    def _update_metrics(self, env_ids: np.ndarray | None = None) -> None:
        del env_ids

    def _sample_velocity(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg.command_config
        count = len(env_ids)
        low = np.asarray(cfg.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(cfg.vel_limit[1], dtype=get_global_dtype())
        commands = self._env.rng.uniform(low, high, size=(count, 3)).astype(get_global_dtype())
        commands[np.all(np.abs(commands) <= 0.1, axis=1)] = 0.0
        prob = float(cfg.zero_command_prob)
        if prob > 0.0:
            zero = self._env.rng.random(count) < prob
            commands[zero] = 0.0
        self.vel_command_b[env_ids] = commands

    def _resample_command(self, env_ids: np.ndarray) -> None:
        self._sample_velocity(env_ids)
        resetting = self._resetting_ids
        if resetting is not None and np.array_equal(env_ids, resetting):
            self._reset_ee_goals(env_ids)
        self._write_feet_phase(env_ids)

    def _update_command(self, env_ids: np.ndarray | None) -> None:
        # Base calls this after both episode and mid-episode resampling.  Mid-episode
        # calls are full-batch and therefore advance the phase/trajectory; reset
        # calls scope only reset rows and leave their newly sampled phase at zero.
        if env_ids is not None:
            return
        cfg = self._task_cfg
        moving = self._command_is_moving(self.vel_command_b)
        advanced = np.fmod(self.phase + cfg.ctrl_dt * self.gait_frequency, 1.0)
        self.phase = np.where(moving, advanced, 0.0).astype(np.float32)
        self._write_feet_phase(slice(None), moving)
        self._update_ee_trajectory()
        self._episode_steps += 1
        self._log_command_state()

    def _log_command_state(self) -> None:
        """Publish the live velocity range so curriculum progress is observable."""
        extras = getattr(self._env, "extras", None)
        log = extras.get("log") if isinstance(extras, dict) else None
        if not isinstance(log, dict):
            return
        low, high = self._task_cfg.command_config.vel_limit
        for axis, label in enumerate("xyz"):
            log[f"command/vel_low_{label}"] = float(low[axis])
            log[f"command/vel_high_{label}"] = float(high[axis])
        log["command/curriculum_expansions"] = self._expansion_count

    def post_compute(self) -> None:
        pos = np.asarray(self._goal_view.read(), dtype=get_global_dtype())
        quat = _sensor_quat_wxyz(self._env._backend, self._goal_quat_view.read())
        rotation = np_matrix_from_quat(quat)
        self.curr_ee_goal_world[:] = pos + np.einsum("nij,nj->ni", rotation, self.curr_ee_goal_cart)

    def _command_is_moving(self, commands: np.ndarray) -> np.ndarray:
        return np.any(np.abs(np.asarray(commands)[:, :3]) > 0.1, axis=1)

    def _write_feet_phase(
        self, env_ids: np.ndarray | slice, moving: np.ndarray | None = None
    ) -> None:
        phase = self.phase[env_ids]
        feet = self.feet_phase[env_ids].copy()
        feet[:, 0] = phase
        feet[:, 3] = phase
        feet[:, 1] = (phase + 0.5) % 1.0
        feet[:, 2] = (phase + 0.5) % 1.0
        if moving is None:
            moving = self._command_is_moving(self.vel_command_b[env_ids])
        feet[~moving] = 0.0
        self.feet_phase[env_ids] = feet

    def _sample_timing(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg.goal_ee
        count = len(env_ids)
        traj = self._env.rng.uniform(*cfg.traj_time_range, size=count)
        hold = self._env.rng.uniform(*cfg.hold_time_range, size=count)
        self._traj_steps[env_ids] = np.maximum(1, np.round(traj / self._ctrl_dt).astype(np.int32))
        self._traj_total_steps[env_ids] = self._traj_steps[env_ids] + np.maximum(
            0, np.round(hold / self._ctrl_dt).astype(np.int32)
        )

    def _collision_unsafe(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        cfg = self._task_cfg.goal_ee
        count = max(2, cfg.num_collision_check_samples)
        fraction = np.linspace(0.0, 1.0, count, dtype=get_global_dtype())
        path = starts[:, None, :] + (goals - starts)[:, None, :] * fraction[None, :, None]
        cartesian = _sphere2cart(path.reshape(-1, 3)).reshape(len(starts), count, 3)
        upper = np.asarray(cfg.collision_upper_limits, dtype=get_global_dtype())
        lower = np.asarray(cfg.collision_lower_limits, dtype=get_global_dtype())
        inside = np.all(cartesian < upper, axis=2) & np.all(cartesian > lower, axis=2)
        underground = np.any(cartesian[:, :, 2] < cfg.underground_limit, axis=1)
        return np.any(inside, axis=1) | underground

    def _sample_goal_spheres(self, env_ids: np.ndarray, starts: np.ndarray) -> None:
        cfg = self._task_cfg.goal_ee
        dtype = get_global_dtype()
        candidates = np.broadcast_to(starts, (len(env_ids), 3)).copy()
        remaining = np.arange(len(env_ids), dtype=np.int32)
        for _ in range(max(1, cfg.num_resample_attempts)):
            samples = np.column_stack(
                [
                    self._env.rng.uniform(*cfg.sphere_l_range, size=len(remaining)),
                    self._env.rng.uniform(*cfg.sphere_phi_range, size=len(remaining)),
                    self._env.rng.uniform(*cfg.sphere_theta_range, size=len(remaining)),
                ]
            ).astype(dtype)
            candidates[remaining] = samples
            unsafe = self._collision_unsafe(starts[remaining], samples)
            remaining = remaining[unsafe]
            if remaining.size == 0:
                break
        self._ee_goal_sphere[env_ids] = candidates

    def _sample_goal_orientation(self, env_ids: np.ndarray, *, is_init: bool) -> None:
        cfg = self._task_cfg.goal_ee
        count = len(env_ids)
        base_roll = np.full(count, cfg.default_orn_roll, dtype=get_global_dtype())
        pitch = np.full(count, cfg.arm_induced_pitch, dtype=get_global_dtype())
        yaw = np.zeros(count, dtype=get_global_dtype())
        if is_init:
            roll_delta = self._env.rng.uniform(*cfg.delta_orn_r, size=count)
            pitch_delta = self._env.rng.uniform(*cfg.delta_orn_p, size=count)
            yaw_delta = self._env.rng.uniform(*cfg.delta_orn_y, size=count)
        else:
            roll_delta = self._env.rng.uniform(*cfg.delta_orn_r, size=count)
            pitch_delta = self._env.rng.uniform(*cfg.delta_orn_p, size=count)
            yaw_delta = self._env.rng.uniform(*cfg.delta_orn_y, size=count)
        self.ee_goal_orn_euler[env_ids, 0] = base_roll + roll_delta
        self.ee_goal_orn_euler[env_ids, 1] = pitch + pitch_delta
        self.ee_goal_orn_euler[env_ids, 2] = yaw + yaw_delta
        quat = np_quat_from_euler_xyz(
            self.ee_goal_orn_euler[env_ids, 0],
            self.ee_goal_orn_euler[env_ids, 1],
            self.ee_goal_orn_euler[env_ids, 2],
        )
        self.ee_goal_orn_quat[env_ids] = np.asarray(quat, dtype=get_global_dtype())

    def _reset_ee_goals(self, env_ids: np.ndarray) -> None:
        cfg = self._task_cfg
        stage = cfg.arm_stage
        if stage.disable_ee_goal_trajectory:
            fixed = np.asarray(stage.fixed_ee_goal_cart, dtype=get_global_dtype())[None, :]
            self._ee_start_sphere[env_ids] = fixed
            self._ee_goal_sphere[env_ids] = fixed
            self._traj_steps[env_ids] = 1
            self._traj_total_steps[env_ids] = 1
            self._arm_goal_timer[env_ids] = 0
            self.curr_ee_goal_sphere[env_ids] = fixed
            self.curr_ee_goal_cart[env_ids] = _sphere2cart(fixed)
            self._sample_goal_orientation(env_ids, is_init=True)
            return
        init_sphere = _cart2sphere(
            np.asarray(cfg.goal_ee.init_ee_cart, dtype=get_global_dtype())[None, :]
        )[0]
        self._ee_start_sphere[env_ids] = init_sphere
        self._sample_goal_spheres(env_ids, np.broadcast_to(init_sphere, (len(env_ids), 3)).copy())
        self._sample_timing(env_ids)
        self._arm_goal_timer[env_ids] = 0
        self.curr_ee_goal_sphere[env_ids] = init_sphere
        self.curr_ee_goal_cart[env_ids] = _sphere2cart(
            np.broadcast_to(init_sphere, (len(env_ids), 3))
        )
        self._sample_goal_orientation(env_ids, is_init=True)

    def _update_ee_trajectory(self) -> None:
        cfg = self._task_cfg
        stage = cfg.arm_stage
        if stage.disable_ee_goal_trajectory:
            fixed = np.asarray(stage.fixed_ee_goal_cart, dtype=get_global_dtype())[None, :]
            self.curr_ee_goal_cart[:] = fixed
            self.curr_ee_goal_sphere[:] = _cart2sphere(fixed)
        else:
            self._arm_goal_timer += 1
            expired = np.flatnonzero(self._arm_goal_timer >= self._traj_total_steps).astype(
                np.int32
            )
            if expired.size:
                self._ee_start_sphere[expired] = self._ee_goal_sphere[expired]
                self._sample_goal_spheres(expired, self._ee_start_sphere[expired])
                self._sample_goal_orientation(expired, is_init=False)
                self._sample_timing(expired)
                self._arm_goal_timer[expired] = 0
            fraction = np.clip(self._arm_goal_timer / self._traj_steps, 0.0, 1.0).astype(
                get_global_dtype()
            )[:, None]
            self.curr_ee_goal_sphere[:] = (
                self._ee_start_sphere + (self._ee_goal_sphere - self._ee_start_sphere) * fraction
            )
            self.curr_ee_goal_cart[:] = _sphere2cart(self.curr_ee_goal_sphere)


class Go2ArmObservation(ManagerTermBase):
    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._task_cfg = _task_cfg(env)
        names = (
            self._task_cfg.sensor.local_linvel,
            self._task_cfg.sensor.gyro,
            "upvector",
            self._task_cfg.sensor.ee_local_pos,
        )
        self._views = env.scene.bind_sensor_data(names)
        if self._views.dimensions != (3, 3, 3, 3):
            raise ValueError(f"Unexpected Go2Arm sensor dimensions: {self._views.dimensions}")
        self._entity: Entity = env.scene["robot"]

    def __call__(self, env: "ManagerBasedRlEnv", actor: bool) -> np.ndarray:
        del env
        raw = self._raw_observation(actor)
        return raw[:, 3:] if actor else raw

    def _raw_observation(self, actor: bool) -> np.ndarray:
        cfg = self._task_cfg
        sensors = np.asarray(self._views.read(), dtype=get_global_dtype())
        linvel = sensors[:, 0:3]
        gyro = sensors[:, 3:6]
        gravity = sensors[:, 6:9]
        ee_pos = sensors[:, 9:12]
        joint_pos = self._entity.data.joint_pos - self._entity.data.default_joint_pos
        joint_vel = self._entity.data.joint_vel
        command = _command(self._env)
        if actor:
            noise = cfg.noise_config
            level = noise.level
            if level > 0.0:

                def add(value: np.ndarray, scale: float) -> np.ndarray:
                    sample = self._env.rng.uniform(-1.0, 1.0, value.shape)
                    return value + sample.astype(get_global_dtype()) * level * scale

                linvel = add(linvel, noise.scale_linvel)
                gyro = add(gyro, noise.scale_gyro)
                gravity = add(gravity, noise.scale_gravity)
                joint_pos = add(joint_pos, noise.scale_joint_angle)
                joint_vel = add(joint_vel, noise.scale_joint_vel)
                ee_pos = add(ee_pos, noise.scale_ee_pos)
        goal = command.curr_ee_goal_cart
        return np.concatenate(
            [
                linvel,
                gyro,
                -gravity,
                command.vel_command_b,
                command.feet_phase,
                joint_pos,
                joint_vel,
                ee_pos,
                goal,
                ee_pos - goal,
                self._env.action_manager.action,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )


class Go2ArmReward(ManagerTermBase):
    """Compute one named reward component from the task-owned reward contract."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._name = str(cfg.params["name"])
        self._task_cfg = _task_cfg(env)
        self._feet_force_names = tuple(self._task_cfg.sensor.feet_force)
        self._feet_pos_names = tuple(self._task_cfg.sensor.feet_pos)
        sensor_names = [
            self._task_cfg.sensor.local_linvel,
            self._task_cfg.sensor.gyro,
            _UPVECTOR_SENSOR,
            *self._feet_force_names,
            *self._feet_pos_names,
            *_ARM_TOUCH_SENSORS,
        ]
        self._views = env.scene.bind_sensor_data(tuple(sensor_names))
        # Slice by the bound view's per-sensor widths. Foot-contact and arm-touch
        # sensors are 1-wide in the MJCF; fixed 3-wide offsets would silently
        # repoint the gait/contact rewards at neighbouring channels.
        slices: dict[str, slice] = {}
        offset = 0
        for name, width in zip(self._views.names, self._views.dimensions, strict=True):
            slices[name] = slice(offset, offset + width)
            offset += width
        self._slices = slices
        for name in self._feet_pos_names:
            width = self._slices[name].stop - self._slices[name].start
            if width != 3:
                raise ValueError(
                    f"Go2Arm reward expects a 3-wide foot position sensor '{name}', "
                    f"got width {width}"
                )
        self._ee_view = env.scene.bind_sensor_data((self._task_cfg.sensor.ee_local_pos,))
        self._entity: Entity = env.scene["robot"]
        self._torque_view = None
        # Bind actuator-force sensors only when the term actually contributes:
        # motrix does not implement jointactuatorfrc sensors at all, and
        # zero-weight terms never evaluate, so binding them would fail env
        # construction on motrix for terms that are disabled anyway.
        if self._name in {"torques", "energy"} and float(cfg.weight) != 0.0:
            self._torque_view = env.scene.bind_sensor_data(
                tuple(f"{joint_name}_torque" for joint_name in self._entity.joint_names)
            )
            if self._torque_view.dimensions != (1,) * len(self._entity.joint_names):
                raise ValueError("Unexpected Go2Arm actuator-force sensor dimensions")
        self._previous_joint_vel = np.array(self._entity.data.joint_vel, copy=True)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        """Restart finite-difference acceleration after the selected envs reset."""
        current = np.asarray(self._entity.data.joint_vel)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids] = current[env_ids]

    def _read_joint_torques(self) -> np.ndarray:
        if self._torque_view is None:
            raise RuntimeError("Go2Arm torque sensors were not initialized")
        return np.asarray(self._torque_view.read())

    def __call__(self, env: "ManagerBasedRlEnv", name: str | None = None) -> np.ndarray:
        if name is not None and name != self._name:
            raise ValueError("Go2Arm reward term was bound to another reward name")
        cfg = self._task_cfg.reward_parameters
        command = _command(env)
        sensors = np.asarray(self._views.read(), dtype=get_global_dtype())
        linvel = sensors[:, self._slices[self._task_cfg.sensor.local_linvel]]
        gyro = sensors[:, self._slices[self._task_cfg.sensor.gyro]]
        gravity = sensors[:, self._slices[_UPVECTOR_SENSOR]]
        feet_force = np.stack(
            [sensors[:, self._slices[name]] for name in self._feet_force_names], axis=1
        )
        feet_pos = np.stack(
            [sensors[:, self._slices[name]] for name in self._feet_pos_names], axis=1
        )
        contacts = np.stack(
            [sensors[:, self._slices[name]] for name in _ARM_TOUCH_SENSORS], axis=1
        )
        ee_pos = np.asarray(self._ee_view.read(), dtype=get_global_dtype())
        joint_pos = self._entity.data.joint_pos
        joint_vel = self._entity.data.joint_vel
        base_height = self._entity.data.root_link_pos_w[:, 2]
        cmd = command.vel_command_b
        defaults = self._entity.data.default_joint_pos
        if name == "tracking_lin_vel":
            value = np.exp(
                -np.sum(np.square(cmd[:, :2] - linvel[:, :2]), axis=1) / cfg.tracking_sigma
            )
            command.record_tracking(value)
        elif name == "tracking_ang_vel":
            value = np.exp(-np.square(cmd[:, 2] - gyro[:, 2]) / cfg.tracking_sigma)
        elif name == "lin_vel_z":
            value = np.square(linvel[:, 2])
        elif name == "ang_vel_xy":
            value = np.sum(np.square(gyro[:, :2]), axis=1)
        elif name == "roll":
            value = np.square(gravity[:, 0])
        elif name == "base_height":
            value = np.square(base_height - cfg.base_height_target)
        elif name == "similar_to_default":
            value = np.sum(np.abs(joint_pos - defaults), axis=1)
        elif name == "leg_pose":
            weights = np.array([1.0, 1.0, 0.1] * 4 + [0.0] * 6, dtype=get_global_dtype())
            value = np.sum(weights * np.square(joint_pos - defaults), axis=1)
        elif name == "dof_pos_limits":
            if not cfg.leg_dof_upper_limits or not cfg.leg_dof_lower_limits:
                value = np.zeros(env.num_envs, dtype=get_global_dtype())
            else:
                upper = np.asarray(cfg.leg_dof_upper_limits, dtype=get_global_dtype())
                lower = np.asarray(cfg.leg_dof_lower_limits, dtype=get_global_dtype())
                margin = cfg.dof_pos_limit_margin
                leg_pos = joint_pos[:, :12]
                over = np.square(np.maximum(leg_pos - upper + margin, 0.0))
                under = np.square(np.maximum(lower + margin - leg_pos, 0.0))
                value = np.sum(over + under, axis=1)
        elif name == "action_rate":
            value = np.sum(
                np.square(env.action_manager.action - env.action_manager.prev_action), axis=1
            )
        elif name == "torques":
            value = np.sum(np.abs(self._read_joint_torques()), axis=1)
        elif name == "energy":
            torques = self._read_joint_torques()
            value = np.sum(np.abs(joint_vel) * np.abs(torques), axis=1)
        elif name == "dof_vel":
            value = np.sum(np.square(joint_vel), axis=1)
        elif name == "dof_acc":
            acceleration = (joint_vel - self._previous_joint_vel) / env.step_dt
            self._previous_joint_vel[:] = joint_vel
            value = np.sum(np.square(acceleration), axis=1)
        elif name == "stand_still":
            still = (~command._command_is_moving(cmd)).astype(get_global_dtype())
            value = still * np.sum(np.abs(joint_pos[:, :12] - defaults[:, :12]), axis=1)
        elif name == "swing_feet_z":
            swing = command.feet_phase >= 0.6
            error = np.square(feet_pos[:, :, 2] - cfg.target_foot_height)
            value = np.sum(np.exp(-error / 0.01) * swing, axis=1) / len(self._feet_pos_names)
        elif name == "foot_drag":
            contact = feet_force[:, :, -1] < 0.5
            height_error = np.clip(cfg.target_foot_height / 2.0 - feet_pos[:, :, 2], 0.0, None)
            value = np.sum(np.square(height_error) * contact, axis=1)
        elif name == "contact":
            contact = feet_force[:, :, -1] > 0.1
            stance = (command.feet_phase < 0.6) | (command.gait_frequency < 1.0e-8)
            value = np.sum(contact == stance, axis=1) / len(self._feet_force_names)
        elif name == "object_distance":
            value = np.exp(
                -np.sum(np.square(ee_pos - command.curr_ee_goal_cart), axis=1) / cfg.object_sigma
            )
        elif name == "object_distance_l2":
            value = np.sum(np.square(ee_pos - command.curr_ee_goal_cart), axis=1)
        elif name == "arm_collision":
            value = np.sum(contacts[:, :, 0], axis=1)
        else:
            raise KeyError(f"Unknown Go2Arm reward term '{name or self._name}'")
        return np.asarray(value, dtype=get_global_dtype())


class Go2ArmFallen(ManagerTermBase):
    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._view = env.scene.bind_sensor_data(("upvector",))

    def __call__(self, env: "ManagerBasedRlEnv") -> np.ndarray:
        del env
        return np.asarray(self._view.read()[:, 2], dtype=get_global_dtype()) <= 0.5


class Go2ArmManagerBasedRlEnv(ManagerBasedRlEnv):
    """MBA runtime with a visual playback oracle for MuJoCo.

    MuJoCo compiles the training model with visual meshes discarded.  UniSim's
    per-env playback path therefore receives that physics-only model unless a
    task supplies its own oracle.  When this task has no fixed model variants,
    the original scene MJCF has exactly the same dynamics topology and is the
    correct visual oracle.
    """

    def get_playback_model(self, env_index: int | None = None) -> Any:
        if getattr(self._cfg, "fixed_model_variants", None) is None:
            visual_model_file = self.get_scene_visual_model_file()
            if visual_model_file:
                return Path(visual_model_file)
        return super().get_playback_model(env_index)


class Go2ArmPush(ManagerTermBase):
    """Fixed world-frame force push dispatched through the Entity wrench contract."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        cfg_task = _task_cfg(env)
        body_name = cfg_task.domain_rand.get("push_body_name", cfg_task.asset.base_name)
        self._entity: Entity = env.scene["robot"]
        body_ids, _ = self._entity.find_bodies((body_name,))
        _, self._body_ids = self._entity.bind_body_wrench(
            body_ids,
            torque=False,
            term_name="go2_arm_push",
        )
        self._force = np.asarray(cfg_task.domain_rand["max_force"], dtype=np.float64)[None, None, :]

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: np.ndarray | None,
        force: np.ndarray | tuple[float, float, float],
    ) -> None:
        ids = (
            np.arange(env.num_envs, dtype=np.int32)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.int32)
        )
        samples = np.asarray(force, dtype=np.float64)
        if samples.shape != (3,):
            raise ValueError(f"go2_arm_push force must have shape (3,), got {samples.shape}")
        forces = np.broadcast_to(
            samples[None, None, :],
            (len(ids), self._body_ids.size, 3),
        ).copy()
        self._entity.apply_body_wrench_to_sim(
            forces,
            None,
            self._body_ids,
            env_ids=ids,
            term_name="go2_arm_push",
        )
