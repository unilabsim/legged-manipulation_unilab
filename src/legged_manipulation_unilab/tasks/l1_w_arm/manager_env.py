"""L1-W reward terms, independent of Go2 reward evaluation."""

from __future__ import annotations

import numpy as np
from unilab.base.entity import Entity
from unilab.dtype_config import get_global_dtype
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
from unilab.managers.manager_base import ManagerTermBase

_UPVECTOR_SENSOR = "upvector"
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


def _command(env: ManagerBasedRlEnv):
    return env.command_manager.get_term("task")


class L1WArmReward(ManagerTermBase):
    """Compute L1-W reward terms independently of Go2 reward evaluation."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        super().__init__(env)
        self._name = str(cfg.params["name"])
        self._task_cfg = env._cfg
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
                    f"L1WArm reward expects a 3-wide foot position sensor '{name}', "
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
                raise ValueError("Unexpected L1WArm actuator-force sensor dimensions")
        self._previous_joint_vel = np.array(self._entity.data.joint_vel, copy=True)

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        """Restart finite-difference acceleration after the selected envs reset."""
        current = np.asarray(self._entity.data.joint_vel)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids] = current[env_ids]

    def _read_joint_torques(self) -> np.ndarray:
        if self._torque_view is None:
            raise RuntimeError("L1WArm torque sensors were not initialized")
        return np.asarray(self._torque_view.read())

    def __call__(self, env: "ManagerBasedRlEnv", name: str | None = None) -> np.ndarray:
        if name is not None and name != self._name:
            raise ValueError("L1WArm reward term was bound to another reward name")
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
        contacts = np.stack([sensors[:, self._slices[name]] for name in _ARM_TOUCH_SENSORS], axis=1)
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
        elif name == "hip_pos":
            # The first 12 action rows are four ABAD/HIP/KNEE triplets.
            # Constrain the lateral ABAD joints.
            hip_error = (joint_pos[:, :12] - defaults[:, :12])[:, ::3]
            value = np.sum(np.square(hip_error), axis=1)
        elif name == "leg_pose":
            weights = np.array([1.0, 1.0, 0.1] * 4 + [0.0] * 10, dtype=get_global_dtype())
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
        elif name == "wheel_stand_still":
            # Keep the wheels free for low-speed forward/back commands too.
            still = np.all(np.abs(cmd[:, :3]) <= 0.01, axis=1).astype(get_global_dtype())
            value = still * np.sum(np.square(joint_vel[:, 12:16]), axis=1)
        elif name == "swing_feet_z":
            swing = command._gait_is_active(cmd)[:, None] & (command.feet_phase >= 0.6)
            error = np.square(feet_pos[:, :, 2] - cfg.target_foot_height)
            value = np.sum(np.exp(-error / 0.01) * swing, axis=1) / len(self._feet_pos_names)
        elif name == "mirrorup":
            # N-P3O foot_mirror_up: raw ABAD/HIP/KNEE angle differences.
            # L1W actuator order is FL, FR, RR, RL.
            rear_right, rear_left = slice(6, 9), slice(9, 12)
            error = 0.5 * (
                np.sum(np.square(joint_pos[:, :3] - joint_pos[:, rear_right]), axis=1)
                + np.sum(np.square(joint_pos[:, 3:6] - joint_pos[:, rear_left]), axis=1)
            )
            # World-frame body-up Z equals -projected_gravity_z in N-P3O.
            value = error * np.clip(gravity[:, 2], 0.0, 1.0)
        elif name == "foot_drag":
            contact = feet_force[:, :, -1] < 0.5
            height_error = np.clip(cfg.target_foot_height / 2.0 - feet_pos[:, :, 2], 0.0, None)
            value = np.sum(np.square(height_error) * contact, axis=1)
        elif name == "contact":
            contact = feet_force[:, :, -1] > 0.1
            stance = (
                ~command._gait_is_active(cmd)[:, None]
                | (command.feet_phase < 0.6)
                | (command.gait_frequency < 1.0e-8)
            )
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
            raise KeyError(f"Unknown L1WArm reward term '{name or self._name}'")
        return np.asarray(value, dtype=get_global_dtype())
