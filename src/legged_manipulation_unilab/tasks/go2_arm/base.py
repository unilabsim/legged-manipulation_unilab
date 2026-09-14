"""Go2 Arm configuration and task-owned IK helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.utils.rotation import np_matrix_from_quat
from unisim.backend.base import SimBackend

from legged_manipulation_unilab.tasks.geometry import np_quat_orientation_error_local


@dataclass
class NoiseConfig:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1
    scale_ee_pos: float = 0.02


@dataclass
class ControlConfig:
    action_scale: float = 0.25
    simulate_action_latency: bool = False
    Kp: float = 35.0
    Kd: float = 0.5
    leg_kp: float | list[float] | None = 60.0
    leg_kd: float | list[float] | None = 2.0
    arm_kp: float | list[float] | None = field(
        default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0]
    )
    arm_kd: float | list[float] | None = field(
        default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
    )
    arm_action_scale: float = 0.0


@dataclass
class IKConfig:
    damping: float = 0.05
    gain: float = 1.0
    dq_clip: float = 0.2
    use_orientation: bool = False
    orientation_mode: str = "target"


@dataclass
class Asset:
    base_name: str = "base"
    foot_name: str = "foot"
    ground: str = "floor"
    ee_site_name: str = "endpoint"
    ee_body_name: str = "link6"
    arm_joint_names: tuple[str, ...] = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class Go2ArmSensor(Sensor):
    feet_force: list[str] = field(
        default_factory=lambda: [
            "FL_foot_contact",
            "FR_foot_contact",
            "RL_foot_contact",
            "RR_foot_contact",
        ]
    )
    feet_pos: list[str] = field(default_factory=lambda: ["FL_pos", "FR_pos", "RL_pos", "RR_pos"])
    ee_local_pos: str = "endpoint_pos"
    ee_local_quat: str = "endpoint_quat"
    ee_local_vel: str = "endpoint_vel"
    ee_relative_pos: str = "endpoint_relative_pos"
    ee_relative_quat: str = "endpoint_relative_quat"
    arm_ref_world_quat: str = "armbasepoint_world_quat"


@dataclass
class Go2ArmBaseCfg(ManagerBasedRlEnvCfg):
    control_config: ControlConfig = field(default_factory=ControlConfig)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    asset: Asset = field(default_factory=Asset)
    sensor: Go2ArmSensor = field(default_factory=Go2ArmSensor)
    iterations: int | None = None
    post_step_forward_sensor: bool = False
    adaptive_chunk_size: bool = True
    chunk_size: int | None = None
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02


def _expand_gain(
    name: str, value: float | list[float] | None, fallback: float, size: int
) -> np.ndarray:
    raw_value = fallback if value is None else value
    gain = np.asarray(raw_value, dtype=np.float64)
    if gain.ndim == 0:
        return np.full((size,), float(gain), dtype=np.float64)
    if gain.shape != (size,):
        raise ValueError(f"{name} must be a scalar or have shape ({size},), got {gain.shape}")
    return gain


def build_go2_arm_position_gains(cfg: ControlConfig) -> dict[str, np.ndarray]:
    leg_kp = _expand_gain("control_config.leg_kp", cfg.leg_kp, cfg.Kp, 12)
    leg_kd = _expand_gain("control_config.leg_kd", cfg.leg_kd, cfg.Kd, 12)
    arm_kp = _expand_gain("control_config.arm_kp", cfg.arm_kp, cfg.Kp, 6)
    arm_kd = _expand_gain("control_config.arm_kd", cfg.arm_kd, cfg.Kd, 6)
    return {
        "kp": np.concatenate([leg_kp, arm_kp]),
        "kd": np.concatenate([leg_kd, arm_kd]),
    }


def compute_arm_ik_delta(
    backend: SimBackend,
    cfg: IKConfig,
    *,
    ee_site_name: str,
    arm_joint_names: tuple[str, ...],
    arm_ref_world_quat_sensor: str,
    goal_local_pos: np.ndarray,
    curr_local_pos: np.ndarray,
    goal_local_quat: np.ndarray | None = None,
    curr_local_quat: np.ndarray | None = None,
) -> np.ndarray:
    """Damped least-squares IK delta used by the Manager-Based action term."""
    site_id = int(backend.get_site_ids([ee_site_name])[0])
    arm_dof_ids = backend.get_joint_dof_indices(arm_joint_names)
    pos_err = np.asarray(goal_local_pos - curr_local_pos)
    jacp_w, jacr_w = backend.get_site_jacobian_w(site_id, arm_dof_ids)
    ref_rot_w = np_matrix_from_quat(backend.get_sensor_data(arm_ref_world_quat_sensor))
    rot_w_to_b = np.swapaxes(ref_rot_w, 1, 2)
    jacp_b = np.matmul(rot_w_to_b, jacp_w)
    if cfg.use_orientation:
        if cfg.orientation_mode == "target":
            if goal_local_quat is None or curr_local_quat is None:
                raise ValueError("orientation targets are required when use_orientation=true")
            orn_err = np_quat_orientation_error_local(goal_local_quat, curr_local_quat)
        elif cfg.orientation_mode == "zero_error":
            orn_err = np.zeros_like(pos_err)
        else:
            raise ValueError("ik.orientation_mode must be one of {'target', 'zero_error'}")
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
        dq = np.clip(dq, -cfg.dq_clip, cfg.dq_clip)
    return dq


__all__ = [
    "Asset",
    "ControlConfig",
    "Go2ArmBaseCfg",
    "Go2ArmSensor",
    "IKConfig",
    "NoiseConfig",
    "build_go2_arm_position_gains",
    "compute_arm_ik_delta",
]
