"""Go2 arm manipulation-locomotion task declared on the Manager-Based runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from unilab.base import registry
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
from unilab.envs.mdp import (
    dof_armature,
    geom_friction,
    pd_gains,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    reset_root_state_uniform,
    reset_scene_to_default,
    time_out,
)
from unilab.managers.event_manager import EventTermCfg
from unilab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from unilab.managers.reward_manager import RewardTermCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.managers.termination_manager import TerminationTermCfg

from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH, resolve_scene
from legged_manipulation_unilab.tasks.go2_arm.base import (
    Go2ArmBaseCfg,
    Go2ArmSensor,
    build_go2_arm_position_gains,
)
from legged_manipulation_unilab.tasks.go2_arm.manager_env import (
    Go2ArmFallen,
    Go2ArmIKActionCfg,
    Go2ArmManagerBasedRlEnv,
    Go2ArmManipLocoCommandCfg,
    Go2ArmObservation,
    Go2ArmPush,
    Go2ArmReward,
)


def _default_go2_arm_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_flat.xml")


# The legacy task keyed every default-state consumer (reset pose, PD targets,
# obs joint diff, pose rewards) to the XML "home" keyframe. The Manager-Based
# runtime only reads a keyframe when the scene declares one, so every Go2Arm
# SceneCfg must carry it or defaults fall back to qpos0 (zeros).
_GO2_ARM_DEFAULT_KEYFRAME = "home"


def _go2_arm_scene(model_file: str) -> SceneCfg:
    return SceneCfg(model_file=model_file, default_keyframe_name=_GO2_ARM_DEFAULT_KEYFRAME)


def _default_go2_arm_scene() -> SceneCfg:
    return _go2_arm_scene(_default_go2_arm_model_file())


_ROBOT_BODY_NAMES = (
    "base",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "arm_base_on_go2",
    "arm_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
)

# ``arm_base_on_go2`` is a zero-mass, zero-inertia adapter body. MotrixSim
# requires a positive-definite mass matrix when a mass override is present, so
# assigning the event's positive minimum mass to that body makes the first step
# produce NaNs. Keep it in scene bindings but exclude it from mass scaling.
_MASS_RANDOMIZATION_BODY_NAMES = tuple(
    name for name in _ROBOT_BODY_NAMES if name != "arm_base_on_go2"
)


@dataclass
class InitState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.42])


@dataclass
class EEGoalConfig:
    """End-effector goal sampling and collision contract in spherical coordinates."""

    sphere_l_range: list[float] = field(default_factory=lambda: [0.3, 0.6])
    sphere_phi_range: list[float] = field(default_factory=lambda: [-1.2566, 1.0472])
    sphere_theta_range: list[float] = field(default_factory=lambda: [-2.3562, 2.3562])
    traj_time_range: list[float] = field(default_factory=lambda: [1.0, 3.0])
    hold_time_range: list[float] = field(default_factory=lambda: [0.5, 2.0])
    collision_upper_limits: list[float] = field(default_factory=lambda: [0.3, 0.15, 0.05 - 0.165])
    collision_lower_limits: list[float] = field(
        default_factory=lambda: [-0.2, -0.15, -0.35 - 0.165]
    )
    underground_limit: float = -0.57
    num_collision_check_samples: int = 10
    num_resample_attempts: int = 10
    default_orn_roll: float = float(np.pi / 2.0)
    arm_induced_pitch: float = 0.78
    delta_orn_r: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_p: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_y: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    init_ee_cart: list[float] = field(default_factory=lambda: [0.30, 0.0, 0.25])


@dataclass
class CommandsConfig:
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]
    )
    resample_time_s: float | None = None
    zero_command_prob: float = 0.2
    # Gait clock cadence in Hz; drives feet_phase advancement while moving.
    gait_frequency: float = 2.0


@dataclass
class CurriculumConfig:
    enable: bool = False
    threshold: float = 0.5
    step_size: float = 0.05
    max_vel_limit: list[float] = field(default_factory=lambda: [1.5, 1.0, 1.2])


@dataclass
class RewardParameters:
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    target_foot_height: float = 0.08
    object_sigma: float = 0.1
    leg_dof_upper_limits: list[float] | None = None
    leg_dof_lower_limits: list[float] | None = None
    dof_pos_limit_margin: float = 0.05


@dataclass
class HistoryConfig:
    num_actor_history: int = 5
    num_critic_history: int = 1


@dataclass
class ArmStageConfig:
    freeze_arm_joints: bool = False
    disable_ee_goal_trajectory: bool = False
    fixed_ee_goal_cart: list[float] = field(default_factory=lambda: [0.15, 0.0, 0.25])


@registry.envcfg("Go2ArmManipLoco")
@dataclass
class Go2ArmManipLocoCfg(Go2ArmBaseCfg):  # pyright: ignore[reportIncompatibleVariableOverride]
    scene: SceneCfg = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=_default_go2_arm_scene
    )
    model_file: str = field(default_factory=_default_go2_arm_model_file)
    max_episode_seconds: float = 20.0  # pyright: ignore[reportIncompatibleVariableOverride]
    init_state: InitState = field(default_factory=InitState)
    command_config: CommandsConfig = field(default_factory=CommandsConfig)
    reward_parameters: RewardParameters = field(default_factory=RewardParameters)
    sensor: Go2ArmSensor = field(default_factory=Go2ArmSensor)
    domain_rand: dict[str, Any] = field(default_factory=dict)
    goal_ee: EEGoalConfig = field(default_factory=EEGoalConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    arm_stage: ArmStageConfig = field(default_factory=ArmStageConfig)
    curriculum_config: CurriculumConfig = field(default_factory=CurriculumConfig)

    def __post_init__(self) -> None:
        # Keep the old model_file owner override working while SceneCfg is the
        # single MBA backend declaration.
        scene = self.scene
        default_model_file = _default_go2_arm_model_file()
        if self.model_file != default_model_file and scene.model_file == default_model_file:
            scene = _go2_arm_scene(self.model_file)
        self.scene = scene  # pyright: ignore[reportIncompatibleVariableOverride]
        self._configure_managers()

    def _configure_managers(self) -> None:
        self.policy_observation_group = "actor"
        self.critic_observation_group = "critic"
        self.scale_rewards_by_dt = True
        self.auto_reset = True
        self.actions = {
            "joint_pos": Go2ArmIKActionCfg(entity_name="robot"),
        }
        actor_history = self.history.num_actor_history
        critic_history = self.history.num_critic_history
        self.observations = {
            "actor": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=Go2ArmObservation,
                        params={"actor": True},
                        history_length=actor_history,
                        flatten_history_dim=True,
                    )
                },
                enable_corruption=False,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=Go2ArmObservation,
                        params={"actor": False},
                        history_length=critic_history,
                        flatten_history_dim=True,
                    )
                },
                enable_corruption=False,
                nan_policy="error",
            ),
        }
        resample_time = (
            self.command_config.resample_time_s
            if self.command_config.resample_time_s is not None
            else 1.0e9
        )
        self.commands = {
            "task": Go2ArmManipLocoCommandCfg(resampling_time_range=(resample_time, resample_time))
        }
        self.terminations = {
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "fallen": TerminationTermCfg(func=Go2ArmFallen, time_out=False),
        }
        self.events = self._build_events()  # pyright: ignore[reportAttributeAccessIssue]
        self.rewards: dict[str, RewardTermCfg] = {}

    def _build_events(self) -> dict[str, EventTermCfg]:
        dr = self.domain_rand
        robot = SceneEntityCfg("robot")
        events: dict[str, EventTermCfg] = {
            "reset_scene_to_default": EventTermCfg(
                func=reset_scene_to_default,
                mode="reset",
            ),
            "reset_root_state_uniform": EventTermCfg(
                func=reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": [-0.5, 0.5],
                        "y": [-0.5, 0.5],
                        "z": [0.0, 0.0],
                        "roll": [0.0, 0.0],
                        "pitch": [0.0, 0.0],
                        "yaw": [-np.pi, np.pi],
                    },
                    "velocity_range": {
                        "x": [-0.5, 0.5],
                        "y": [-0.5, 0.5],
                        "z": [-0.5, 0.5],
                        "roll": [-0.5, 0.5],
                        "pitch": [-0.5, 0.5],
                        "yaw": [-0.5, 0.5],
                    },
                    "asset_cfg": robot,
                },
            ),
        }
        if dr.get("randomize_body_mass", False):
            events["body_mass"] = EventTermCfg(
                func=randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=_MASS_RANDOMIZATION_BODY_NAMES),
                    "mass_distribution_params": tuple(dr["body_mass_multiplier_range"]),
                    "operation": "scale",
                    "recompute_inertia": False,
                },
            )
        if dr.get("randomize_base_mass", False):
            events["base_mass"] = EventTermCfg(
                func=randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(self.asset.base_name,)),
                    "mass_distribution_params": tuple(dr["added_mass_range"]),
                    "operation": "add",
                    "recompute_inertia": False,
                },
            )
        if dr.get("random_com", False):
            events["base_com"] = EventTermCfg(
                func=randomize_rigid_body_com,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(self.asset.base_name,)),
                    "com_range": {
                        "x": tuple(dr.get("com_offset_x", [-0.03, 0.03])),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                    },
                },
            )
        if dr.get("randomize_ground_friction", False):
            events["ground_friction"] = EventTermCfg(
                func=geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=(self.asset.ground,)),
                    "ranges": tuple(dr["ground_friction_multiplier_range"]),
                    "operation": "scale",
                    "axes": [0],
                },
            )
        if dr.get("randomize_dof_armature", False):
            events["joint_armature"] = EventTermCfg(
                func=dof_armature,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "ranges": tuple(dr["dof_armature_multiplier_range"]),
                    "operation": "scale",
                    "axes": [0],
                },
            )
        if dr.get("randomize_kp", False) or dr.get("randomize_kd", False):
            events["pd_gains"] = EventTermCfg(
                func=pd_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "kp_range": tuple(dr.get("kp_multiplier_range", [1.0, 1.0])),
                    "kd_range": tuple(dr.get("kd_multiplier_range", [1.0, 1.0])),
                    "operation": "scale",
                },
            )
        if dr.get("push_robots", False):
            interval_s = float(dr["push_interval"]) * self.ctrl_dt
            events["push_robot"] = EventTermCfg(
                func=Go2ArmPush,
                mode="interval",
                interval_range_s=(interval_s, interval_s),
                is_global_time=True,
                params={
                    "force": tuple(dr["max_force"]),
                },
            )
        return events

    def set_reward_terms(self, terms: dict[str, float]) -> None:
        """Install manager reward terms after typed config materialization."""
        self.rewards = {  # pyright: ignore[reportIncompatibleVariableOverride]
            name: RewardTermCfg(
                func=Go2ArmReward,
                weight=float(weight),
                params={"name": name},
            )
            for name, weight in terms.items()
        }

    def prepare_runtime(self) -> None:
        """Rebuild event declarations after Hydra overrides are applied."""
        self.events = self._build_events()  # pyright: ignore[reportAttributeAccessIssue]


def make_go2_arm_manip_loco_env(
    cfg: Go2ArmManipLocoCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Create the task on the generic MBA runtime with its backend-specific knobs."""
    if not isinstance(cfg, Go2ArmManipLocoCfg):
        raise TypeError(f"Expected Go2ArmManipLocoCfg, got {type(cfg).__name__}")
    if not cfg.rewards:
        raise ValueError("Go2ArmManipLocoCfg.rewards must contain manager reward terms")
    if backend_type not in {"mujoco", "motrix", "drake"}:
        raise ValueError(f"Unsupported Go2Arm backend: {backend_type}")
    cfg.prepare_runtime()
    cfg.validate()
    cfg.scene.model_file = resolve_scene(cfg.scene.model_file)
    # Every Go2Arm scene model declares the home keyframe and the task keys
    # its default state to it. The play-profile adapter rebuilds the scene as
    # a bare SceneCfg around the materialized model and drops the field, which
    # silently reverts eval/play to the all-zero qpos0 defaults; re-assert the
    # invariant after overrides have been applied.
    if cfg.scene.default_keyframe_name is None:
        cfg.scene.default_keyframe_name = _GO2_ARM_DEFAULT_KEYFRAME
    apply_env_cpu_runtime(cfg.cpu_ids)
    backend_kwargs: dict[str, Any] = {
        "base_name": cfg.asset.base_name,
        "push_body_name": cfg.domain_rand.get("push_body_name", cfg.asset.base_name),
        "position_actuator_gains": build_go2_arm_position_gains(cfg.control_config),
        "iterations": cfg.iterations,
        **env_backend_kwargs(cfg),
        "motrix_max_iterations": cfg.motrix_max_iterations,
    }
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        **backend_kwargs,
    )
    cfg.scene.entities = {
        "robot": EntityCfg(
            root_body_name=cfg.asset.base_name,
            joint_names=backend.get_actuator_joint_names(),
            body_names=_ROBOT_BODY_NAMES,
            geom_names=(cfg.asset.ground,),
            actuator_names=backend.get_actuator_names(),
        )
    }
    try:
        return Go2ArmManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


for _backend_type in ("mujoco", "motrix", "drake"):
    registry.register_env("Go2ArmManipLoco", make_go2_arm_manip_loco_env, _backend_type)
