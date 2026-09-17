"""L1-W + Airbot manipulation-locomotion task on the Manager-Based runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unilab.base import registry
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv

from legged_manipulation_unilab.assets import ASSETS_ROOT_PATH, resolve_scene
from legged_manipulation_unilab.tasks.go2_arm.base import (
    ActionLayout,
    Asset,
    ControlConfig,
    Go2ArmSensor,
    build_l1w_arm_position_gains,
)
from legged_manipulation_unilab.tasks.go2_arm.manager_env import Go2ArmManagerBasedRlEnv
from legged_manipulation_unilab.tasks.go2_arm.manip_loco import (
    Go2ArmManipLocoCfg,
    InitState,
    RewardParameters,
)

_DEFAULT_KEYFRAME = "home"
_L1W_ACTION_DIM = 22


def _default_l1w_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "l1_w_arm" / "scene_flat.xml")


def _l1w_scene(model_file: str) -> SceneCfg:
    return SceneCfg(model_file=model_file, default_keyframe_name=_DEFAULT_KEYFRAME)


def _default_l1w_scene() -> SceneCfg:
    return _l1w_scene(_default_l1w_model_file())


def _l1w_control() -> ControlConfig:
    return ControlConfig(leg_kp=20.0, leg_kd=0.7)


def _l1w_layout() -> ActionLayout:
    return ActionLayout(
        dim=_L1W_ACTION_DIM,
        leg_slice=(0, 12),
        wheel_slice=(12, 16),
        arm_slice=(16, 22),
    )


_ROBOT_BODY_NAMES = (
    "BASE_LINK",
    "FL_ABAD_LINK",
    "FL_HIP_LINK",
    "FL_KNEE_LINK",
    "FR_ABAD_LINK",
    "FR_HIP_LINK",
    "FR_KNEE_LINK",
    "RL_ABAD_LINK",
    "RL_HIP_LINK",
    "RL_KNEE_LINK",
    "RR_ABAD_LINK",
    "RR_HIP_LINK",
    "RR_KNEE_LINK",
    "arm_base_on_l1",
    "arm_base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
)


@dataclass
class L1WArmAsset(Asset):
    base_name: str = "BASE_LINK"


@dataclass
class L1WArmSensor(Go2ArmSensor):
    # FL, FR, RL, RR order so the inherited trot clock (0+3 / 1+2) stays diagonal.
    feet_force: list[str] = field(
        default_factory=lambda: [
            "FBL_foot_contact",
            "FAR_foot_contact",
            "RBL_foot_contact",
            "RAR_foot_contact",
        ]
    )
    feet_pos: list[str] = field(
        default_factory=lambda: ["FBL_pos", "FAR_pos", "RBL_pos", "RAR_pos"]
    )


@registry.envcfg("L1WArmManipLoco")
@dataclass
class L1WArmManipLocoCfg(Go2ArmManipLocoCfg):  # pyright: ignore[reportIncompatibleVariableOverride]
    scene: SceneCfg = field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=_default_l1w_scene
    )
    model_file: str = field(default_factory=_default_l1w_model_file)
    init_state: InitState = field(default_factory=lambda: InitState(pos=[0.0, 0.0, 0.4]))
    reward_parameters: RewardParameters = field(
        default_factory=lambda: RewardParameters(base_height_target=0.38, target_foot_height=0.05)
    )
    control_config: ControlConfig = field(default_factory=_l1w_control)
    asset: Asset = field(default_factory=L1WArmAsset)
    sensor: Go2ArmSensor = field(default_factory=L1WArmSensor)
    action_layout: ActionLayout = field(default_factory=_l1w_layout)
    robot_body_names: tuple[str, ...] = _ROBOT_BODY_NAMES
    mass_randomization_exclude: tuple[str, ...] = ("arm_base_on_l1",)

    def __post_init__(self) -> None:
        scene = self.scene
        default_model_file = _default_l1w_model_file()
        if self.model_file != default_model_file and scene.model_file == default_model_file:
            scene = _l1w_scene(self.model_file)
        self.scene = scene  # pyright: ignore[reportIncompatibleVariableOverride]
        self._configure_managers()


def make_l1_w_arm_manip_loco_env(
    cfg: L1WArmManipLocoCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ManagerBasedRlEnv:
    """Create the L1-W+Airbot task on the generic MBA runtime."""
    if not isinstance(cfg, L1WArmManipLocoCfg):
        raise TypeError(f"Expected L1WArmManipLocoCfg, got {type(cfg).__name__}")
    if not cfg.rewards:
        raise ValueError("L1WArmManipLocoCfg.rewards must contain manager reward terms")
    if backend_type not in {"mujoco", "motrix", "drake"}:
        raise ValueError(f"Unsupported L1-W+Airbot backend: {backend_type}")
    cfg.prepare_runtime()
    cfg.validate()
    cfg.scene.model_file = resolve_scene(cfg.scene.model_file)
    if cfg.scene.default_keyframe_name is None:
        cfg.scene.default_keyframe_name = _DEFAULT_KEYFRAME
    apply_env_cpu_runtime(cfg.cpu_ids)
    backend_kwargs: dict[str, Any] = {
        "base_name": cfg.asset.base_name,
        "push_body_name": cfg.domain_rand.get("push_body_name", cfg.asset.base_name),
        "position_actuator_gains": build_l1w_arm_position_gains(cfg.control_config),
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
    registry.register_env("L1WArmManipLoco", make_l1_w_arm_manip_loco_env, _backend_type)
