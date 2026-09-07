"""Package-owned owner selection; algorithms consume the composed configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONF_ROOT = Path(__file__).resolve().parent / "conf"


def compose_config(algo: str, sim: str, overrides: list[str]) -> DictConfig:
    group = "ppo_him" if algo == "him_ppo" else "ppo"
    owner = f"go2_arm_manip_loco/{sim}"
    if not (CONF_ROOT / group / "task" / f"{owner}.yaml").is_file():
        raise ValueError(f"No owner for algo={algo}, sim={sim}")
    reserved = {"task", "training.task_name", "training.sim_backend", "training.play_only"}
    for override in overrides:
        if override.lstrip("+~").split("=", 1)[0] in reserved:
            raise ValueError("Use CLI flags to select task, backend and train/eval mode")
    with initialize_config_dir(config_dir=str(CONF_ROOT / group), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={owner}", *overrides])
    if cfg.training.task_name != "Go2ArmManipLoco" or cfg.training.sim_backend != sim:
        raise ValueError("Overrides must preserve the selected task owner identity")
    return cfg


def _main(*, play: bool, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Go2 arm locomotion and manipulation")
    parser.add_argument("--algo", choices=["ppo", "him_ppo"], default="ppo")
    parser.add_argument("--sim", choices=["mujoco", "motrix"], default="mujoco")
    parser.add_argument("--task", choices=["go2_arm_manip_loco"], default="go2_arm_manip_loco")
    parser.add_argument(
        "--cfg", action="store_true", help="Print composed config without loading assets"
    )
    parser.add_argument("--export", action="store_true", help="Export checkpoint during eval")
    args, overrides = parser.parse_known_args(argv)
    cfg = compose_config(args.algo, args.sim, overrides)
    cfg.training.play_only = play
    if args.cfg:
        print(OmegaConf.to_yaml(cfg))
        return

    from legged_manipulation_unilab.assets import ensure_assets

    root = ensure_assets(include_floor=args.sim == "motrix")
    asset_overrides: list[str] = []
    if OmegaConf.select(cfg, "play_profile.scene.source_model_file") is not None:
        for key, relative in [
            ("source_model_file", "robots/go2_arm/scene_flat.xml"),
            ("ground_texture_file", "robots/g1/textures/floor.png"),
        ]:
            path = str(root / relative)
            cfg.play_profile.scene[key] = path
            asset_overrides.append(f"play_profile.scene.{key}={path}")
    if args.algo == "him_ppo":
        from legged_manipulation_unilab.training import him

        him.EXPORT_POLICY = args.export
        him.main(cfg)
    else:
        from unilab.scripts import train_rsl_rl

        # The common PPO launcher re-enters its own script under torchrun.
        # Forward our config root and owner to every worker explicitly.
        original_argv = sys.argv
        original_export = train_rsl_rl.EXPORT_POLICY
        try:
            train_rsl_rl.EXPORT_POLICY = args.export
            sys.argv = [
                original_argv[0],
                f"--config-path={CONF_ROOT / 'ppo'}",
                f"task=go2_arm_manip_loco/{args.sim}",
                *overrides,
                f"training.play_only={str(play).lower()}",
                *asset_overrides,
            ]
            train_rsl_rl.main(cfg)
        finally:
            sys.argv = original_argv
            train_rsl_rl.EXPORT_POLICY = original_export


def train_main() -> None:
    _main(play=False)


def eval_main() -> None:
    _main(play=True)
