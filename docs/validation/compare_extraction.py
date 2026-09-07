"""One-off extraction evidence: compare an archived source tree with this package."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from legged_manipulation_unilab.assets import resolve_scene
from legged_manipulation_unilab.cli import CONF_ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-source", type=Path)
parser.add_argument("--algo", choices=["ppo", "ppo_him"], required=True)
parser.add_argument("--sim", choices=["mujoco", "motrix"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare-to", type=Path)
args = parser.parse_args()
scene = resolve_scene()
if args.baseline_source:
    sys.path.insert(0, str(args.baseline_source / "src"))
    module = importlib.import_module("unilab.tasks.locomotion.go2_arm.manip_loco")
else:
    module = importlib.import_module("legged_manipulation_unilab.tasks.go2_arm.manip_loco")

from unilab.base.config_materialization import apply_cfg_overrides

with initialize_config_dir(config_dir=str(CONF_ROOT / args.algo), version_base="1.3"):
    owner = compose(config_name="config", overrides=[f"task=go2_arm_manip_loco/{args.sim}"])
cfg = module.Go2ArmManipLocoCfg()
apply_cfg_overrides(cfg, OmegaConf.to_container(owner.env, resolve=True))
cfg.reward_config = module.RewardConfig(**OmegaConf.to_container(owner.reward, resolve=True))
cfg.model_file = scene
np.random.seed(2026)
env = module.Go2ArmManipLocoEnv(cfg, num_envs=2, backend_type=args.sim)
results = {}
try:
    obs, _ = env.reset(np.arange(2, dtype=np.int32))
    for key, value in obs.items():
        results["reset_" + key] = value.copy()
    rng = np.random.default_rng(4)
    for step in range(12):
        state = env.step(rng.uniform(-0.2, 0.2, (2, 18)).astype(np.float32))
        for key, value in state.obs.items():
            results[f"{step}_{key}"] = value.copy()
        results[f"{step}_reward"] = state.reward.copy()
        results[f"{step}_terminated"] = state.terminated.copy()
        results[f"{step}_truncated"] = state.truncated.copy()
        results[f"{step}_goal"] = env.curr_ee_goal_cart.copy()
finally:
    env.close()
np.savez(args.output, **results)
if args.compare_to:
    with np.load(args.compare_to) as reference:
        assert set(reference.files) == set(results)
        for key, value in results.items():
            np.testing.assert_array_equal(reference[key], value, err_msg=key)
    print(args.algo, args.sim, "EXACT MATCH", len(results), "arrays")
