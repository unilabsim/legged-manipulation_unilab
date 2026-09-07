"""Run from outside any source checkout after installing the three packages."""

import importlib.metadata
import os
import socket
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LEGGED_MANIPULATION_ASSET_CACHE"] = tempfile.mkdtemp(prefix="legged-wheel-assets-")


def deny(*args, **kwargs):
    raise AssertionError("Network access during installed robot loading")


socket.socket.connect = deny

import numpy as np
import uni_rl
import unilab
from unilab.base.config_adapter import BackendAdapter
from unilab.base.registry import ensure_registries, make

import legged_manipulation_unilab as package
from legged_manipulation_unilab.assets import resolve_scene
from legged_manipulation_unilab.cli import compose_config

for module in (package, unilab, uni_rl):
    assert "site-packages" in str(Path(module.__file__)), module.__file__
    print(module.__name__, module.__file__)
ensure_registries()
cfg = compose_config("ppo", "mujoco", ["algo.num_envs=2"])
overrides = BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="ppo").build_task_env_cfg_override()
env = make("Go2ArmManipLoco", "mujoco", env_cfg_override=overrides, num_envs=2)
try:
    obs, _ = env.reset(np.arange(2, dtype=np.int32))
    state = env.step(np.zeros((2, 18), dtype=np.float32))
    assert state.obs["obs"].shape == (2, 76)
    assert np.isfinite(state.reward).all()
finally:
    env.close()
for name in ("unilab", "unilab-rl", "legged-manipulation-unilab"):
    dist = importlib.metadata.distribution(name)
    print(name, dist.version, dist.read_text("direct_url.json"))
print("OFFLINE WHEEL RESET/STEP PASS", resolve_scene())
