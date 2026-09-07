from __future__ import annotations

import json
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from legged_manipulation_unilab import assets
from legged_manipulation_unilab.algos.him_ppo.runner import HIMOnPolicyRunner
from legged_manipulation_unilab.cli import compose_config


@pytest.mark.parametrize("algo,sim", [("ppo", "mujoco"), ("ppo", "motrix"), ("him_ppo", "mujoco")])
def test_owner_composition_and_identity(algo, sim):
    cfg = compose_config(algo, sim, [])
    assert cfg.training.task_name == "Go2ArmManipLoco"
    assert cfg.training.sim_backend == sim
    with pytest.raises(ValueError, match="identity"):
        compose_config(algo, sim, ["training={sim_backend:unknown}"])


def test_installed_registration_in_spawn(tmp_path):
    script = tmp_path / "spawn_check.py"
    script.write_text("""
import multiprocessing
def worker(queue):
    from unilab.base.registry import ensure_registries, list_registered_envs
    ensure_registries()
    queue.put(list_registered_envs()["Go2ArmManipLoco"]["available_backends"])
if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    assert {"mujoco", "motrix"} <= set(queue.get(timeout=30))
    process.join(timeout=30)
    assert process.exitcode == 0
""")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path, timeout=60)


def test_bundled_assets_work_offline_and_repair_cache(monkeypatch, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setenv("LEGGED_MANIPULATION_ASSET_CACHE", str(tmp_path / "assets"))
    monkeypatch.setattr(
        socket.socket, "connect", lambda *_: pytest.fail("Unexpected network access")
    )
    root = assets.ensure_assets()
    model = mujoco.MjModel.from_xml_path(str(root / "robots/go2_arm/scene_flat.xml"))
    assert model.nu == 18
    manifest = json.loads((assets.ASSETS_ROOT_PATH / "manifest.json").read_text())
    relative = next(path for path in manifest["sha256"] if path.endswith(".obj"))
    mesh = root / relative
    original = mesh.read_bytes()
    mesh.write_bytes(b"x" * len(original))
    assert assets.ensure_assets() == root
    assert mesh.read_bytes() == original


class ToyEnv:
    num_envs = 4
    num_obs = 20
    num_privileged_obs = 7
    num_actions = 2
    max_episode_length = 20

    def __init__(self):
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def observations(self):
        return TensorDict(
            {"actor": torch.randn(4, 20), "critic": torch.randn(4, 7)}, batch_size=[4]
        )

    def reset(self):
        return self.observations(), {}

    def step(self, actions):
        assert torch.isfinite(actions).all()
        return (
            self.observations(),
            -actions.square().sum(dim=1),
            torch.zeros(4, dtype=torch.bool),
            {},
        )


def runner_config():
    return dict(
        num_one_step_obs=4,
        num_actor_history=5,
        num_steps_per_env=4,
        policy=dict(actor_hidden_dims=[16], critic_hidden_dims=[16]),
        estimator=dict(enc_hidden_dims=[16, 8], tar_hidden_dims=[16], num_prototype=4),
        algorithm=dict(num_learning_epochs=1, num_mini_batches=1),
    )


def test_him_update_checkpoint_resume_and_legacy_load(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    runner = HIMOnPolicyRunner(ToyEnv(), runner_config())
    runner.learn(1)
    assert runner.current_learning_iteration == 1
    assert runner.actor_critic.estimator.optimizer.state
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    restored = HIMOnPolicyRunner(ToyEnv(), runner_config())
    restored.load(str(checkpoint))
    obs = torch.randn(3, 20)
    torch.testing.assert_close(
        runner.get_inference_policy()(obs), restored.get_inference_policy()(obs)
    )
    assert restored.actor_critic.estimator.optimizer.state
    restored.learn(1)
    assert restored.current_learning_iteration == 2
    legacy = torch.load(checkpoint, weights_only=True)
    del legacy["estimator_optimizer_state_dict"]
    torch.save(legacy, checkpoint)
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 1


def test_him_rejects_environment_dimension_mismatch():
    env = ToyEnv()
    env.num_obs = 24
    with pytest.raises(ValueError, match="observation dimension mismatch"):
        HIMOnPolicyRunner(env, runner_config())


def test_him_export_matches_policy_for_multiple_batch_sizes(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    runner = HIMOnPolicyRunner(ToyEnv(), runner_config())
    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    jit = torch.jit.load(str(tmp_path / "policy.pt"))
    onnx = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    for batch in [1, 3]:
        obs = torch.randn(batch, 20)
        expected = runner.get_inference_policy()(obs).detach().numpy()
        np.testing.assert_allclose(jit(obs).detach().numpy(), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            onnx.run(None, {"obs_history": obs.numpy()})[0], expected, rtol=1e-5, atol=1e-6
        )


def test_him_checkpoint_shape_guard_runs_before_environment(monkeypatch, tmp_path):
    from legged_manipulation_unilab.training import him

    checkpoint = tmp_path / "model.pt"
    torch.save({"actor_state_dict": {"estimator.encoder.0.weight": torch.zeros(8, 12)}}, checkpoint)
    monkeypatch.setattr(him, "parse_checkpoint_path", lambda *a, **k: (checkpoint, tmp_path))
    monkeypatch.setattr(him, "make_sim2sim_preflight", lambda *a, **k: lambda _: None)
    monkeypatch.setattr(him, "create_env", lambda *a, **k: pytest.fail("Env built before guard"))
    with pytest.raises(ValueError, match="checkpoint observation dimension mismatch"):
        him.play_him_ppo(compose_config("him_ppo", "mujoco", []), device="cpu")
