import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

import legged_manipulation_unilab
from legged_manipulation_unilab.cli import compose_config
from legged_manipulation_unilab.training import him

_REPO_ROOT = Path(legged_manipulation_unilab.__file__).parent


def _train_him_ppo():
    return him


def _him_ppo_cfg(overrides=None):
    cfg = compose_config("him_ppo", "mujoco", [])
    cfg.training.play_only = True
    return cfg


def _train_rsl_rl(monkeypatch):
    from unilab.scripts import train_rsl_rl

    return train_rsl_rl


def _ppo_cfg(overrides=None):
    cfg = compose_config("ppo", "motrix", [])
    cfg.training.play_only = True
    cfg.play_profile.scene.source_model_file = "/tmp/scene.xml"
    cfg.play_profile.scene.ground_texture_file = "/tmp/floor.png"
    return cfg


def _compose(algo, overrides=None):
    return compose_config(algo, "motrix", [])


def test_go2_arm_manip_loco_motrix_eval_uses_visual_floor(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_arm_manip_loco/motrix", "training.play_only=true"])

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/go2_arm_manip_loco_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert captured["source_model_file"] == "/tmp/scene.xml"
    assert captured["ground_texture_file"] == "/tmp/floor.png"
    assert captured["skybox_rgb1"] == [0.90, 0.90, 0.91]
    assert captured["skybox_rgb2"] == [0.68, 0.68, 0.70]
    assert captured["ground_texrepeat"] == [0.25, 0.25]
    assert env_cfg_override["scene"].model_file == "/tmp/go2_arm_manip_loco_play_scene.xml"


def test_train_him_ppo_play_missing_checkpoint_returns_none_without_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    mod = _train_him_ppo()
    cfg = _him_ppo_cfg(["training.play_only=true"])

    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        mod,
        "create_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("play_him_ppo should not create an env before checkpoint resolution")
        ),
    )

    result = mod.play_him_ppo(cfg, device="cpu")

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in capsys.readouterr().out


def test_train_him_ppo_play_uses_shared_playback_session_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_him_ppo()
    cfg = _him_ppo_cfg(["training.play_only=true"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)
    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self):
            self.env = types.SimpleNamespace(
                cfg=types.SimpleNamespace(render_spacing=1.0),
            )
            self.runner = object()
            self.policy = lambda obs: obs
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return {"actor": "obs_0"}

        def step_once(self):
            self.step_calls += 1
            return {"actor": "obs_1"}

    fake_session = FakeSession()

    def fake_create_session(**kwargs: Any):
        captured["factory_kwargs"] = kwargs
        return fake_session, "actor", str(checkpoint)

    def fake_render_play_mode(env, **kwargs: Any):
        captured["render_kwargs"] = kwargs
        captured["init_obs"] = kwargs["initialize"]()
        captured["next_obs"] = kwargs["step"](captured["init_obs"])

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "create_rsl_rl_playback_session", fake_create_session)
    monkeypatch.setattr(mod, "render_play_mode", fake_render_play_mode)

    result = mod.play_him_ppo(cfg, device="cpu")

    assert result == str(run_dir / "play_video.mp4")
    factory_kwargs = captured["factory_kwargs"]
    playback_cfg = factory_kwargs["playback_cfg"]
    assert playback_cfg.task == cfg.training.task_name
    assert playback_cfg.action_mode == "policy"
    assert playback_cfg.num_envs == cfg.training.play_env_num
    assert factory_kwargs["device"] == "cpu"
    assert factory_kwargs["wrapper_cls"] is mod.RslRlVecEnvWrapper
    assert factory_kwargs["runner_cls"] is mod.HIMOnPolicyRunner
    assert factory_kwargs["guard_algo_name"] == "him_ppo"
    assert callable(factory_kwargs["runner_loader"])
    assert factory_kwargs["checkpoint_resolver"]() == str(checkpoint)
    assert callable(factory_kwargs["sim2sim_preflight"])
    assert fake_session.reset_calls == 1
    assert fake_session.step_calls == 1
    assert captured["init_obs"] == "obs_0"
    assert captured["next_obs"] == "obs_1"
    assert captured["render_kwargs"]["output_video"] == run_dir / "play_video.mp4"


def test_ppo_go2_arm_manip_loco_motrix_preserves_backend_overrides():
    cfg = _compose("ppo", overrides=["task=go2_arm_manip_loco/motrix"])

    assert cfg.training.task_name == "Go2ArmManipLoco"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 4096
    assert cfg.algo.max_iterations == 3000
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(2.0)
    assert cfg.env.domain_rand.randomize_dof_armature is False
    assert cfg.env.domain_rand.randomize_kp is False
    assert cfg.env.domain_rand.randomize_kd is False


def test_site_jacobian_benchmark_imports_with_mujoco_stub() -> None:
    code = textwrap.dedent(
        """
        import importlib.util
        import sys
        import types
        from pathlib import Path

        sys.modules["mujoco"] = types.ModuleType("mujoco")
        path = Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location("benchmark_site_jacobian", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        print(module.materialize_scene_visual_override.__module__)
        print("mujoco_backend", "unisim.backend.mujoco.backend" in sys.modules)
        """
    )
    script = _REPO_ROOT / "tools" / "benchmark_site_jacobian.py"
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "unisim.backend.mujoco.xml",
        "mujoco_backend False",
    ]
