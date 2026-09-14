from __future__ import annotations

from typing import Any

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

from legged_manipulation_unilab.algos.him_ppo.runner import HIMOnPolicyRunner


class _LoggingEnv:
    """Minimal environment that publishes an RSL-RL-compatible rollout."""

    def __init__(self) -> None:
        self.cfg = {"purpose": "HIM-PPO logging contract test"}
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_obs = 4
        self.num_privileged_obs = 8
        self.num_actions = 1
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self._step = 0

    def _observations(self) -> TensorDict:
        return TensorDict(
            {
                "actor": torch.randn(self.num_envs, self.num_obs),
                "critic": torch.randn(self.num_envs, self.num_privileged_obs),
            },
            batch_size=[self.num_envs],
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        self._step = 0
        return self._observations(), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        del actions
        self._step += 1
        dones = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        if self._step == 2:
            dones[0] = True

        return (
            self._observations(),
            torch.tensor([[1.0, 4.0], [3.0, 2.0]][self._step - 1]),
            dones,
            {
                "log": {
                    "reward/foo": (1.0, 3.0)[self._step - 1],
                    "terminal_metric": (5.0, 9.0)[self._step - 1],
                }
            },
        )


def _runner_config() -> dict[str, Any]:
    return {
        "num_one_step_obs": 4,
        "num_actor_history": 1,
        "num_steps_per_env": 2,
        "save_interval": 10,
        "policy": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
        "estimator": {
            "enc_hidden_dims": [4],
            "tar_hidden_dims": [4],
            "num_prototype": 2,
            "velocity_target_start": 0,
            "target_obs_start": 4,
        },
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "desired_kl": None,
            "schedule": "fixed",
        },
    }


def test_him_logging_uses_rsl_rl_rollout_aggregation(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    runner = HIMOnPolicyRunner(_LoggingEnv(), _runner_config(), log_dir=str(tmp_path))

    try:
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    finally:
        runner.close()

    accumulator = EventAccumulator(str(tmp_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    assert "reward/foo" in tags
    assert "Episode/terminal_metric" in tags
    assert "Train/mean_reward" in tags
    assert "Loss/value" in tags
    assert "Loss/estimation" in tags
    assert not any(tag.startswith("Episode_Reward/") for tag in tags)

    def scalar(tag: str) -> float:
        events = accumulator.Scalars(tag)
        assert len(events) == 1
        return float(events[0].value)

    # RSL-RL aggregates every rollout step, rather than using only the final
    # step's ``infos["log"]`` dictionary.
    assert scalar("reward/foo") == pytest.approx(2.0)
    assert scalar("Episode/terminal_metric") == pytest.approx(7.0)
    assert scalar("Train/mean_reward") == pytest.approx(4.0)
    assert scalar("Train/mean_episode_length") == pytest.approx(2.0)
    assert runner.logger.tot_timesteps == 4
    assert list(runner.logger.rewbuffer) == [4.0]
    assert list(runner.logger.lenbuffer) == [2.0]
    assert (tmp_path / "model_1.pt").is_file()

    console = capsys.readouterr().out
    assert "Learning iteration 0/1" in console
    assert "reward/foo:" in console
    assert "Episode_Reward/" not in console
