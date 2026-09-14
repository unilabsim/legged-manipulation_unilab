# HIM-PPO

HIM-PPO trains an actor from observation history while giving the critic
privileged state information. In this repository it is intended for the Go2 +
Airbot task and runs on MuJoCo.

Use:

```bash
uv run legged-train --algo him_ppo --sim mujoco training.no_play=true
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

There is no HIM-PPO Motrix recipe.

## Model inputs

| Item | Value | Meaning |
| --- | ---: | --- |
| Single-step observation | 76 | Features used by the actor estimator input. |
| Actor history | 5 | Actor receives 380 history features. |
| Critic history | 1 | Critic receives a one-step 79-feature privileged observation. |
| Privileged velocity | 3 components | Linear velocity available to the critic. |

The MuJoCo recipe starts with a fixed arm:
`env.arm_stage.freeze_arm_joints=true` and
`env.arm_stage.disable_ee_goal_trajectory=true`. This keeps early HIM-PPO
training focused on locomotion while preserving the full observation interface.

## Checkpoint contract

Observation history is part of the policy contract. Before loading or comparing
checkpoints, keep these values consistent:

- `algo.num_one_step_obs`
- `algo.num_actor_history`
- `algo.num_critic_history`
- estimator observation offsets such as `velocity_target_start` and
  `target_obs_start`
- actor and critic network dimensions

The runner and evaluation guards reject mismatched histories or checkpoint
dimensions. If these values change, train a new policy rather than reshaping an
old checkpoint.

## Export

Install the `export` extra and add `--export` to evaluation:

```bash
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco --export
```

The exported policy is intended for deployment experiments; always review it
with playback and your target safety controls before running on hardware.

## See also

- [Go2 Arm Manip-Loco task](task.md)
- [Tuning guide](tuning.md)
