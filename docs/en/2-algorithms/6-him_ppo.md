# HIM-PPO

The package owns HIM-PPO in `algos/him_ppo/` and `training/him.py`,
with `legged-train --algo him_ppo` and `legged-eval --algo him_ppo` entrypoints.
A MuJoCo owner is provided; there is no HIM-PPO Motrix owner.

The base configuration is `src/legged_manipulation_unilab/conf/ppo_him/config.yaml`.
The task owner is `src/legged_manipulation_unilab/conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`.

## Train and evaluate

```bash
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

`--run` accepts the parent log group, one run directory, or one checkpoint
file; it selects the latest checkpoint by default. Use `--checkpoint 100` to
select an earlier checkpoint.

Replace the absolute path with the run directory containing your checkpoint.
With the export extra installed, add `--export` during evaluation to export the policy.

## Observations and training stage

- `algo.num_one_step_obs=76`.
- `algo.num_actor_history=5`: the actor receives 380 history features.
- `algo.num_critic_history=1`: the critic receives 79 features including
  3 privileged linear-velocity components.
- `training.task_name=Go2ArmManipLoco`.
- The owner defaults to `env.arm_stage.freeze_arm_joints=true` and
  `env.arm_stage.disable_ee_goal_trajectory=true` for a fixed-arm stage.

When changing observation history, check the estimator's `velocity_target_start`,
`target_obs_start`, and checkpoint dimensions together.
See the [tuning notes](../8-manipulation/2-manip_loco.md).
