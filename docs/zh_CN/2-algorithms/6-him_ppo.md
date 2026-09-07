# HIM-PPO

HIM-PPO 由本包的 `algos/him_ppo/` 与 `training/him.py` 实现，
入口为 `legged-train --algo him_ppo` 和 `legged-eval --algo him_ppo`。
当前提供 MuJoCo owner；没有 HIM-PPO Motrix owner。

基础配置为 `src/legged_manipulation_unilab/conf/ppo_him/config.yaml`。
任务 owner 为 `src/legged_manipulation_unilab/conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml`。

## 训练与回放

```bash
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
uv run legged-eval --algo him_ppo --sim mujoco training.log_root=./logs/eval-him algo.load_run=/absolute/path/to/him/run algo.checkpoint=-1
```

将绝对路径替换为实际 checkpoint 所在的 run 目录。
安装 export extra 后，可在回放命令中增加 `--export` 导出策略。

## 观测与训练阶段

- `algo.num_one_step_obs=76`。
- `algo.num_actor_history=5`，actor 输入为 380 维历史。
- `algo.num_critic_history=1`，critic 为包含 3 维线速度的 79 维观测。
- `training.task_name=Go2ArmManipLoco`。
- owner 默认设置 `env.arm_stage.freeze_arm_joints=true` 和
  `env.arm_stage.disable_ee_goal_trajectory=true`，用于固定机械臂阶段。

修改观测历史时同步核对 estimator 的 `velocity_target_start`、
`target_obs_start` 与 checkpoint 维度。
任务调参见[操作说明](../8-manipulation/2-manip_loco.md)。
