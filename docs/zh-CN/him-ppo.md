# HIM-PPO

HIM-PPO 让 actor 使用观测历史训练，同时给 critic 提供特权状态信息。在本
仓库中，它用于 Go2 + Airbot 任务，并运行在 MuJoCo 后端。

使用：

```bash
uv run legged-train --algo him_ppo --sim mujoco training.no_play=true
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

当前没有 HIM-PPO Motrix 配置。

## 模型输入

| 项目 | 数值 | 含义 |
| --- | ---: | --- |
| 单步观测 | 76 | actor estimator 输入使用的特征。 |
| actor 历史 | 5 | actor 输入 380 维历史特征。 |
| critic 历史 | 1 | critic 输入单步 79 维特权观测。 |
| 特权速度 | 3 维 | critic 可见线速度。 |

MuJoCo 配置从固定机械臂阶段开始：
`env.arm_stage.freeze_arm_joints=true` 且
`env.arm_stage.disable_ee_goal_trajectory=true`。这样早期 HIM-PPO 训练更关注
运动控制，同时保留完整观测接口。

## Checkpoint 契约

观测历史属于策略契约。加载或对比 checkpoint 前，应保持以下数值一致：

- `algo.num_one_step_obs`
- `algo.num_actor_history`
- `algo.num_critic_history`
- estimator 观测偏移，例如 `velocity_target_start` 与 `target_obs_start`
- actor/critic 网络维度

runner 和评估保护会拒绝不匹配的历史或 checkpoint 维度。修改这些值后应重新
训练，而不是 reshape 旧 checkpoint。

## 导出

安装 `export` extra 后，在评估命令中增加 `--export`：

```bash
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco --export
```

导出策略用于部署实验；上机前必须结合回放和目标安全控制流程重新审查。

## 相关文档

- [Go2 Arm Manip-Loco 任务](task.md)
- [调参指南](tuning.md)
