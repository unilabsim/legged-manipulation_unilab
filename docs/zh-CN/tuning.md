# 调参指南

从一个内置配置出发，少量修改参数，并保存每次实验的组合配置。长时间训练前
先用 `legged-train --cfg` 确认最终生效的 YAML 值。

## 常用控制项

| 目标 | 配置 | 说明 |
| --- | --- | --- |
| 训练长度 | `algo.max_iterations` | 内置配置为 3000 次更新。 |
| 机械臂 residual 幅度 | `env.control_config.arm_action_scale` | 为 0 时机械臂跟随 IK 解。 |
| 末端目标 | `env.goal_ee` | 采样范围、轨迹时间、保持时间和碰撞限制。 |
| 机械臂阶段 | `env.arm_stage` | 冻结关节或关闭目标轨迹，用于分阶段训练。 |
| 底盘跟踪权衡 | `reward.tracking_lin_vel`、`reward.tracking_ang_vel`、`reward.stand_still` | 影响步态和站立行为。 |
| 训练难度 | `env.domain_rand` | 质量、摩擦、armature、质心、PD 和推力随机化。 |
| 观测历史 | `algo.num_actor_history`、`algo.num_critic_history` | 兼容性维度；修改前见 HIM-PPO。 |

奖励项是 Manager-Based term。调整权重时保留每个 term 的 `func`、`weight`
和 `params.name`。

Iteration 不能直接比较并行环境数不同的配置。PPO 每次更新采集 4096 × 24 条
transition，HIM-PPO 采集 128 × 24 条。需要相同采样预算时，请比较总环境步数，
或设置 `training.num_timesteps`。

## 建议流程

1. 先用少于 20 次更新的配置做冒烟训练。
2. 恢复预期的更新次数。
3. 每次只修改一个行为组。
4. 对比单一改动时保持相同 seed。
5. 使用相同的 checkpoint 选择规则评估。
6. 记录完整组合配置或精确 Hydra override。

随机化和奖励变化会改变 episode 长度与终止率。不要只看标量奖励，也要查看
回放行为。

## 修改后的检查

聚焦检查：

```bash
uv run pytest tests/tasks tests/algos tests/test_entrypoints.py
uv run legged-assets
```

完整测试：

```bash
uv run pytest
```

修改资产或 XML 后，验证 MuJoCo 可以构造包内场景：

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

修改 IK 或 Jacobian 后，同时运行 [IK 诊断](ik-diagnostics.md)。
