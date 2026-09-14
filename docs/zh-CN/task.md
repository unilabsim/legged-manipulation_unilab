# Go2 Arm Manip-Loco 任务

`go2_arm_manip_loco` 训练带 Airbot 六轴机械臂的 Unitree Go2 四足机器人。
注册环境名为 `Go2ArmManipLoco`。策略输出 18 维动作：12 维腿部关节 residual
和 6 维机械臂 residual。机械臂由阻尼最小二乘 IK 引导，跟踪采样得到的末端
目标。

每个 episode 为 20 秒仿真时间。环境采样底盘速度命令和经过碰撞检查的末端
目标，并支持机体质量、摩擦、armature、质心、PD 增益和间隔推力的随机化。

## 训练配置

| 算法 | 后端 | 并行环境数 | 策略更新次数 | 默认日志目录 | 说明 |
| --- | --- | ---: | ---: | --- | --- |
| PPO | MuJoCo | 4096 | 3000 | `logs/ppo-mujoco` | 完整移动操作配置。 |
| PPO | Motrix | 4096 | 3000 | `logs/ppo-motrix` | 完整移动操作配置。 |
| HIM-PPO | MuJoCo | 128 | 3000 | `logs/him-mujoco` | 固定机械臂阶段，使用观测历史。 |

上表描述本仓库支持的配置，不代表最终策略质量基准。硬件、后端、随机种子
和 override 都会影响结果。

## 安装并准备资产

```bash
uv sync --extra mujoco --extra motrix
uv run legged-assets
```

内置场景、网格和纹理会被复制到可写缓存；可用
`LEGGED_MANIPULATION_ASSET_CACHE` 指定缓存目录。加载机器人不需要访问
Hugging Face，也不需要运行时联网。

## 训练

```bash
uv run legged-train --algo ppo --sim mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.no_play=true
```

使用 `--sim` 选择后端，不要单独 override `training.sim_backend`。任务固定为
`go2_arm_manip_loco`。调参时在命令后追加 Hydra override。`--cfg` 会打印
组合后的配置，且不构造环境。
日志目录由对应配置自动选择；只有需要自定义位置时才覆盖
`training.log_root`。

快速冒烟训练：

```bash
uv run legged-train --algo ppo --sim mujoco algo.max_iterations=10 training.no_play=true
```

## 评估

```bash
uv run legged-eval --algo ppo --sim mujoco --run logs/ppo-mujoco/Go2ArmManipLoco
uv run legged-eval --algo ppo --sim motrix --run logs/ppo-motrix/Go2ArmManipLoco
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

`--run` 可以指向日志父目录、单个 run 目录或 checkpoint 文件。用
`--checkpoint 100` 选择较早的 checkpoint。无显示环境时设置
`MUJOCO_GL=egl`。如果 checkpoint 与当前策略或配置契约不一致，加载保护会
拒绝它，而不是静默 reshape。

## 下一步

- [调参指南](tuning.md)
- [HIM-PPO](him-ppo.md)
- [IK 诊断](ik-diagnostics.md)
