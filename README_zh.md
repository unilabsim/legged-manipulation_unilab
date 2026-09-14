# UniLab 足式移动操作

Go2 + Airbot 的运动控制与操作任务，提供 PPO 和 HIM-PPO。
本仓库是从 UniLab 与 unilab_rl 拆出的任务、配置、IK 工具、机器人资源和
HIM-PPO 实现的唯一维护位置。

[English](README.md) · [拆分 roadmap：UniLab #1528](https://github.com/unilabsim/UniLab/issues/1528)

## 安装

```bash
git clone https://github.com/unilabsim/legged-manipulation_unilab.git
cd legged-manipulation_unilab
uv sync --extra mujoco
```

Motrix 使用 `uv sync --extra motrix`；需要两个后端时同时选择两个 extra。
导出策略时增加 `--extra export`。UniLab 与 unilab-rl 的原版本号保持 1.1.0；
[pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 中的 Git 来源选择与此次
拆分兼容的提交。

约 37 MB 的机器人 XML、网格和纹理随 Git 仓库与 Python 包提供。
加载机器人资源无需访问 Hugging Face，也无需运行时联网。
`uv run legged-assets` 准备可写本地缓存；
可通过 `LEGGED_MANIPULATION_ASSET_CACHE` 指定缓存目录。

## 训练与回放

| 算法 | 已提供的 owner 配置 |
| --- | --- |
| PPO | MuJoCo、Motrix |
| HIM-PPO | MuJoCo |

该表列出配置范围，不代表训练效果基准。

```bash
uv run legged-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
```

任务默认为 `go2_arm_manip_loco`。在命令后追加 Hydra override 调参；
`--cfg` 打印组合后的配置。使用 `--sim` 选择后端。

评估时把 `--run` 指向日志父目录，命令会自动选择最新的训练 run 和 checkpoint：

```bash
uv run legged-eval --algo ppo --sim mujoco --run logs/ppo-mujoco/Go2ArmManipLoco
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

`--run` 也接受单个训练 run 目录或 `model_*.pt` 文件。用
`--checkpoint 100` 选择指定 checkpoint。无显示环境时加
`MUJOCO_GL=egl`。如果 checkpoint 与当前策略/配置契约不一致，sim2sim 检查会
拒绝加载；修改这些设置后请重新训练。

安装 export extra 后，HIM-PPO 回放支持 `--export`。
观测历史维度和默认机械臂训练阶段见 [HIM-PPO](docs/zh_CN/2-algorithms/6-him_ppo.md)。

## 工具与文档

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

IK 查看器需要图形显示环境。诊断、姿态标定与 Jacobian benchmark 使用包内
MuJoCo 场景；标定工具没有 ONNX 策略参数。通过各命令的 `--help` 查看选项。

- [任务 owner 与命令](docs/zh_CN/4-tasks/4-manip_loco.md)
- [调参与 IK 检查](docs/zh_CN/8-manipulation/2-manip_loco.md)
- [架构与拆分记录](docs/ARCHITECTURE.md)
- [迁移验收记录](docs/VALIDATION.md)
- [源文件清单](MIGRATION_MANIFEST.json)与[许可说明](NOTICE.md)
