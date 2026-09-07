# Manip-Loco

`go2_arm_manip_loco` 将 Go2 运动控制与 Airbot 机械臂结合。
注册环境为 `Go2ArmManipLoco`。安装本包后，任务通过 `unilab.tasks`
entry-point group 注册。

## Owner 配置

以下路径均相对于 `src/legged_manipulation_unilab/`。

| 算法 | 后端 | Owner |
| --- | --- | --- |
| PPO | MuJoCo | `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml` |
| PPO | Motrix | `conf/ppo/task/go2_arm_manip_loco/motrix.yaml` |
| HIM-PPO | MuJoCo | `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` |

场景为 `assets/robots/go2_arm/scene_flat.xml`。XML、网格和纹理随包提供，
并复制到可写本地缓存；加载机器人资源不需要从 Hugging Face 下载。

## 命令

```bash
uv run legged-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
```

通过 `--sim` 选择后端，不要单独 override `training.sim_backend`。
本仓库只有该任务，因此可以省略 `--task go2_arm_manip_loco`。
`--cfg` 打印组合后的 owner 配置，不构造环境。

观测历史配置见 [HIM-PPO](../2-algorithms/6-him_ppo.md)，
调参与 IK 工具见[操作说明](../8-manipulation/2-manip_loco.md)。
