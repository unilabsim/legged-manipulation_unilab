# 架构与仓库边界

本包只维护一个任务：`go2_arm_manip_loco`。UniLab 继续提供通用
Manager-Based 环境运行时和训练适配层；unilab-rl 提供共享 PPO/运行时基础
设施；unisim 提供仿真接口和后端实现。

## 目录结构

下表路径相对于 `src/legged_manipulation_unilab/`。

| 内容 | 位置 |
| --- | --- |
| Manager-Based 任务配置和环境工厂 | `tasks/go2_arm/manip_loco.py` |
| action、command、observation、reward、termination 和 event terms | `tasks/go2_arm/manager_env.py` |
| 任务 dataclass、增益、传感器和阻尼最小二乘 IK | `tasks/go2_arm/base.py` |
| 球坐标与姿态工具 | `tasks/geometry.py` |
| HIM-PPO actor/critic、estimator、update、storage 和 runner | `algos/him_ppo/` |
| PPO 与 HIM-PPO YAML 配置 | `conf/ppo/`、`conf/ppo_him/` |
| HIM-PPO 训练与评估组装 | `training/him.py` |
| IK 查看器、诊断、标定和 Jacobian benchmark | `tools/` |
| Go2/Airbot XML、网格和纹理 | `assets/` |

本任务不保留 UniLab 通用 locomotion helper 的私有副本。共享 reset 随机化、
manager 接口、通用几何和后端能力仍由上游维护。

## 运行时契约

- `unilab.tasks` entry point 注册 `Go2ArmManipLoco`，UniLab 不需要硬编码导入
  该任务。
- 本地 CLI 选择算法和后端配置，并保持任务与后端身份。PPO 委托给 UniLab
  launcher 和 unilab-rl PPO；HIM-PPO 使用本包 runner。
- Manager-Based action term 将 18 维策略动作映射为腿部目标和机械臂 IK。
  command、observation、reward、event 与 termination terms 负责 Manip-Loco
  行为。
- 仿真差异位于 unisim 公共后端接口之后。任务代码不应依赖安装路径或仿真器
  私有对象。
- HIM-PPO 使用 76 维单步 actor 特征、5 步历史，以及 79 维 privileged critic
  观测。这些维度和 checkpoint 保护属于任务/算法边界。
- 约 37 MB 资产随 Git 与 wheel 提供。资产准备会复制到可写缓存；这个机器人
  不需要网络访问。

## 维护约定

- `docs/en/` 与 `docs/zh-CN/` 保持平行；行为变更需要同时更新两种语言。
- 运行时 CLI 工具放在 `src/legged_manipulation_unilab/tools/`，并在
  `pyproject.toml` 声明 entry point。
- 持续回归检查放在 `tests/`。一次性本地诊断应写成文档命令，而不是保存为
  永久可执行附件。
- 调整 reward term 权重时保留 term 函数名和参数；这些名称属于配置兼容性。
