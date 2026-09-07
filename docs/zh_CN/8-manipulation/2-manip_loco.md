# Manip-Loco 调参与 IK

Go2 + Airbot 任务、场景和训练配置均由本包维护。
入口见[任务命令](../4-tasks/4-manip_loco.md)。
PPO 提供 MuJoCo 和 Motrix owner；HIM-PPO 提供 MuJoCo owner。

## 调参提示

- `env.control_config.arm_action_scale`：机械臂 residual action 的幅度。
  PPO owner 默认 residual scale 为零。
- `env.goal_ee`：末端目标采样范围与轨迹时间。
- `reward.scales.tracking_lin_vel`、`reward.scales.tracking_ang_vel`、
  `reward.scales.stand_still`：底盘行为的权衡。
- `env.domain_rand`：质量、摩擦、推力和 PD 随机化改变训练难度；
  可用字段和默认值由所选 owner 决定。
- `env.arm_stage`：比较纯运动与操作训练前，检查机械臂冻结和目标轨迹设置。
  HIM 默认使用固定机械臂阶段。

## IK 工具

以下命令使用包内 MuJoCo 场景：

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

`legged-ik` 打开交互查看器。`legged-diagnose-ik` 检查 Jacobian 和闭环行为。
`legged-calibrate` 搜索末端姿态参数，不接受 ONNX 策略参数。
benchmark 测量当前机器，不构成通用性能保证。

## 修改后的检查

```bash
uv run pytest tests/tasks tests/algos tests/test_entrypoints.py
uv run legged-assets
```

修改 XML 或资源后，确认 MuJoCo 可加载包内场景：

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

修改 IK 或 Jacobian 行为后运行诊断。比较修改后的策略时，
保持观测历史与 checkpoint 维度一致。
