# IK 诊断

IK 工具加载本仓库内置的 Go2 + Airbot MuJoCo 场景，用于回答三个实际问题：
末端目标是否可达、Jacobian 方向是否正确、当前机器上的控制参数是否稳定。

## 命令

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

| 命令 | 用途 | 说明 |
| --- | --- | --- |
| `legged-ik` | 交互式目标查看器 | 需要图形显示。 |
| `legged-diagnose-ik` | 检查 Jacobian 与闭环行为 | 修改 IK 或传感器后运行。 |
| `legged-calibrate` | 搜索末端姿态参数 | 不接受 policy 参数。 |
| `legged-benchmark-jacobian` | 对比数值 Jacobian 实现 | 只报告当前机器。 |

每个命令都可用 `--help` 查看完整选项。

## 无显示环境冒烟检查

```bash
uv run legged-ik --headless-steps 3 --jacobian-source direct
uv run legged-diagnose-ik --steps 3 --disable-gain-randomization
```

短扫描可以发现 site 缺失、Jacobian 方向错误或控制符号错误，但不能证明闭环
收敛，也不应单独用它选择生产增益。

## 场景加载检查

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

替换机器人 XML、网格、纹理或资产缓存逻辑后运行该检查。更完整的修改检查见
[调参指南](tuning.md)。
