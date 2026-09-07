# Manip-Loco tuning and IK

The Go2 + Airbot task, scene, and training configurations are owned by this
package. Start with the [task commands](../4-tasks/4-manip_loco.md).
PPO has MuJoCo and Motrix owners; HIM-PPO has a MuJoCo owner.

## Tuning hints

- `env.control_config.arm_action_scale`: magnitude of the arm residual action.
  The PPO owners start at zero residual scale.
- `env.goal_ee`: end-effector goal sampling ranges and trajectory timing.
- `reward.scales.tracking_lin_vel`, `reward.scales.tracking_ang_vel`, and
  `reward.scales.stand_still`: base-behavior trade-offs.
- `env.domain_rand`: mass, friction, push, and PD randomization change training
  difficulty; available keys and defaults belong to the selected owner.
- `env.arm_stage`: check arm freezing and goal-trajectory settings before
  comparing locomotion-only and manipulation runs. HIM defaults to a fixed arm.

## IK tools

The following commands use the bundled MuJoCo scene:

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

`legged-ik` opens an interactive viewer. `legged-diagnose-ik` checks Jacobian
and closed-loop behavior. `legged-calibrate` searches end-effector orientation
settings and accepts no ONNX-policy argument. The benchmark measures the current
machine; running it does not establish a general performance guarantee.

## Checks after changes

```bash
uv run pytest tests/tasks tests/algos tests/test_entrypoints.py
uv run legged-assets
```

When changing XML or assets, confirm that MuJoCo loads the bundled scene:

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

Run the diagnostics after changing IK or Jacobian behavior. Preserve observation
history and checkpoint dimensions when comparing a modified policy.
