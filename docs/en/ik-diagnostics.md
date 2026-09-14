# IK diagnostics

The IK tools load the packaged Go2 + Airbot MuJoCo scene. They help answer
three practical questions: whether the end-effector target is reachable, whether
the Jacobian direction is correct, and whether a controller setting is stable on
the current machine.

## Commands

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

| Command | Use | Notes |
| --- | --- | --- |
| `legged-ik` | Interactive target viewer | Requires a graphical display. |
| `legged-diagnose-ik` | Check Jacobian and closed-loop behavior | Useful after IK or sensor changes. |
| `legged-calibrate` | Search end-effector orientation settings | Takes no policy argument. |
| `legged-benchmark-jacobian` | Compare numerical Jacobian implementations | Reports only the current machine. |

Run `--help` with any command for all options.

## Headless smoke checks

```bash
uv run legged-ik --headless-steps 3 --jacobian-source direct
uv run legged-diagnose-ik --steps 3 --disable-gain-randomization
```

These short sweeps can reveal missing sites, broken Jacobian directions, or
invalid controller signs. They do not prove closed-loop convergence and should
not be used alone to choose production gains.

## Scene loading check

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

Use this after replacing robot XML, meshes, textures, or asset-cache logic. For
broader change checks, see the [tuning guide](tuning.md).
