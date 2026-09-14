# Legged manipulation for UniLab

[中文](README_zh.md)

## Overview

This is a reinforcement-learning task repository for a Unitree Go2 quadruped
combined with a six-DoF Airbot arm. It provides the complete Manip-Loco
training configuration, robot scene, IK diagnostics, and both PPO and HIM-PPO
training recipes.

The task is registered with UniLab as `Go2ArmManipLoco` and supports MuJoCo and
Motrix. The robot XML, meshes, and textures are bundled with the repository, so
robot data does not need to be downloaded from Hugging Face before training.

## Capabilities and showcase

### Capabilities

| Capability | Description |
| --- | --- |
| Mobile manipulation | Adds Airbot end-effector goal tracking to Go2 locomotion. |
| PPO training | Provides MuJoCo and Motrix training recipes. |
| HIM-PPO training | Uses actor observation history and a privileged critic input. |
| IK diagnostics | Visualizes targets, checks Jacobians, calibrates orientation, and compares numerical methods. |
| Local robot assets | Bundles the Go2 + Airbot scene, meshes, and textures for offline loading. |
| Playback and export | Selects checkpoints from training logs, renders playback, and exports HIM-PPO policies. |

### Showcase

<!-- TODO: replace this with a 10–20 second training/playback overview GIF. -->
<!-- `docs/assets/showcase.gif` -->

## Reproduction

### Install

```bash
git clone https://github.com/unilabsim/legged-manipulation_unilab.git
cd legged-manipulation_unilab
uv sync --extra mujoco --extra motrix
```

Add `--extra export` when exporting HIM-PPO policies. Prepare the local asset
cache with:

```bash
uv run legged-assets
```

### Train

All built-in training recipes default to 3000 policy updates:

```bash
uv run legged-train --algo ppo --sim mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.no_play=true
```

Run directories are created under `logs/ppo-mujoco`, `logs/ppo-motrix`, or
`logs/him-mujoco`, respectively. Override `training.log_root` only when a
different location is required.

Use `--cfg` to inspect the composed configuration. For a short smoke run,
override the update count:

```bash
uv run legged-train --algo ppo --sim mujoco algo.max_iterations=10 training.no_play=true
```

### Evaluate and export

Pass a parent log group, one training-run directory, or one `model_*.pt` file to
`--run`; the latest checkpoint is selected by default:

```bash
uv run legged-eval --algo ppo --sim mujoco --run logs/ppo-mujoco/Go2ArmManipLoco
uv run legged-eval --algo ppo --sim motrix --run logs/ppo-motrix/Go2ArmManipLoco
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

Select another checkpoint with `--checkpoint 100`. Set `MUJOCO_GL=egl` on a
headless machine. With the export extra installed, add `--export` to HIM-PPO
evaluation to export the policy.

### IK checks

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

The viewer requires a graphical display. Diagnosis, calibration, and the
benchmark use the bundled MuJoCo scene.

## Documentation

- [Documentation index](docs/README.md)
- [Task and reproduction](docs/en/task.md)
- [Using HIM-PPO](docs/en/him-ppo.md)
- [Tuning guide](docs/en/tuning.md)
- [IK diagnostics](docs/en/ik-diagnostics.md)
- [Maintainer architecture notes](docs/en/architecture.md)

## License

The repository is primarily Apache-2.0 and also contains BSD-3-Clause components
and upstream robot-asset notices. See [LICENSE](LICENSE), [LICENSES](LICENSES),
and [NOTICE](NOTICE.md).
