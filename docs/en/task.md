# Go2 Arm Manip-Loco task

`go2_arm_manip_loco` trains a Unitree Go2 quadruped with an Airbot six-DoF arm.
The registered environment name is `Go2ArmManipLoco`. The policy outputs an
18-dimensional action: twelve leg joint residuals and six arm residuals. Arm
motion is guided by damped-least-squares IK toward sampled end-effector goals.

Each episode lasts 20 simulated seconds. The environment samples base velocity
commands and collision-checked end-effector goals, and can randomize body mass,
friction, armature, center of mass, PD gains, and interval pushes.

## Training recipes

| Algorithm | Backend | Parallel environments | Policy updates | Default logs | Notes |
| --- | --- | ---: | ---: | --- | --- |
| PPO | MuJoCo | 4096 | 3000 | `logs/ppo-mujoco` | Full mobile-manipulation recipe. |
| PPO | Motrix | 4096 | 3000 | `logs/ppo-motrix` | Full mobile-manipulation recipe. |
| HIM-PPO | MuJoCo | 128 | 3000 | `logs/him-mujoco` | Fixed-arm stage with observation history. |

The table describes repository support; it is not a benchmark of final policy
quality. Results can vary with hardware, backend, random seeds, and overrides.

## Install and prepare assets

```bash
uv sync --extra mujoco --extra motrix
uv run legged-assets
```

The bundled scene, meshes, and textures are copied to a writable cache. Set
`LEGGED_MANIPULATION_ASSET_CACHE` to choose its location. Robot loading does
not require a Hugging Face download or runtime network access.

## Train

```bash
uv run legged-train --algo ppo --sim mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.no_play=true
```

Select the simulator with `--sim`; do not override `training.sim_backend` alone.
The task is already fixed to `go2_arm_manip_loco`. Append Hydra overrides after
the command to vary training length or task settings. `--cfg` prints the
composed configuration without constructing an environment.
Each owner selects the log directory shown above; override `training.log_root`
only when a custom location is required.

For a quick smoke run:

```bash
uv run legged-train --algo ppo --sim mujoco algo.max_iterations=10 training.no_play=true
```

## Evaluate

```bash
uv run legged-eval --algo ppo --sim mujoco --run logs/ppo-mujoco/Go2ArmManipLoco
uv run legged-eval --algo ppo --sim motrix --run logs/ppo-motrix/Go2ArmManipLoco
uv run legged-eval --algo him_ppo --sim mujoco --run logs/him-mujoco/Go2ArmManipLoco
```

`--run` accepts a log parent, one run directory, or one checkpoint file. Use
`--checkpoint 100` to select an older checkpoint. Set `MUJOCO_GL=egl` when
running without a display. Checkpoints with incompatible policy/configuration
contracts are rejected rather than silently reshaped.

## Next steps

- [Tuning guide](tuning.md)
- [HIM-PPO](him-ppo.md)
- [IK diagnostics](ik-diagnostics.md)
