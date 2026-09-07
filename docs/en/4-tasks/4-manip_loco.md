# Manip-Loco

`go2_arm_manip_loco` combines Go2 locomotion with the Airbot arm.
The registered environment is `Go2ArmManipLoco`. Installing this package exposes
its task registration through the `unilab.tasks` entry-point group.

## Owner configurations

All paths below are relative to `src/legged_manipulation_unilab/`.

| Algorithm | Backend | Owner |
| --- | --- | --- |
| PPO | MuJoCo | `conf/ppo/task/go2_arm_manip_loco/mujoco.yaml` |
| PPO | Motrix | `conf/ppo/task/go2_arm_manip_loco/motrix.yaml` |
| HIM-PPO | MuJoCo | `conf/ppo_him/task/go2_arm_manip_loco/mujoco.yaml` |

The scene is `assets/robots/go2_arm/scene_flat.xml`. Its XML, meshes, and textures
ship with the package and are copied into a writable local cache.
Robot asset loading does not fetch data from Hugging Face.

## Commands

```bash
uv run legged-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
```

Select the backend with `--sim`; do not override `training.sim_backend` alone.
`--task go2_arm_manip_loco` is optional because this is the only task.
Use `--cfg` to inspect the composed owner without constructing an environment.

See [HIM-PPO](../2-algorithms/6-him_ppo.md) for history configuration and
[manipulation notes](../8-manipulation/2-manip_loco.md) for tuning and IK tools.
