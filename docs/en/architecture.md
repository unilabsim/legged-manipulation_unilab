# Architecture and repository boundaries

This package owns one task: `go2_arm_manip_loco`. UniLab remains the general
Manager-Based environment runtime and training adapter, unilab-rl provides
shared PPO/runtime infrastructure, and unisim provides simulator interfaces and
backend implementations.

## Layout

Paths are relative to `src/legged_manipulation_unilab/`.

| Concern | Location |
| --- | --- |
| Manager-Based task configuration and environment factory | `tasks/go2_arm/manip_loco.py` |
| Action, command, observation, reward, termination, and event terms | `tasks/go2_arm/manager_env.py` |
| Task dataclasses, gains, sensors, and damped-least-squares IK | `tasks/go2_arm/base.py` |
| Spherical-coordinate and orientation helpers | `tasks/geometry.py` |
| HIM-PPO actor/critic, estimator, update, storage, and runner | `algos/him_ppo/` |
| PPO and HIM-PPO YAML recipes | `conf/ppo/`, `conf/ppo_him/` |
| HIM-PPO training and evaluation assembly | `training/him.py` |
| IK viewer, diagnostics, calibration, and Jacobian benchmark | `tools/` |
| Go2/Airbot XML, meshes, and textures | `assets/` |

The task does not keep a private copy of UniLab's generic locomotion helper
tree. Shared reset-randomization utilities, manager interfaces, generic geometry,
and backend capabilities stay upstream.

## Runtime contracts

- The `unilab.tasks` entry point registers `Go2ArmManipLoco` without requiring a
  hard-coded task import inside UniLab.
- The local CLI selects an algorithm and backend owner, then preserves task and
  backend identity. PPO delegates to UniLab's launcher and unilab-rl's PPO;
  HIM-PPO uses the runner in this package.
- The Manager-Based action term maps the 18-dimensional policy action through
  task-owned leg targets and arm IK. Command, observation, reward, event, and
  termination terms own Manip-Loco behavior.
- Simulator differences remain behind unisim's public backend interface. Task
  code should not branch on installation paths or private simulator objects.
- HIM-PPO uses 76 single-step actor features over five history steps and a
  79-feature privileged critic observation. These dimensions and checkpoint
  guards are part of the task/algorithm boundary.
- Approximately 37 MB of assets ship in Git and the wheel. Asset preparation
  copies and verifies files in a writable cache and does not require network
  access for this robot.

## Maintenance expectations

- User documentation stays parallel in `docs/en/` and `docs/zh-CN/`; behavior
  changes should update both language paths.
- Runtime CLI tools belong under `src/legged_manipulation_unilab/tools/` and must
  declare their entry point in `pyproject.toml`.
- Continuous regression checks belong in `tests/`. One-off local diagnostics can
  be documented as commands rather than stored as permanent executable fixtures.
- Preserve reward term function names and parameters when changing term weights;
  those names are part of configuration compatibility.
