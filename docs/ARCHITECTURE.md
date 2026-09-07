# Architecture and extraction record

Roadmap: [UniLab #1528](https://github.com/unilabsim/UniLab/issues/1528).
This repository owns the complete Go2 + Airbot task and its HIM-PPO algorithm.
UniLab supplies the general environment framework and training adapters;
unilab-rl supplies shared PPO and runtime infrastructure; unisim supplies the
backend interface and simulator implementations.

## Ownership

Paths in the destination column are relative to
`src/legged_manipulation_unilab/`.

| Extracted concern | Destination |
| --- | --- |
| Go2 arm environment, IK, observation history, rewards and domain randomization | `tasks/go2_arm/` |
| Legacy locomotion helpers exclusively consumed by this task | `tasks/common/` |
| Task-specific spherical-coordinate and orientation helpers | `tasks/geometry.py` |
| HIM-PPO actor/critic, estimator, PPO update, storage and runner | `algos/him_ppo/` |
| PPO and HIM owner YAMLs | `conf/ppo/`, `conf/ppo_him/` |
| HIM training and evaluation assembly | `training/him.py` |
| IK viewer, diagnostics, orientation calibration and Jacobian benchmark | `tools/` |
| XML, Go2/Airbot meshes and floor texture | `assets/` |

The old `locomotion/common` base, commands, rewards, domain-randomization config
and provider had no remaining consumers outside the extracted task. They moved
with the task despite their former common-directory location. The task-specific
push helpers moved into `tasks/common/dr_utils.py`; shared reset-randomization
helpers remain in UniLab. Generic geometry helpers also remain in UniLab;
the source manifest lists the specific symbols extracted here.

UniLab and unilab_rl remove the task-specific implementation, configurations,
tools, tests, registrations and documentation from their maintained trees.
They provide no compatibility copy or alias for the removed HIM namespace.
General PPO, backend capabilities, scene interfaces and shared training helpers
remain at their existing owners.

## Integration contracts

- The `unilab.tasks` entry point registers the external task. Importing the
  package does not require a hard-coded Go2 arm import in UniLab.
- The local CLI selects an owner configuration and preserves task/backend
  identity. PPO delegates to UniLab's shared launcher and unilab-rl's PPO;
  HIM uses this package's runner. Backend behavior stays behind unisim's public
  interface.
- HIM consumes 76 single-step actor features over 5 history steps and a
  79-feature privileged critic observation. Observation dimensions and
  checkpoint guards remain part of the environment/algorithm boundary.
- Approximately 37 MB of assets ship in Git and the wheel. Asset preparation
  copies and verifies files in a writable cache; no Hugging Face runtime
  dependency is needed for this robot.

The integration follows UniLab's existing decisions:

- [ADR-0001: runtime and layer boundaries](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0001-runtime-model-and-layer-boundaries.md)
- [ADR-0003: task owner and config composition](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0003-task-owner-and-config-compose-contract.md)
- [ADR-0004: registry bootstrap](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0004-registry-bootstrap-contract.md)
- [ADR-0005: observation, critic, environment and IPC contracts](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/source/adr/ADR-0005-unified-obs-critic-env-and-ipc-contract.md)

## Provenance and dependency selection

[`MIGRATION_MANIFEST.json`](../MIGRATION_MANIFEST.json) records source commits,
source paths, pre-extraction SHA-256 hashes and partial-module symbol lists.
Those hashes describe the source files before import and integration changes;
they are not hashes of the edited destination files.

The separate [asset manifest](../src/legged_manipulation_unilab/assets/manifest.json)
records the pinned robot-data revision and packaged file hashes.
[NOTICE.md](../NOTICE.md) preserves attribution and points to upstream licenses.

UniLab and unilab-rl retain version 1.1.0. The Git sources and lockfile identify
the exact compatible dependency commits; the version number alone does not
distinguish a source tree before and after extraction.
