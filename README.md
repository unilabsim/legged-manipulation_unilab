# Legged manipulation for UniLab

Go2 + Airbot locomotion and manipulation with PPO and HIM-PPO. This repository
is the sole owner of the task, its configuration, IK tools, robot assets, and
HIM-PPO implementation extracted from UniLab and unilab_rl.

[中文](README_zh.md) · [Roadmap: UniLab #1528](https://github.com/unilabsim/UniLab/issues/1528)

## Install

```bash
git clone https://github.com/unilabsim/legged-manipulation_unilab.git
cd legged-manipulation_unilab
uv sync --extra mujoco
```

Use `uv sync --extra motrix` for Motrix, or select both extras to install both
backends. Add `--extra export` when exporting policies. UniLab and unilab-rl
retain their original version numbers, 1.1.0; the Git sources in
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) select the
extraction-compatible revisions.

Approximately 37 MB of robot XML, meshes, and textures are bundled in Git and
the Python package. Robot asset loading requires no Hugging Face download or
runtime network access. `uv run legged-assets` prepares a writable local cache;
`LEGGED_MANIPULATION_ASSET_CACHE` can select its directory.

## Train and evaluate

| Algorithm | Owner configurations |
| --- | --- |
| PPO | MuJoCo, Motrix |
| HIM-PPO | MuJoCo |

These are the available owner configurations, not a training-quality benchmark.

```bash
uv run legged-train --algo ppo --sim mujoco training.log_root=./logs/ppo-mujoco training.no_play=true
uv run legged-train --algo ppo --sim motrix training.log_root=./logs/ppo-motrix training.no_play=true
uv run legged-train --algo him_ppo --sim mujoco training.log_root=./logs/him-mujoco training.no_play=true
```

The task defaults to `go2_arm_manip_loco`. Append Hydra overrides for tuning;
`--cfg` prints the composed configuration. Select the backend with `--sim`.
For evaluation, replace the absolute example path with the run directory
containing your checkpoint; `algo.checkpoint=-1` selects its latest checkpoint.

```bash
uv run legged-eval --algo ppo --sim mujoco training.log_root=./logs/eval-ppo algo.load_run=/absolute/path/to/ppo/run algo.checkpoint=-1
uv run legged-eval --algo him_ppo --sim mujoco training.log_root=./logs/eval-him algo.load_run=/absolute/path/to/him/run algo.checkpoint=-1
```

HIM-PPO evaluation accepts `--export` with the export extra installed.
See [HIM-PPO](docs/en/2-algorithms/6-him_ppo.md) for history dimensions and
the default arm training stage.

## Tools and documentation

```bash
uv run legged-ik
uv run legged-diagnose-ik
uv run legged-calibrate --target 0.30 0.0 0.25
uv run legged-benchmark-jacobian
```

The IK viewer needs a display. Diagnosis, orientation calibration, and Jacobian
benchmarking use the bundled MuJoCo scene; calibration does not take an ONNX
policy argument. Use each command's `--help` for its options.

- [Task owners and commands](docs/en/4-tasks/4-manip_loco.md)
- [Tuning and IK checks](docs/en/8-manipulation/2-manip_loco.md)
- [Architecture and extraction record](docs/ARCHITECTURE.md)
- [Source provenance](MIGRATION_MANIFEST.json) and [license notices](NOTICE.md)
