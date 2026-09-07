# Extraction validation — UniLab #1528

Executed on Linux x86_64, Python 3.13.14, Torch 2.8.0+cu128 (CPU learning),
MuJoCo 3.11.0 / mujoco-uni-runtime 0.5.0, MotrixSim 0.8.2, on 2026-09-07.
This records migration checks, not a convergence benchmark or a new platform claim.

## Coordinated source revisions

| Repository | Implemented commit | Child PR |
| --- | --- | --- |
| UniLab | `132c7056dcba3796955d54e568072099b8c9bd39` | [#1536](https://github.com/Motphys/UniLab/pull/1536) |
| unilab_rl | `e7a6c3c01798ec55254ca77d63d283d9d3394b4a` | [#14](https://github.com/unilabsim/unilab_rl/pull/14) |
| legged-manipulation_unilab | `c2891f6bb3bd4bcd5caf717cc9c5445f50c909fe` | [#1](https://github.com/unilabsim/legged-manipulation_unilab/pull/1) |

The two dependencies remain version **1.1.0**. The destination's wheel metadata
and lockfile use exact Git revisions, so installation does not depend on a
release with a new version number. No tag, GitHub release or PyPI publication
was performed. Child PRs were integrated into their roadmap development branches;
the final PRs to `main` are the review boundary for the coordinated cutover.

## Local gates

| Check | Result |
| --- | --- |
| UniLab `UV_NO_SYNC=1 make test-all` | Format/type checks passed; 1589 passed, 20 skipped, 884 deselected, 1 xfailed; 72% coverage |
| UniLab additional modified CLI/config/Jacobian/chunk-size tests with `-m ''` | 242 passed |
| UniLab benchmark smoke (part of test-all) | Module mode 35 passed, script mode 36 passed; each skipped the optional MLX case |
| UniLab docs tests / Sphinx HTML | 20 passed / build succeeded without warnings |
| unilab_rl ruff/mypy/pyright | Passed |
| unilab_rl `uv run --no-sync pytest --cov=src/uni_rl` | 380 passed, 8 skipped, 3 deselected; 65% coverage |
| New package `UV_NO_SYNC=1 make test-all` | Ruff/mypy/pyright passed; 33 tests passed |
| New package tests after installing the wheel | 33 passed |
| All three `uv build` invocations | Wheel and sdist built successfully |

Native MuJoCo diagnostic tools are checked by execution rather than pyright:
the installed MuJoCo stubs omit generated C API exports. Core task/algorithm/
asset/entrypoint code is type checked. Existing Gymnasium float-bound and Torch
ONNX deprecation warnings were observed. UniLab's optional Drake import produced
one pyright warning. No check was marked successful by suppressing a failed test.

## Behavior equivalence

The pre-extraction source was archived from UniLab
`d9afc9656e63c82a64263f5a4dbd938ba1652af2`. In separate processes, the original
and migrated environments used identical owner configuration, the same bundled
scene, NumPy seed 2026, two environments, and 12 actions drawn from
`default_rng(4).uniform(-0.2, 0.2, (2, 18))` per step.

Each combination compared **74 arrays**: reset actor/critic observations,
per-step observations, rewards, terminated/truncated flags and EE goals.
`numpy.testing.assert_array_equal` passed for every array for:

- PPO / MuJoCo;
- PPO / Motrix;
- HIM-PPO / MuJoCo.

The replay inputs and process script are in [validation/compare_extraction.py](validation/compare_extraction.py).
Run the script once with `--baseline-source` and once without it, then compare
the two NPZ files. The baseline directory must contain `src/unilab` from the
commit above; only the baseline process uses that source path.

## Training, checkpoints and playback

Each owner completed a real one-iteration rollout/update with two environments,
four rollout steps and CPU learning:

```bash
uv run legged-train --algo ppo --sim mujoco algo.num_envs=2 algo.max_iterations=1 algo.num_steps_per_env=4 algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 training.device=cpu training.no_play=true training.log_root=/tmp/1528-runs/ppo
uv run legged-train --algo ppo --sim motrix algo.num_envs=2 algo.max_iterations=1 algo.num_steps_per_env=4 algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 training.device=cpu training.no_play=true training.log_root=/tmp/1528-runs/motrix
uv run legged-train --algo him_ppo --sim mujoco algo.num_envs=2 algo.max_iterations=1 algo.num_steps_per_env=4 algo.algorithm.num_mini_batches=1 algo.algorithm.num_learning_epochs=1 training.device=cpu training.no_play=true training.log_root=/tmp/1528-runs/him
```

PPO names its first saved checkpoint `model_0.pt`; HIM names it `model_1.pt`.
HIM resumed the generated checkpoint and saved `model_2.pt`. Its tests additionally
verify estimator optimizer restoration and loading the legacy checkpoint shape
without that optional optimizer key.

```bash
MUJOCO_GL=egl uv run legged-eval --algo ppo --sim mujoco --export algo.load_run=/absolute/path/to/model_0.pt training.device=cpu training.play_env_num=2 training.play_steps=3
MUJOCO_GL=egl uv run legged-eval --algo ppo --sim motrix algo.load_run=/absolute/path/to/model_0.pt training.device=cpu training.play_env_num=2 training.play_steps=3
MUJOCO_GL=egl uv run legged-eval --algo him_ppo --sim mujoco --export algo.load_run=/absolute/path/to/model_1.pt training.device=cpu training.play_env_num=2 training.play_steps=3
```

PPO/MuJoCo and HIM/MuJoCo produced playback MP4s and ONNX/JIT exports.
Motrix completed its native playback path. The test suite compares HIM ONNX/JIT
outputs with the Torch policy at batch sizes 1 and 3 (`rtol=1e-5`, `atol=1e-6`),
checks checkpoint dimensions before environment construction, and checks
observation-history mismatch at runner construction.

## Installed assets and entrypoints

The wheel and both Git-pinned dependencies were installed as non-editable
packages. A separate process started in `/tmp`, verified all three modules came
from `site-packages`, set `HF_HUB_OFFLINE=1`, replaced socket connection with a
function that raises, and used a new empty `LEGGED_MANIPULATION_ASSET_CACHE`.
Scene loading, reset and step passed with actor observations `(2, 76)` and finite
rewards. The offline probe is [validation/offline_install.py](validation/offline_install.py).
The regular tests also repair a same-size corrupted mesh and register the task
in a real spawn subprocess.

Executed tool checks:

```bash
uv run legged-benchmark-jacobian --cfg job
uv run legged-diagnose-ik --steps 3 --disable-gain-randomization
uv run legged-ik --headless-steps 3 --jacobian-source direct
uv run legged-calibrate --roll-steps 1 --pitch-steps 1 --steps-per-target 3 --target 0.25 0.08 0.25 --csv-out /tmp/1528-calibration.csv
```

The calibration command now computes 6D local-frame IK directly; it no longer
references the missing ONNX sim2sim script. Short diagnostic sweeps do not prove
closed-loop convergence or select production controller gains.

## Source-tree cleanup

Both source trees were audited with `git ls-files` and case-insensitive content
searches for task/algorithm names, aliases and removed dotted/slash module paths.
The only remaining named references are historical changelog entries and the
new migration notice. No dedicated source/config/asset/registration/tool/test
or current usage page remains in either source tree. Git history is preserved.
