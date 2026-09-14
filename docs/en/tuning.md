# Tuning guide

Start from one built-in recipe, change a small number of values, and keep the
composed configuration with each experiment. Use `legged-train --cfg` before a
long run to confirm which YAML values are active.

## Useful controls

| Goal | Configuration | Notes |
| --- | --- | --- |
| Training length | `algo.max_iterations` | Built-in recipes use 3000 updates. |
| Arm residual magnitude | `env.control_config.arm_action_scale` | Zero keeps the arm on the IK solution. |
| End-effector goals | `env.goal_ee` | Sampling range, trajectory time, hold time, and collision limits. |
| Arm stage | `env.arm_stage` | Freezes joints or disables goal trajectories for staged training. |
| Base tracking trade-off | `reward.tracking_lin_vel`, `reward.tracking_ang_vel`, `reward.stand_still` | Changes gait and standing behavior. |
| Training difficulty | `env.domain_rand` | Controls mass, friction, armature, COM, PD, and push randomization. |
| Observation history | `algo.num_actor_history`, `algo.num_critic_history` | Compatibility dimensions; see HIM-PPO before changing. |

Reward terms are Manager-Based term entries. Preserve each term's `func`,
`weight`, and `params.name` when changing its weight.

## Suggested workflow

1. Run a short smoke configuration with fewer than 20 updates.
2. Restore the intended update count.
3. Change one behavior group at a time.
4. Keep the same seed when comparing a single change.
5. Evaluate from the same checkpoint-selection rule.
6. Record the full composed configuration or the exact Hydra overrides.

Randomization and reward changes can shift episode length and termination rates.
Inspect playback, not only the scalar reward.

## Validation after changes

Focused checks:

```bash
uv run pytest tests/tasks tests/algos tests/test_entrypoints.py
uv run legged-assets
```

Full suite:

```bash
uv run pytest
```

After asset or XML changes, verify that MuJoCo can construct the packaged scene:

```bash
uv run python -c "import mujoco; from legged_manipulation_unilab.assets import resolve_scene; m = mujoco.MjModel.from_xml_path(resolve_scene()); print(m.nq, m.nv, m.nu, m.nsensor)"
```

For IK or Jacobian changes, also run [IK diagnostics](ik-diagnostics.md).
