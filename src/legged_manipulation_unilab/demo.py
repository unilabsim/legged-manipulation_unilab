"""Demo entrypoint: bundled checkpoint, then launch playback via ``legged-eval``.

Mirrors UniLab's demo registry format, but checkpoints ship inside the
repository (this repo bundles all assets offline instead of fetching from
Hugging Face). Each demo directory pairs a ``model_*.pt`` with the training
``run_config.json`` so eval restores the trained task stage.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from legged_manipulation_unilab.cli import _main


@dataclass(frozen=True)
class DemoSpec:
    algo: str
    sim: str


DEMO_REGISTRY: dict[str, DemoSpec] = {
    "maniploco-ppo": DemoSpec(algo="ppo", sim="mujoco"),
    "maniploco-him": DemoSpec(algo="him_ppo", sim="mujoco"),
}


def checkpoint_root() -> Path:
    return Path(__file__).resolve().parent / "assets" / "checkpoints"


def get_demo_spec(demo_name: str) -> DemoSpec:
    try:
        return DEMO_REGISTRY[demo_name]
    except KeyError as exc:
        available = ", ".join(sorted(DEMO_REGISTRY))
        raise SystemExit(f"Unknown demo {demo_name!r}. Available demos: {available}") from exc


def resolve_demo_checkpoint(demo_name: str) -> Path:
    demo_dir = checkpoint_root() / demo_name
    models = sorted(demo_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        raise SystemExit(
            f"Demo {demo_name!r} has no bundled checkpoint under {demo_dir}. "
            "Run `legged-train` and copy the run directory there to add a demo."
        )
    return demo_dir


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: legged-demo <name> [legged-eval overrides...]\n"
            f"Available demos: {', '.join(sorted(DEMO_REGISTRY))}"
        )
        return
    demo_name = sys.argv[1]
    spec = get_demo_spec(demo_name)
    run_dir = resolve_demo_checkpoint(demo_name)
    print(f"Demo {demo_name!r}: algo={spec.algo} sim={spec.sim} checkpoint={run_dir}")
    overrides = [f"algo.load_run={run_dir}"]
    if len(sys.argv) > 2:
        overrides += sys.argv[2:]
    _main(play=True, argv=["--algo", spec.algo, "--sim", spec.sim, *overrides])


if __name__ == "__main__":
    main()
