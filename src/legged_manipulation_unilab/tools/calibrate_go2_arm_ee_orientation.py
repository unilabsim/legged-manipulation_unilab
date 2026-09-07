"""Sweep arm goal orientation with fixed-base MuJoCo IK, without a policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from unilab.utils.rotation import np_quat_from_euler_xyz

from legged_manipulation_unilab.tasks.geometry import np_quat_orientation_error_local
from legged_manipulation_unilab.tools import play_go2_arm_ik_only as ik

DEFAULT_TARGETS = [
    [0.15, 0.0, 0.25],
    [0.20, 0.0, 0.25],
    [0.25, 0.0, 0.25],
    [0.30, 0.0, 0.25],
    [0.25, 0.08, 0.25],
    [0.25, -0.08, 0.25],
    [0.25, 0.0, 0.15],
    [0.25, 0.0, 0.35],
]


def evaluate(ctx, target, roll, pitch_offset, args):
    """Use the task's local-frame 6D DLS convention; reset each candidate."""
    model, data = ctx["model"], ctx["data"]
    ik._reset_home(model, data, ctx["home_qpos"])
    data.ctrl[:] = ctx["home_ctrl"]
    yaw = np.arctan2(target[1], target[0])
    pitch = -np.arctan2(target[2], target[0]) + pitch_offset
    goal_quat = np_quat_from_euler_xyz(np.array([roll]), np.array([pitch]), np.array([yaw]))
    ids = ctx["arm_actuator_ids"]
    low, high = model.actuator_ctrlrange[ids].T
    pos_errors, orn_errors, margins = [], [], []
    substeps = max(1, round(args.ctrl_dt / args.sim_dt))
    for _ in range(args.steps_per_target):
        mujoco.mj_forward(model, data)
        pos_error = target - data.sensor("endpoint_pos").data
        orn_error = np_quat_orientation_error_local(
            goal_quat, data.sensor("endpoint_quat").data[None, :]
        )[0]
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ctx["ee_site_id"])
        rotation = data.site_xmat[ctx["armbase_site_id"]].reshape(3, 3).T
        jac = np.concatenate(
            [rotation @ jacp[:, ctx["arm_qvel_ids"]], rotation @ jacr[:, ctx["arm_qvel_ids"]]]
        )
        error = np.concatenate([pos_error, orn_error])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.05**2, error)
        joints = data.qpos[ctx["arm_qpos_ids"]]
        data.ctrl[ids] = np.clip(joints + np.clip(dq, -0.2, 0.2), low, high)
        pos_errors.append(float(np.linalg.norm(pos_error)))
        orn_errors.append(float(np.linalg.norm(orn_error)))
        margins.append(float(np.min(np.minimum(joints - low, high - joints))))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            ik._hold_go2_default(data, home_qpos=ctx["home_qpos"], home_ctrl=ctx["home_ctrl"])
    tail = max(10, args.steps_per_target // 4)
    return {
        "mean_pos_err": float(np.mean(pos_errors[-tail:])),
        "max_pos_err": float(np.max(pos_errors[-tail:])),
        "mean_orn_err": float(np.mean(orn_errors[-tail:])),
        "max_orn_err": float(np.max(orn_errors[-tail:])),
        "min_margin": float(np.min(margins)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--target", type=float, nargs=3, action="append")
    parser.add_argument("--roll-min", type=float, default=-np.pi)
    parser.add_argument("--roll-max", type=float, default=np.pi)
    parser.add_argument("--roll-steps", type=int, default=9)
    parser.add_argument("--pitch-min", type=float, default=0.0)
    parser.add_argument("--pitch-max", type=float, default=1.2)
    parser.add_argument("--pitch-steps", type=int, default=13)
    parser.add_argument("--steps-per-target", type=int, default=120)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--limit-margin-threshold", type=float, default=0.05)
    parser.add_argument("--pos-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args(argv)
    if (
        min(args.roll_steps, args.pitch_steps, args.steps_per_target, args.sim_dt, args.ctrl_dt)
        <= 0
    ):
        parser.error("step counts and time steps must be positive")
    targets = np.asarray(args.target or DEFAULT_TARGETS)
    ctx_args = argparse.Namespace(
        **vars(args), initial_goal=targets[0], jacobian_source="direct", compare_jacobians=False
    )
    ctx = ik._build_context(ctx_args)
    rows = []
    try:
        for roll in np.linspace(args.roll_min, args.roll_max, args.roll_steps):
            for pitch in np.linspace(args.pitch_min, args.pitch_max, args.pitch_steps):
                results = [evaluate(ctx, target, roll, pitch, args) for target in targets]
                row = {"roll": float(roll), "pitch_offset": float(pitch)}
                for key in results[0]:
                    reduction = np.mean if key.startswith("mean") else np.max
                    if key == "min_margin":
                        reduction = np.min
                    row[key] = float(reduction([result[key] for result in results]))
                row["score"] = (
                    row["mean_pos_err"]
                    + 0.25 * row["mean_orn_err"]
                    + 8 * max(0, args.limit_margin_threshold - row["min_margin"])
                    + 4 * max(0, row["mean_pos_err"] - args.pos_threshold)
                )
                row["feasible"] = all(
                    r["mean_pos_err"] <= args.pos_threshold
                    and r["min_margin"] >= args.limit_margin_threshold
                    for r in results
                )
                rows.append(row)
    finally:
        ik._close_context(ctx)
    rows.sort(key=lambda row: (not row["feasible"], row["score"]))
    for row in rows[: max(1, args.top_k)]:
        print(row)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
