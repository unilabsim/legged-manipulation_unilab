"""IK unit tests for the task-owned Go2 Arm helper."""

from __future__ import annotations

import numpy as np

from legged_manipulation_unilab.tasks.go2_arm.base import IKConfig, compute_arm_ik_delta


class _FakeBackend:
    def __init__(self, jacp: np.ndarray, jacr: np.ndarray):
        self._jacp = jacp
        self._jacr = jacr

    def get_site_ids(self, names):
        del names
        return np.asarray([0], dtype=np.int32)

    def get_joint_dof_indices(self, names):
        return np.arange(len(names), dtype=np.int32)

    def get_site_jacobian_w(self, site_id: int, dof_indices: np.ndarray):
        del site_id, dof_indices
        return self._jacp, self._jacr

    def get_sensor_data(self, name: str) -> np.ndarray:
        del name
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)


def _ik_helper(use_orientation: bool, orientation_mode: str):
    jacp = np.asarray(
        [
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    jacr = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    cfg = IKConfig(use_orientation=use_orientation, orientation_mode=orientation_mode)
    cfg.damping = 0.0
    cfg.dq_clip = 0.0
    return lambda goal, curr: compute_arm_ik_delta(
        _FakeBackend(jacp, jacr),
        cfg,
        ee_site_name="endpoint",
        arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        arm_ref_world_quat_sensor="armbasepoint_world_quat",
        goal_local_pos=goal,
        curr_local_pos=curr,
    )


def test_go2_arm_ik_zero_error_orientation_regularizes_rotation_nullspace():
    goal = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    curr = np.zeros((1, 3), dtype=np.float64)

    position_only = _ik_helper(False, "target")(goal, curr)
    zero_error = _ik_helper(True, "zero_error")(goal, curr)

    np.testing.assert_allclose(position_only, [[0.5, 1.0, 1.5, 0.5, 1.0, 1.5]])
    np.testing.assert_allclose(zero_error, [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])


def test_go2_arm_ik_rejects_unknown_orientation_mode():
    helper = _ik_helper(True, "invalid")
    goal = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    curr = np.zeros((1, 3), dtype=np.float64)
    try:
        helper(goal, curr)
    except ValueError as exc:
        assert "ik.orientation_mode" in str(exc)
    else:
        raise AssertionError("expected invalid ik.orientation_mode to raise ValueError")
