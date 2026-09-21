"""Map Pico headset/controller state onto chassis velocity and EE pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from unilab.utils.rotation import np_matrix_from_quat

# OpenXR / Pico Y-up to robot Z-up. Matches GR00T PicoStreamer.
R_HEADSET_TO_WORLD = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class PicoCommand:
    velocity: np.ndarray
    ee_cart: np.ndarray | None
    ee_quat: np.ndarray | None
    clutching: bool
    home_ee: bool = False
    home_started: bool = False


def apply_dead_zone(value: float, dead_zone: float) -> float:
    """Apply dead zone and renormalize to [-1, 1], matching GR00T PicoStreamer."""
    if abs(value) < dead_zone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - dead_zone) / max(1.0 - dead_zone, 1.0e-6)


def radial_square_normalize(x: float, y: float) -> tuple[float, float]:
    """Map circular stick readings onto the unit square while keeping deflection.

    A physical stick is gated by a circle, so a full diagonal is about
    ``(0.707, 0.707)``. Stretching by ``mag / max(|x|, |y|)`` turns that into
    ``(1, 1)`` without changing pure-axis full throws or soft mid-deflections.
    """
    mag = float(np.hypot(x, y))
    if mag < 1.0e-8:
        return 0.0, 0.0
    max_abs = max(abs(x), abs(y), 1.0e-8)
    scale = mag / max_abs
    return (
        float(np.clip(x * scale, -1.0, 1.0)),
        float(np.clip(y * scale, -1.0, 1.0)),
    )


def quat_wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a canonical wxyz quaternion."""
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def retarget_ee_pose(
    T_ctrl0: np.ndarray,
    T_ctrl: np.ndarray,
    T_ee0: np.ndarray,
    motion_scale: float,
) -> np.ndarray:
    """Map controller motion onto the EE in the yaw-aligned operator / arm-base frame.

    Translation uses the headset-yaw Z-up position delta, not the controller or EE
    local frames. Otherwise a pitched handle or the default ~45° EE pitch folds
    forward and up into each other. Rotation is the same operator-frame delta.
    """
    scale = float(motion_scale)
    delta_pos = scale * (T_ctrl[:3, 3] - T_ctrl0[:3, 3])
    delta_rot = T_ctrl[:3, :3] @ T_ctrl0[:3, :3].T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = delta_rot @ T_ee0[:3, :3]
    transform[:3, 3] = T_ee0[:3, 3] + delta_pos
    return transform


def pose_matrix(cart: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np_matrix_from_quat(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    transform[:3, 3] = np.asarray(cart, dtype=np.float64).reshape(3)
    return transform


def _pose7_to_xyz_quat_xyzw(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    xyz = np.zeros(3, dtype=np.float64) if value.size < 3 else value[:3].copy()
    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if value.size >= 7:
        quat_xyzw = value[3:7].copy()
    if np.allclose(quat_xyzw, 0.0):
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return xyz, quat_xyzw


def _z_up_transform(xyz: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_zup = R_HEADSET_TO_WORLD @ xyz
    rot = np_matrix_from_quat(
        np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    )
    rot_zup = R_HEADSET_TO_WORLD @ rot @ R_HEADSET_TO_WORLD.T
    return xyz_zup, rot_zup


def process_xr_pose(controller_pose: np.ndarray, headset_pose: np.ndarray) -> np.ndarray:
    """Controller pose in yaw-aligned, headset-relative, Z-up coordinates.

    Copied from GR00T ``PicoStreamer._process_xr_pose``.
    """
    ctrl_xyz, ctrl_quat = _pose7_to_xyz_quat_xyzw(controller_pose)
    head_xyz, head_quat = _pose7_to_xyz_quat_xyzw(headset_pose)
    ctrl_xyz, ctrl_rot = _z_up_transform(ctrl_xyz, ctrl_quat)
    head_xyz, head_rot = _z_up_transform(head_xyz, head_quat)
    delta_xyz = ctrl_xyz - head_xyz
    yaw = float(np.arctan2(head_rot[1, 0], head_rot[0, 0]))
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    inverse_yaw = np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = inverse_yaw @ ctrl_rot
    transform[:3, 3] = inverse_yaw @ delta_xyz
    return transform


def joystick_to_velocity(
    left_joystick: np.ndarray | list[float],
    *,
    vel_limit: np.ndarray,
    dead_zone: float,
) -> np.ndarray:
    """Scheme A: left stick Y → vx, left stick X → vyaw, no strafe.

    Diagonal throws are radially remapped to the unit square first so both axes
    can approach full scale together.
    """
    axis = np.asarray(left_joystick, dtype=np.float64).reshape(-1)
    x_axis = float(axis[0]) if axis.size > 0 else 0.0
    y_axis = float(axis[1]) if axis.size > 1 else 0.0
    x_axis, y_axis = radial_square_normalize(x_axis, y_axis)
    low = np.asarray(vel_limit[0], dtype=np.float64).reshape(3)
    high = np.asarray(vel_limit[1], dtype=np.float64).reshape(3)
    fwd = apply_dead_zone(y_axis, dead_zone)
    yaw = apply_dead_zone(-x_axis, dead_zone)
    scale_x = max(abs(float(low[0])), abs(float(high[0])))
    scale_yaw = max(abs(float(low[2])), abs(float(high[2])))
    velocity = np.array([fwd * scale_x, 0.0, yaw * scale_yaw], dtype=np.float64)
    return np.clip(velocity, low, high)


def _smoothstep(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    start = np.asarray(q0, dtype=np.float64).reshape(4)
    end = np.asarray(q1, dtype=np.float64).reshape(4)
    start = start / max(float(np.linalg.norm(start)), 1.0e-8)
    end = end / max(float(np.linalg.norm(end)), 1.0e-8)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = (1.0 - t) * start + t * end
        return out / max(float(np.linalg.norm(out)), 1.0e-8)
    theta = float(np.arccos(dot))
    sine = float(np.sin(theta))
    return (np.sin((1.0 - t) * theta) * start + np.sin(t * theta) * end) / max(sine, 1.0e-8)


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    start = np.asarray(q0, dtype=np.float64).reshape(4)
    end = np.asarray(q1, dtype=np.float64).reshape(4)
    start = start / max(float(np.linalg.norm(start)), 1.0e-8)
    end = end / max(float(np.linalg.norm(end)), 1.0e-8)
    dot = abs(float(np.clip(np.dot(start, end), -1.0, 1.0)))
    return 2.0 * float(np.arccos(dot))


class PicoMapper:
    """Relative EE teleop with right-grip clutch; left stick always drives the chassis."""

    def __init__(
        self,
        *,
        motion_scale: float = 1.0,
        dead_zone: float = 0.1,
        grip_threshold: float = 0.5,
        vel_limit: np.ndarray | list[list[float]],
        home_cart: np.ndarray | list[float] | None = None,
        home_quat: np.ndarray | list[float] | None = None,
        home_duration: float = 1.2,
        home_lin_speed: float = 0.15,
        home_ang_speed: float = 1.0,
    ) -> None:
        self.motion_scale = float(motion_scale)
        self.dead_zone = float(dead_zone)
        self.grip_threshold = float(grip_threshold)
        self.vel_limit = np.asarray(vel_limit, dtype=np.float64)
        self.home_cart = np.asarray(
            [0.30, 0.0, 0.25] if home_cart is None else home_cart, dtype=np.float64
        ).reshape(3)
        self.home_quat = np.asarray(
            [1.0, 0.0, 0.0, 0.0] if home_quat is None else home_quat, dtype=np.float64
        ).reshape(4)
        self.home_duration = float(home_duration)
        self.home_lin_speed = float(home_lin_speed)
        self.home_ang_speed = float(home_ang_speed)
        self._trigger_held = False
        self.reset_clutch()
        self.reset_homing()

    def reset_clutch(self) -> None:
        self._clutch = False
        self._T_ctrl0: np.ndarray | None = None
        self._T_ee0: np.ndarray | None = None

    def reset_homing(self) -> None:
        self._homing = False
        self._home_elapsed = 0.0
        self._home_span = 0.0
        self._home_start_cart: np.ndarray | None = None
        self._home_start_quat: np.ndarray | None = None

    def _start_homing(self, ee_cart: np.ndarray, ee_quat: np.ndarray) -> None:
        start_cart = np.asarray(ee_cart, dtype=np.float64).reshape(3)
        start_quat = np.asarray(ee_quat, dtype=np.float64).reshape(4)
        dist = float(np.linalg.norm(self.home_cart - start_cart))
        angle = _quat_angle(start_quat, self.home_quat)
        lin_time = dist / max(self.home_lin_speed, 1.0e-6)
        ang_time = angle / max(self.home_ang_speed, 1.0e-6)
        self._home_span = max(self.home_duration, lin_time, ang_time)
        self._homing = True
        self._home_elapsed = 0.0
        self._home_start_cart = start_cart.copy()
        self._home_start_quat = start_quat.copy()
        self.reset_clutch()

    def _homing_pose(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        eased = _smoothstep(alpha)
        cart = (1.0 - eased) * self._home_start_cart + eased * self.home_cart
        quat = slerp_wxyz(self._home_start_quat, self.home_quat, eased)
        return cart, quat

    def update(
        self,
        pico_data: dict,
        ee_cart: np.ndarray,
        ee_quat: np.ndarray,
        *,
        dt: float = 0.02,
    ) -> PicoCommand:
        velocity = joystick_to_velocity(
            pico_data["left_joystick"],
            vel_limit=self.vel_limit,
            dead_zone=self.dead_zone,
        )
        gripping = float(pico_data.get("right_grip", 0.0)) > self.grip_threshold
        T_ctrl = process_xr_pose(pico_data["right_pose"], pico_data["head_pose"])
        triggered = float(pico_data.get("right_trigger", 0.0)) > self.grip_threshold
        home_edge = triggered and not self._trigger_held
        self._trigger_held = triggered
        if home_edge:
            self._start_homing(ee_cart, ee_quat)

        if self._homing and self._home_start_cart is not None and self._home_start_quat is not None:
            self._home_elapsed += max(float(dt), 0.0)
            span = max(self._home_span, 1.0e-6)
            alpha = self._home_elapsed / span
            cart, quat = self._homing_pose(min(alpha, 1.0))
            finished = alpha >= 1.0
            if finished:
                self.reset_homing()
                cart = self.home_cart.copy()
                quat = self.home_quat.copy()
                if gripping:
                    self._clutch = True
                    self._T_ctrl0 = T_ctrl.copy()
                    self._T_ee0 = pose_matrix(cart, quat)
            return PicoCommand(
                velocity=velocity,
                ee_cart=cart,
                ee_quat=quat,
                clutching=self._clutch,
                home_ee=True,
                home_started=home_edge,
            )
        if gripping and not self._clutch:
            self._clutch = True
            self._T_ctrl0 = T_ctrl.copy()
            self._T_ee0 = pose_matrix(ee_cart, ee_quat)
        elif not gripping:
            self.reset_clutch()

        ee_out_cart = None
        ee_out_quat = None
        if self._clutch and self._T_ctrl0 is not None and self._T_ee0 is not None:
            T_ee = retarget_ee_pose(self._T_ctrl0, T_ctrl, self._T_ee0, self.motion_scale)
            ee_out_cart = T_ee[:3, 3].copy()
            ee_out_quat = quat_wxyz_from_matrix(T_ee[:3, :3])
        return PicoCommand(
            velocity=velocity,
            ee_cart=ee_out_cart,
            ee_quat=ee_out_quat,
            clutching=self._clutch,
        )
