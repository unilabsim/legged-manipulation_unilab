import numpy as np
from unilab.utils.rotation import np_quat_canonicalize, np_quat_inv, np_quat_mul


def np_quat_orientation_error_local(goal_quat: np.ndarray, curr_quat: np.ndarray) -> np.ndarray:
    """Local-frame orientation error as the signed xyz of the relative quaternion.

    Both inputs are re-normalized to unit length. Returns the imaginary
    part of ``goal * inv(curr)`` after canonicalization, giving a
    signed-axis-scaled error suitable for PD-style feedback terms.
    """
    goal = np_quat_normalize(goal_quat)
    curr = np_quat_normalize(curr_quat)
    if goal.ndim == 1:
        goal = goal[None, :]
    if curr.ndim == 1:
        curr = curr[None, :]
    rel = np_quat_mul(goal, np_quat_inv(curr))
    rel = np_quat_canonicalize(rel)
    sign = np.where(rel[:, 0:1] < 0.0, -1.0, 1.0)
    return rel[:, 1:] * sign


def np_spherical_to_cartesian(sphere: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)[l, phi, theta]`` to ``(..., 3)[x, y, z]``.

    Uses the go2-arm spherical convention: ``phi`` sweeps in the x-z plane
    from the positive x axis, and ``theta`` measures elevation toward
    positive y.
    """
    length = sphere[..., 0]
    phi = sphere[..., 1]
    theta = sphere[..., 2]
    x = length * np.cos(phi) * np.cos(theta)
    y = length * np.sin(theta)
    z = length * np.sin(phi) * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def np_cartesian_to_spherical(cart: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)[x, y, z]`` to ``(..., 3)[l, phi, theta]`` (inverse of
    :func:`np_spherical_to_cartesian`)."""
    cart = np.asarray(cart)
    l_sq = np.sum(cart**2, axis=-1, keepdims=True)
    length = np.sqrt(np.maximum(l_sq, 1e-12))
    phi = np.arctan2(cart[..., 2:3], cart[..., 0:1])
    theta = np.arcsin(np.clip(cart[..., 1:2] / length, -1.0, 1.0))
    return np.concatenate([length, phi, theta], axis=-1)


def np_quat_normalize(q: np.ndarray) -> np.ndarray:
    """L2-normalize quaternion(s), clamping tiny norms to 1e-8 to avoid divide-by-zero."""
    q = np.asarray(q)
    if q.ndim == 1:
        norm = float(np.linalg.norm(q))
        return q / max(norm, 1.0e-8)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, 1.0e-8, None)
