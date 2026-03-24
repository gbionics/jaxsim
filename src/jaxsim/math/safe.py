"""Gradient-safe math primitives with custom JVP rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp

# ===========================
# smooth_relu with custom JVP
# ===========================


def smooth_relu(
    x: jtp.ArrayLike,
    *,
    transition_width: float = 1e-4,
) -> jtp.Array:
    """
    ReLU (max(0, x)) with an exact forward pass but a sigmoid-smoothed gradient.

    The forward output is identical to ``jnp.maximum(0, x)``. The custom JVP
    replaces the step-function sub-gradient with a sigmoid of configurable width,
    eliminating the gradient discontinuity at x = 0. This is particularly useful
    for contact penetration depth, where the hard step produces gradient jumps
    of several orders of magnitude at the contact surface.

    Args:
        x: Input array.
        transition_width:
            Width of the smooth gradient transition zone around zero (in the
            same units as *x*).  Default 1e-4 (0.1 mm for SI contact problems).

    Returns:
        ``max(0, x)`` with a smooth gradient.
    """

    if transition_width <= 0:
        raise ValueError(
            f"transition_width must be strictly positive, got {transition_width}"
        )

    return _make_smooth_relu(transition_width)(jnp.asarray(x))


def _make_smooth_relu(transition_width):

    @jax.custom_jvp
    def _smooth_relu(x):
        return jnp.maximum(0.0, x)

    @_smooth_relu.defjvp
    def _smooth_relu_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents

        primal_out = jnp.maximum(0.0, x)

        # Sigmoid-smoothed gradient: converges to step(x) as width → 0.
        beta = 1.0 / transition_width
        tangent_out = jax.nn.sigmoid(x * beta) * x_dot

        return primal_out, tangent_out

    return _smooth_relu


def safe_divide(
    numerator: jtp.ArrayLike,
    denominator: jtp.ArrayLike,
    *,
    default: jtp.ArrayLike | float = 0.0,
    min_denominator: jtp.FloatLike | None = None,
) -> jtp.Array:
    """
    Divide two arrays while guarding small denominators.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: The value returned when the denominator is too small.
        min_denominator:
            The minimum absolute denominator considered safe. If omitted, machine
            epsilon of the promoted dtype is used.

    Returns:
        The guarded quotient.
    """

    numerator = jnp.asarray(numerator)
    denominator = jnp.asarray(denominator)

    # Promote to a floating dtype so that jnp.finfo is always valid.
    dtype = jnp.result_type(numerator, denominator)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(dtype, jnp.float32)

    numerator = numerator.astype(dtype)
    denominator = denominator.astype(dtype)
    default = jnp.asarray(default, dtype=dtype)

    eps = (
        jnp.asarray(min_denominator, dtype=dtype)
        if min_denominator is not None
        else jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    )

    safe_denominator = jnp.where(
        jnp.abs(denominator) > eps, denominator, jnp.asarray(1, dtype=dtype)
    )
    quotient = numerator / safe_denominator

    return jnp.where(jnp.abs(denominator) > eps, quotient, default)


# ==============================
# safe_normalize with custom JVP
# ==============================


def safe_normalize(
    array: jtp.ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = -1,
    keepdims: bool = False,
    default: jtp.ArrayLike | None = None,
    min_norm: jtp.FloatLike | None = None,
) -> tuple[jtp.Array, jtp.Array]:
    """
    Normalize an array while guarding zero-norm inputs.

    Uses a custom JVP rule that projects the tangent vector onto the tangent
    plane of the unit sphere, avoiding gradient blowup near zero-norm inputs.

    Args:
        array: The array to normalize.
        axis: The axis or axes along which to normalize.
        keepdims: Whether to keep the reduced dimensions in the returned norm.
        default:
            Optional normalized fallback returned when the norm is too small.
            If omitted, zeros are returned.
        min_norm: Optional minimum safe norm.

    Returns:
        A tuple with the normalized array and the corresponding norm.
    """

    array = jnp.asarray(array)
    normalized, norm = _make_safe_normalize(axis, keepdims, min_norm)(array)

    if default is not None:
        default = jnp.asarray(default, dtype=array.dtype)
        norm_check = jnp.linalg.norm(array, axis=axis, keepdims=True)
        dtype = array.dtype
        eps = (
            jnp.asarray(min_norm, dtype=dtype)
            if min_norm is not None
            else jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        )
        normalized = jnp.where(norm_check > eps, normalized, default)

    return normalized, norm


def _make_safe_normalize(axis, keepdims, min_norm):

    @jax.custom_jvp
    def _safe_normalize(array):
        norm = jnp.linalg.norm(array, axis=axis, keepdims=True)
        dtype = array.dtype
        eps = (
            jnp.asarray(min_norm, dtype=dtype)
            if min_norm is not None
            else jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        )
        is_safe = norm > eps
        safe_n = jnp.where(is_safe, norm, 1.0)
        normalized = jnp.where(is_safe, array / safe_n, jnp.zeros_like(array))

        norm_out = norm if keepdims else jnp.squeeze(norm, axis=axis)
        return normalized, norm_out

    @_safe_normalize.defjvp
    def _safe_normalize_jvp(primals, tangents):
        (array,) = primals
        (array_dot,) = tangents

        normalized, norm_out = _safe_normalize(array)

        norm = jnp.linalg.norm(array, axis=axis, keepdims=True)
        dtype = array.dtype
        eps = (
            jnp.asarray(min_norm, dtype=dtype)
            if min_norm is not None
            else jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        )
        is_safe = norm > eps
        safe_n = jnp.where(is_safe, norm, 1.0)

        # Unit vector (finite even when norm ≈ 0 because safe_n = 1).
        n = array / safe_n

        # Project tangent onto the tangent plane of the unit sphere:
        #   d(x/‖x‖) = (ẋ − n⟨n, ẋ⟩) / ‖x‖
        proj = jnp.sum(n * array_dot, axis=axis, keepdims=True)
        normalized_tangent = jnp.where(
            is_safe,
            (array_dot - n * proj) / safe_n,
            jnp.zeros_like(array),
        )

        # Norm tangent: d‖x‖ = ⟨n, ẋ⟩
        norm_tangent = jnp.where(is_safe, proj, jnp.zeros_like(norm))
        norm_tangent = (
            norm_tangent if keepdims else jnp.squeeze(norm_tangent, axis=axis)
        )

        return (normalized, norm_out), (normalized_tangent, norm_tangent)

    return _safe_normalize


# ====================================
# normalize_quaternion with custom JVP
# ====================================


@jax.custom_jvp
def normalize_quaternion(quaternion: jtp.ArrayLike) -> jtp.Array:
    """
    Normalize a quaternion with an identity fallback for zero-norm inputs.

    Uses a custom JVP rule that projects the tangent onto the tangent plane
    of S³, producing zero gradients for degenerate (zero-norm) quaternions.

    Args:
        quaternion: The quaternion in WXYZ representation.

    Returns:
        A safely normalized quaternion.
    """

    quaternion = jnp.asarray(quaternion)
    norm = jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    eps = jnp.finfo(quaternion.dtype).eps
    is_safe = norm > eps
    safe_n = jnp.where(is_safe, norm, 1.0)
    result = quaternion / safe_n

    default = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=quaternion.dtype),
        quaternion.shape,
    )
    return jnp.where(is_safe, result, default)


@normalize_quaternion.defjvp
def _normalize_quaternion_jvp(primals, tangents):
    (quaternion,) = primals
    (q_dot,) = tangents

    quaternion = jnp.asarray(quaternion)
    norm = jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    eps = jnp.finfo(quaternion.dtype).eps
    is_safe = norm > eps
    safe_n = jnp.where(is_safe, norm, 1.0)

    n = quaternion / safe_n

    default = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=quaternion.dtype),
        quaternion.shape,
    )
    primal_out = jnp.where(is_safe, n, default)

    # Project tangent onto S³ tangent plane: (q̇ − n⟨n, q̇⟩) / ‖q‖
    proj = jnp.sum(n * q_dot, axis=-1, keepdims=True)
    tangent_out = jnp.where(
        is_safe,
        (q_dot - n * proj) / safe_n,
        jnp.zeros_like(quaternion),
    )

    return primal_out, tangent_out
