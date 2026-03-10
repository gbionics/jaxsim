from __future__ import annotations

import jax.numpy as jnp

import jaxsim.typing as jtp

from .utils import safe_norm


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
    default = jnp.asarray(default, dtype=jnp.result_type(numerator, denominator))

    dtype = jnp.result_type(numerator, denominator)
    eps = (
        jnp.asarray(min_denominator, dtype=dtype)
        if min_denominator is not None
        else jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    )

    safe_denominator = jnp.where(jnp.abs(denominator) > eps, denominator, 1)
    quotient = numerator / safe_denominator

    return jnp.where(jnp.abs(denominator) > eps, quotient, default)


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
    norm = safe_norm(array=array, axis=axis, keepdims=True)

    normalized = safe_divide(
        numerator=array,
        denominator=norm,
        default=jnp.zeros_like(array),
        min_denominator=min_norm,
    )

    if default is not None:
        default = jnp.asarray(default, dtype=array.dtype)
        normalized = jnp.where(norm > 0, normalized, default)

    if not keepdims:
        norm = jnp.squeeze(norm, axis=axis)

    return normalized, norm


def normalize_quaternion(quaternion: jtp.ArrayLike) -> jtp.Array:
    """
    Normalize a quaternion with an identity fallback for zero-norm inputs.

    Args:
        quaternion: The quaternion in WXYZ representation.

    Returns:
        A safely normalized quaternion.
    """

    quaternion = jnp.asarray(quaternion)
    default = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=quaternion.dtype), quaternion.shape
    )
    normalized, _ = safe_normalize(quaternion, axis=-1, default=default)
    return normalized
