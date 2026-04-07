import jax
import jax.numpy as jnp

import jaxsim.typing as jtp


def _make_safe_norm(axis, keepdims):
    @jax.custom_jvp
    def _safe_norm(array: jtp.ArrayLike) -> jtp.Array:
        """
        Compute an array norm handling NaNs and making sure that
        it is safe to get the gradient.

        Args:
            array: The array for which to compute the norm.

        Returns:
            The norm of the array with handling for zero arrays to avoid NaNs.
        """
        # Compute the norm of the array along the specified axis.
        return jnp.linalg.norm(array, axis=axis, keepdims=keepdims)

    @_safe_norm.defjvp
    def _safe_norm_jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents

        # Check if the entire array is composed of zeros.
        is_zero = jnp.allclose(x, 0)

        # Replace zeros with an array of ones temporarily to avoid division by zero.
        # This ensures the computation of norm does not produce NaNs or Infs.
        array = jnp.where(is_zero, jnp.ones_like(x), x)

        # Compute the norm of the array along the specified axis.
        norm = jnp.linalg.norm(array, axis=axis, keepdims=keepdims)

        dot = jnp.sum(array * x_dot, axis=axis, keepdims=keepdims)
        tangent = jnp.where(is_zero, 0.0, dot / norm)

        return jnp.where(is_zero, 0.0, norm), tangent

    return _safe_norm


def safe_norm(array: jtp.ArrayLike, *, axis=None, keepdims: bool = False) -> jtp.Array:
    """
    Compute an array norm handling NaNs and making sure that
    it is safe to get the gradient.

    Args:
        array: The array for which to compute the norm.
        axis: The axis for which to compute the norm.
        keepdims: Whether to keep the dimensions of the input

    Returns:
        The norm of the array with handling for zero arrays to avoid NaNs.
    """
    return _make_safe_norm(axis, keepdims)(array)
