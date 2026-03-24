import functools
from typing import Literal

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp


def _get_eps(dtype: jnp.dtype) -> float:
    """Get appropriate epsilon for regularization based on dtype."""
    return jnp.finfo(dtype).eps ** 0.5


# =================================================
# Symmetric Positive-Definite Solve with Custom JVP
# =================================================


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def spd_solve(
    A: jtp.Matrix,
    b: jtp.Array,
    regularization: float | None = None,
    solver: Literal["cholesky", "lu"] = "cholesky",
) -> jtp.Array:
    """
    Solve a linear system Ax = b where A is SPD.

    This uses Cholesky decomposition by default. It includes automatic
    regularization to handle near-singular matrices in float32.

    Args:
        A: Symmetric positive-definite matrix of shape (n, n).
        b: Right-hand side vector of shape (n,) or matrix of shape (n, m).
        regularization: Optional regularization to add to diagonal. If None,
            uses dtype-appropriate default.
        solver: Solver method - "cholesky" (default) or "lu".

    Returns:
        Solution x such that Ax = b.
    """
    return _spd_solve_impl(A, b, regularization, solver)


def _spd_solve_impl(
    A: jtp.Matrix,
    b: jtp.Array,
    regularization: float | None,
    solver: Literal["cholesky", "lu"],
) -> jtp.Array:

    dtype = A.dtype

    # Apply regularization if specified or use default for float32
    if regularization is not None:
        A_reg = A + regularization * jnp.eye(A.shape[0], dtype=dtype)
    elif dtype == jnp.float32:
        # Auto-regularize for float32 to improve stability
        eps = _get_eps(dtype)
        A_reg = A + eps * jnp.trace(A) / A.shape[0] * jnp.eye(A.shape[0], dtype=dtype)
    else:
        A_reg = A

    if solver == "cholesky":

        # Cholesky decomposition: A = L @ L.T
        L = jax.scipy.linalg.cholesky(A_reg, lower=True)

        # Solve L @ y = b, then L.T @ x = y
        y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
        x = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)

        # Iterative refinement for float32 reusing L factorization
        def chol_refinement_body(carry, _):
            x_curr, L_fac, A_mat, b_vec = carry
            residual = b_vec - A_mat @ x_curr
            y = jax.scipy.linalg.solve_triangular(L_fac, residual, lower=True)
            correction = jax.scipy.linalg.solve_triangular(L_fac.T, y, lower=False)
            x_next = x_curr + correction
            return (x_next, L_fac, A_mat, b_vec), None

        if dtype == jnp.float32:
            (x, _, _, _), _ = jax.lax.scan(
                chol_refinement_body, (x, L, A_reg, b), jnp.arange(1)
            )
    else:
        # Fall back to LU decomposition
        x = jnp.linalg.solve(A_reg, b)

        # Iterative refinement for float32 using standard solve
        def lu_refinement_body(carry, _):
            x_curr, A_mat, b_vec = carry
            residual = b_vec - A_mat @ x_curr
            correction = jnp.linalg.solve(A_mat, residual)
            x_next = x_curr + correction
            return (x_next, A_mat, b_vec), None

        if dtype == jnp.float32:
            (x, _, _), _ = jax.lax.scan(
                lu_refinement_body, (x, A_reg, b), jnp.arange(1)
            )

    return x


@spd_solve.defjvp
def _spd_solve_jvp(regularization, solver, primals, tangents):
    """
    Use custom JVP for SPD solve using implicit differentiation.

    For Ax = b, the derivatives are:
        dx = A^{-1} @ (db - dA @ x)

    This is more efficient than the default JVP because we reuse the
    factorization from the forward pass.
    """
    A, b = primals
    dA, db = tangents

    # Forward pass
    x = _spd_solve_impl(A, b, regularization, solver)

    # Compute tangent using implicit differentiation
    # dx = A^{-1} @ (db - dA @ x)
    rhs = db - dA @ x
    dx = _spd_solve_impl(A, rhs, regularization, solver)

    return x, dx


# =================================================
# Standard Solve with Regularization and Custom JVP
# =================================================


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def standard_solve(
    A: jtp.Matrix,
    b: jtp.Array,
    regularization: float | None = None,
) -> jtp.Array:
    """
    Solve a linear system Ax = b with optional regularization.

    This is a general-purpose solver that handles non-SPD matrices.
    For SPD matrices, use `spd_solve` instead for better performance.

    Args:
        A: Square matrix of shape (n, n).
        b: Right-hand side vector of shape (n,) or matrix of shape (n, m).
        regularization: Optional regularization to add to diagonal.

    Returns:
        Solution x such that Ax = b.
    """
    return _standard_solve_impl(A, b, regularization)


def _standard_solve_impl(
    A: jtp.Matrix,
    b: jtp.Array,
    regularization: float | None,
) -> jtp.Array:
    """Use iterative refinement for float32 of standard solve."""
    dtype = A.dtype

    if regularization is not None:
        A_reg = A + regularization * jnp.eye(A.shape[0], dtype=dtype)
    elif dtype == jnp.float32:
        # Auto-regularize for float32
        eps = _get_eps(dtype)
        A_reg = A + eps * jnp.eye(A.shape[0], dtype=dtype)
    else:
        A_reg = A

    # Initial solve
    x = jnp.linalg.solve(A_reg, b)

    # Iterative refinement for float32 using JIT-compatible loop
    def refinement_step(i, x_carry):
        residual = b - A_reg @ x_carry
        correction = jnp.linalg.solve(A_reg, residual)
        return x_carry + correction

    # Apply refinement iteration for float32
    x = jax.lax.cond(
        dtype == jnp.float32,
        lambda x: jax.lax.fori_loop(0, 1, refinement_step, x),
        lambda x: x,
        x,
    )

    return x


@standard_solve.defjvp
def _standard_solve_jvp(regularization, primals, tangents):
    """Use custom JVP for standard solve using implicit differentiation."""
    A, b = primals
    dA, db = tangents

    x = _standard_solve_impl(A, b, regularization)
    rhs = db - dA @ x
    dx = _standard_solve_impl(A, rhs, regularization)

    return x, dx


# ===================================
# Safe Matrix Inverse with Custom JVP
# ===================================


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def safe_inv(
    A: jtp.Matrix,
    regularization: float | None = None,
) -> jtp.Matrix:
    """
    Compute the inverse of a matrix with optional regularization.

    Args:
        A: Square matrix of shape (n, n).
        regularization: Optional regularization to add to diagonal.

    Returns:
        Inverse of A.
    """
    return _safe_inv_impl(A, regularization)


def _safe_inv_impl(
    A: jtp.Matrix,
    regularization: float | None,
) -> jtp.Matrix:

    dtype = A.dtype

    if regularization is not None:
        A = A + regularization * jnp.eye(A.shape[0], dtype=dtype)
    elif dtype == jnp.float32:
        eps = _get_eps(dtype)
        A = A + eps * jnp.eye(A.shape[0], dtype=dtype)

    return jnp.linalg.inv(A)


@safe_inv.defjvp
def _safe_inv_jvp(regularization, primals, tangents):
    """
    Use custom JVP for matrix inverse.

    For A^{-1}, the derivative is:
        d(A^{-1}) = -A^{-1} @ dA @ A^{-1}
    """
    (A,) = primals
    (dA,) = tangents

    A_inv = _safe_inv_impl(A, regularization)
    dA_inv = -A_inv @ dA @ A_inv

    return A_inv, dA_inv


# ==================================
# Safe Least Squares with Custom JVP
# ==================================


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def safe_lstsq(
    A: jtp.Matrix,
    b: jtp.Array,
    rcond: float | None = None,
) -> jtp.Array:
    """
    Solve a least-squares problem min ||Ax - b||_2.

    This provides a numerically stable least-squares solver with a custom JVP
    for efficient differentiation.

    Args:
        A: Matrix of shape (m, n).
        b: Right-hand side vector of shape (m,) or matrix of shape (m, k).
        rcond: Cut-off ratio for small singular values. If None, uses
            dtype-appropriate default.

    Returns:
        Least-squares solution x.
    """
    return _safe_lstsq_impl(A, b, rcond)


def _safe_lstsq_impl(
    A: jtp.Matrix,
    b: jtp.Array,
    rcond: float | None,
) -> jtp.Array:
    dtype = A.dtype

    if rcond is None:
        rcond = _get_eps(dtype) * max(A.shape)

    x, _, _, _ = jnp.linalg.lstsq(A, b, rcond=rcond)
    return x


@safe_lstsq.defjvp
def _safe_lstsq_jvp(rcond, primals, tangents):
    """
    Use pseudo-inverse based differentiation on custom JVP for least squares.
    """
    A, b = primals
    dA, db = tangents

    x = _safe_lstsq_impl(A, b, rcond)

    # Compute pseudo-inverse of A
    dtype = A.dtype
    actual_rcond = rcond if rcond is not None else _get_eps(dtype) * max(A.shape)
    A_pinv = jnp.linalg.pinv(A, rcond=actual_rcond)

    dx = A_pinv @ (db - dA @ x)

    return x, dx
