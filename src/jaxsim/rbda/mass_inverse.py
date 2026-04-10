import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


def mass_inverse(
    model: js.model.JaxSimModel,
    *,
    joint_transforms: jtp.MatrixLike,
) -> jtp.Matrix:
    """
    Compute the inverse of the mass matrix using an ABA-like algorithm.
    The implementation follows the approach described in https://laas.hal.science/hal-01790934v2.

    Args:
        model: The model to consider.
        joint_transforms: The parent-to-child transforms of the joints.

    Returns:
        The inverse of the mass matrix.
    """

    # Get the 6D spatial inertia matrices of all links.
    I_A = js.model.link_spatial_inertia_matrices(model=model)

    # Get the parent array λ(i).
    #   λ[0] ~ -1 (world)
    #   λ[i] = parent link index for link i.
    λ = model.kin_dyn_parameters.parent_array

    # Extract the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = jnp.asarray(joint_transforms)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    NB = model.number_of_links()
    N = model.number_of_joints()

    # Total generalized velocities: 6 base + N.
    nv = N + 6

    # Allocate buffers.
    F = jnp.zeros((NB, 6, nv), dtype=float)
    P = jnp.zeros((NB, 6, nv), dtype=float)
    U = jnp.zeros((NB, 6), dtype=float)
    D = jnp.zeros((NB,), dtype=float)

    # Pre-allocate mass matrix inverse
    M_inv = jnp.zeros((nv, nv), dtype=float)

    # Pre-compute indices.
    idx_fwd = jnp.arange(1, NB)
    idx_rev = jnp.arange(NB - 1, 0, -1)

    # =============
    # Backward Pass
    # =============

    BackwardPassCarry = tuple[
        jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix
    ]
    backward_pass_carry: BackwardPassCarry = (I_A, F, U, D, M_inv)

    def loop_backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> tuple[BackwardPassCarry, None]:
        I_A, F, U, D, M_inv = carry

        Si = jnp.squeeze(S[i], axis=-1)
        Fi = F[i]
        Xi = i_X_λi[i]
        parent = λ[i]

        Ui = I_A[i] @ Si
        Di = jnp.dot(Si, Ui)

        U = U.at[i].set(Ui)
        D = D.at[i].set(Di)

        # Row index in ν for joint i: 6 + (i - 1)
        r = 6 + (i - 1)

        Minv_row = M_inv[r]

        # Diagonal element
        Minv_row = Minv_row.at[r].add(1.0 / Di)

        # Off-diagonals: Minv[r,:] -= (1/Di) * Sᵢᵀ Fᵢ
        sTFi = jnp.einsum("s,sn->n", Si, Fi)
        Minv_row = Minv_row - sTFi / Di

        M_inv = M_inv.at[r].set(Minv_row)

        # Propagate to parent if any (parent >= 0)
        def propagate(IA_F):
            I_A_, F_ = IA_F

            Ui_col = Ui[:, None]

            # F_a_i = F_i + U_i * Minv[r,:]
            Fa_i = Fi + Ui_col @ Minv_row[None, :]

            # F_parent += Xᵢᵀ F_a_i
            F_parent_new = F_[parent] + Xi.T @ Fa_i
            F_ = F_.at[parent].set(F_parent_new)

            # I_a_i = IAi - U_i D_i^{-1} U_iᵀ
            Ia_i = I_A[i] - jnp.outer(Ui, Ui) / Di

            # I_A[parent] += Xᵢᵀ I_a_i Xᵢ
            I_parent_new = I_A_[parent] + Xi.T @ Ia_i @ Xi
            I_A_ = I_A_.at[parent].set(I_parent_new)

            return I_A_, F_

        I_A, F = jax.lax.cond(
            parent >= 0,
            propagate,
            lambda IA_F: IA_F,
            (I_A, F),
        )

        return (I_A, F, U, D, M_inv), None

    (I_A, F, U, D, M_inv), _ = jax.lax.scan(
        loop_backward_pass, backward_pass_carry, idx_rev
    )

    S0 = jnp.eye(6, dtype=float)
    U0 = I_A[0] @ S0
    D0 = S0.T @ U0
    D0_inv = jnp.linalg.inv(D0)

    # Base rows 0..5 in ν
    base_rows = slice(0, 6)

    # Diagonal base block
    M_inv = M_inv.at[base_rows, base_rows].add(D0_inv)

    # Off-diagonal base contribution: M_inv[base,:] -= D0^{-T} F[0]
    term0 = D0_inv.T @ F[0]
    M_inv = M_inv.at[base_rows, :].add(-term0)

    # ============
    # Forward Pass
    # ============

    # Initialize P_0 = S0 * Minv[base,:] = I * Minv[base,:]
    Minv_base = M_inv[base_rows, :]
    P = P.at[0].set(Minv_base)

    ForwardPassCarry = tuple[jtp.Matrix, jtp.Matrix]
    forward_pass_carry: ForwardPassCarry = (M_inv, P)

    def loop_forward_pass(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> tuple[ForwardPassCarry, None]:
        M_inv, P = carry

        Si = jnp.squeeze(S[i], axis=-1)
        Ui = U[i]
        Di = D[i]
        Xi = i_X_λi[i]
        parent = λ[i]

        P_parent = jax.lax.cond(
            parent >= 0,
            lambda P_: P_[parent],
            lambda P_: jnp.zeros_like(P_[i]),
            P,
        )

        # Row index in ν for joint i
        r = 6 + (i - 1)

        # Row update: M_inv[r,:] -= D_i^{-1} U_iᵀ Xᵢ P_parent
        def update_row(Minv_):
            X_P = Xi @ P_parent
            UiT_XP = jnp.einsum("s,sn->n", Ui, X_P)
            Minv_row = Minv_[r, :] - UiT_XP / Di
            return Minv_.at[r, :].set(Minv_row)

        M_inv = jax.lax.cond(
            parent >= 0,
            update_row,
            lambda Minv_: Minv_,
            M_inv,
        )

        Minv_row = M_inv[r, :]

        # P_i = S_i Minv[r,:] + Xᵢ P_parent
        Pi = jnp.expand_dims(Si, 1) @ jnp.expand_dims(Minv_row, 0)
        Pi = Pi + Xi @ P_parent

        P = P.at[i].set(Pi)

        return (M_inv, P), None

    (M_inv, P), _ = jax.lax.scan(loop_forward_pass, forward_pass_carry, idx_fwd)

    # Symmetrize numerically
    M_inv = 0.5 * (M_inv + M_inv.T)

    return M_inv
