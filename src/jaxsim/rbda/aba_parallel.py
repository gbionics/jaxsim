import math

import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import STANDARD_GRAVITY, Adjoint, Cross

from . import utils


def aba_parallel(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
    base_linear_velocity: jtp.VectorLike,
    base_angular_velocity: jtp.VectorLike,
    joint_velocities: jtp.VectorLike,
    joint_transforms: jtp.MatrixLike,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    standard_gravity: jtp.FloatLike = STANDARD_GRAVITY,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute forward dynamics using a hybrid parallel ABA.

    Passes 1 and 3 use pointer jumping in O(log D) parallel steps.
    Pass 2 uses level-parallel processing in O(D) steps because the backward
    inertia accumulation is not associative.

    The interface and semantics are identical to :func:`aba`.
    """

    W_p_B, W_Q_B, s, W_v_WB, ṡ, _, _, τ, W_f, W_g = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        joint_velocities=joint_velocities,
        base_linear_acceleration=None,
        base_angular_acceleration=None,
        joint_accelerations=None,
        joint_forces=joint_forces,
        link_forces=link_forces,
        standard_gravity=standard_gravity,
    )

    W_g = jnp.atleast_2d(W_g).T
    W_v_WB = jnp.atleast_2d(W_v_WB).T

    # Get the 6D spatial inertia matrices of all links.
    M = js.model.link_spatial_inertia_matrices(model=model)

    # Get the parent array λ(i).
    λ = model.kin_dyn_parameters.parent_array

    # Get the tree level structure for level-parallel processing.
    level_nodes = jnp.asarray(model.kin_dyn_parameters.level_nodes)
    level_mask = jnp.asarray(model.kin_dyn_parameters.level_mask)
    n_levels = level_nodes.shape[0]
    max_width = level_nodes.shape[1]

    # Compute the base transform.
    W_H_B = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=W_Q_B),
        translation=W_p_B,
    )

    # Compute 6D transforms of the base velocity.
    W_X_B = W_H_B.adjoint()
    B_X_W = W_H_B.inverse().adjoint()

    # Extract the parent-to-child adjoints of the joints.
    i_X_λi = jnp.asarray(joint_transforms)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    n = model.number_of_links()

    # Parent array with root self-loop.
    # Note: λ(0) is set to 0 to enable root self-referencing.
    ptr0 = jnp.asarray(λ).at[0].set(0)

    # Number of pointer-jumping rounds.
    n_rounds = max(1, math.ceil(math.log2(max(n_levels, 2))))

    # ======
    # Pass 1
    # ======

    # Two coupled affine recurrences propagated via pointer jumping:
    #   v_i = i_X_λi[i] @ v_parent + vJ_i
    #   T_i = i_X_λi[i] @ T_parent
    #
    # Associative operator on (A, b, T):
    #   compose(parent, child) = (A @ A_p, A @ b_p + b, A @ T_p)

    # Local transforms and joint velocities.
    ṡ_col = jnp.atleast_1d(ṡ).reshape(-1, 1)  # (n_joints, 1)
    ṡ_padded = jnp.concatenate([jnp.zeros((1, 1)), ṡ_col])  # (n, 1)
    vJ = S * ṡ_padded[:, :, None]  # (n, 6, 1)

    # Initialize pointer-jumping state for each node.
    A = i_X_λi.copy()  # (n, 6, 6)
    b = vJ.copy()  # (n, 6, 1)
    T = i_X_λi.copy()  # (n, 6, 6)

    # Root initial values.
    if model.floating_base():
        v_0 = B_X_W @ W_v_WB
        A = A.at[0].set(jnp.eye(6))
        b = b.at[0].set(v_0)
        T = T.at[0].set(jnp.eye(6))
    else:
        A = A.at[0].set(jnp.eye(6))
        b = b.at[0].set(jnp.zeros((6, 1)))
        T = T.at[0].set(jnp.eye(6))

    ptr = ptr0.copy()
    done = jnp.arange(n) == 0

    def _pass1_jump(carry, _):
        A, b, T, ptr, done = carry
        need = ~done

        A_par = A[ptr]
        b_par = b[ptr]
        T_par = T[ptr]

        # Associative compose.
        A_new = jnp.where(need[:, None, None], A @ A_par, A)
        b_new = jnp.where(need[:, None, None], A @ b_par + b, b)
        T_new = jnp.where(need[:, None, None], A @ T_par, T)

        ptr_new = jnp.where(need, ptr[ptr], ptr)
        done_new = done | done[ptr]

        return (A_new, b_new, T_new, ptr_new, done_new), None

    (_, v, i_X_0, _, _), _ = (
        jax.lax.scan(
            f=_pass1_jump,
            init=(A, b, T, ptr, done),
            xs=jnp.arange(n_rounds),
        )
        if n > 1
        else ((A, b, T, ptr, done), None)
    )

    # v now contains the 6D body velocity of every link.
    # i_X_0 contains the body-to-base transform for every link.

    # Compute c, MA, pA for all nodes in parallel.
    def _init_node(node_i):
        ii = node_i - 1
        vJ_i = S[node_i] * ṡ_padded[node_i]
        c_i = Cross.vx(v[node_i]) @ vJ_i
        MA_i = M[node_i]
        i_Xf_W = Adjoint.inverse(i_X_0[node_i] @ B_X_W).T
        pA_i = Cross.vx_star(v[node_i]) @ M[node_i] @ v[node_i] - i_Xf_W @ jnp.vstack(
            W_f[node_i]
        )
        return c_i, MA_i, pA_i

    c, MA, pA = jax.vmap(_init_node)(jnp.arange(n))

    # Override base MA and pA if floating base.
    if model.floating_base():
        MA = MA.at[0].set(M[0])
        pA_0 = Cross.vx_star(v[0]) @ M[0] @ v[0] - W_X_B.T @ jnp.vstack(W_f[0])
        pA = pA.at[0].set(pA_0)

    # ======
    # Pass 2
    # ======

    # The Schur complement and multi-child scatter-add make this pass
    # non-associative, so it remains level-parallel.

    U = jnp.zeros_like(S)
    d = jnp.ones(shape=(n, 1))  # Ones to avoid NaN for the base node.
    u = jnp.zeros(shape=(n, 1))

    def _masked_scatter_add(arr, indices, values, m):
        """Add values[j] to arr[indices[j]] only where m[j] is True."""
        for j in range(max_width):
            arr = jnp.where(m[j], arr.at[indices[j]].add(values[j]), arr)
        return arr

    def _pass2_level(carry, level_idx):
        U, d, u, MA, pA = carry
        actual_level = n_levels - 1 - level_idx
        nodes = level_nodes[actual_level]
        mask = level_mask[actual_level]

        def _process_node_pass2(node_i):
            ii = node_i - 1
            parent = λ[node_i]

            U_i = MA[node_i] @ S[node_i]
            d_i = (S[node_i].T @ U_i).squeeze()
            u_i = (τ[ii] - S[node_i].T @ pA[node_i]).squeeze()

            Ma_i = MA[node_i] - U_i / d_i @ U_i.T
            pa_i = pA[node_i] + Ma_i @ c[node_i] + U_i * (u_i / d_i)

            Ma_parent = i_X_λi[node_i].T @ Ma_i @ i_X_λi[node_i]
            pa_parent = i_X_λi[node_i].T @ pa_i

            return U_i, d_i, u_i, Ma_parent, pa_parent, parent

        U_lev, d_lev, u_lev, Ma_par, pa_par, parents = jax.vmap(_process_node_pass2)(
            nodes
        )

        mask_6x1 = mask[:, None, None]
        mask_1 = mask[:, None]

        U = carry[0].at[nodes].set(jnp.where(mask_6x1, U_lev, carry[0][nodes]))
        d = carry[1].at[nodes].set(jnp.where(mask_1, d_lev[:, None], carry[1][nodes]))
        u = carry[2].at[nodes].set(jnp.where(mask_1, u_lev[:, None], carry[2][nodes]))

        should_propagate = jnp.where(
            model.floating_base(),
            mask,
            jnp.logical_and(mask, parents != 0),
        )

        MA = _masked_scatter_add(carry[3], parents, Ma_par, should_propagate)
        pA = _masked_scatter_add(carry[4], parents, pa_par, should_propagate)

        return (U, d, u, MA, pA), None

    n_backward_levels = n_levels - 1
    (U, d, u, MA, pA), _ = (
        jax.lax.scan(
            f=_pass2_level,
            init=(U, d, u, MA, pA),
            xs=jnp.arange(n_backward_levels),
        )
        if n_backward_levels > 0
        else ((U, d, u, MA, pA), None)
    )

    # ======
    # Pass 3
    # ======

    # The acceleration recurrence is an affine recurrence:
    #   a_i = P_i @ i_X_λi[i] @ a_parent + P_i @ c_i + S_i * u_i / d_i
    # where P_i = I - S_i @ U_i^T / d_i is the 6x6 projection matrix.

    if model.floating_base():
        a0 = jnp.linalg.solve(-MA[0], pA[0])
    else:
        a0 = -B_X_W @ W_g

    # Pre-compute the affine recurrence coefficients for all nodes.
    def _init_pass3(node_i):
        P_i = jnp.eye(6) - S[node_i] @ U[node_i].T / d[node_i]
        A_i = P_i @ i_X_λi[node_i]
        b_i = P_i @ c[node_i] + S[node_i] * (u[node_i] / d[node_i])
        return A_i, b_i

    A, b = jax.vmap(_init_pass3)(jnp.arange(n))

    # Root acceleration is known.
    A = A.at[0].set(jnp.eye(6))
    b = b.at[0].set(a0)

    # Pointer jumping for the affine recurrence.
    ptr = ptr0.copy()
    done = jnp.arange(n) == 0

    def _pass3_jump(carry, _):
        A, b, ptr, done = carry
        need = ~done

        A_par = A[ptr]
        b_par = b[ptr]

        # Associative compose.
        A_new = jnp.where(need[:, None, None], A @ A_par, A)
        b_new = jnp.where(need[:, None, None], A @ b_par + b, b)

        ptr_new = jnp.where(need, ptr[ptr], ptr)
        done_new = done | done[ptr]

        return (A_new, b_new, ptr_new, done_new), None

    (_, a, _, _), _ = (
        jax.lax.scan(
            f=_pass3_jump,
            init=(A, b, ptr, done),
            xs=jnp.arange(n_rounds),
        )
        if n > 1
        else ((A, b, ptr, done), None)
    )

    # Recover joint accelerations: s̈_i = (u_i - U_i^T @ a_before_i) / d_i
    # where a_before_i = i_X_λi[i] @ a_parent + c_i.
    a_λi = a[ptr0]
    a_before = i_X_λi @ a_λi + c
    Ut_a = (U.transpose(0, 2, 1) @ a_before).squeeze(-1)  # (n, 1)
    s̈ = (u - Ut_a) / d  # (n, 1)

    # ==============
    # Adjust outputs
    # ==============

    if model.floating_base():
        B_a_WB = a[0]
        W_a_WB = W_X_B @ B_a_WB + W_g
    else:
        W_a_WB = jnp.zeros(6)

    # Joint accelerations: skip base index, take indices 1..n-1.
    s̈_out = s̈[1:]

    return W_a_WB.squeeze(), jnp.atleast_1d(s̈_out.squeeze())
