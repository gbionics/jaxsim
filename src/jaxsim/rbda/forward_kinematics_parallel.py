import math

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint

from . import utils


def forward_kinematics_model_parallel(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike,
    base_quaternion: jtp.VectorLike,
    joint_positions: jtp.VectorLike,
    base_linear_velocity_inertial: jtp.VectorLike,
    base_angular_velocity_inertial: jtp.VectorLike,
    joint_velocities: jtp.VectorLike,
    joint_transforms: jtp.MatrixLike,
) -> jtp.Array:
    """
    Compute forward kinematics using pointer jumping on the kinematic tree.

    Uses an associative binary operator on transform-velocity pairs to
    compute all world-frame transforms and velocities in O(log D) parallel
    steps, where D is the tree depth.

    The interface and semantics are identical to
    :func:`forward_kinematics_model`.
    """

    _, _, _, W_v_WB, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity_inertial,
        base_angular_velocity=base_angular_velocity_inertial,
        joint_velocities=joint_velocities,
    )

    # Extract the parent-to-child adjoints of the joints.
    i_X_λi = jnp.asarray(joint_transforms)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    n = model.number_of_links()

    # Compute local transforms λ(i)_X_i by inverting the child-to-parent adjoints.
    L = jax.vmap(Adjoint.inverse)(i_X_λi)  # (n, 6, 6)

    # Compute local velocity contributions.
    ṡ_padded = jnp.concatenate([jnp.zeros(1), jnp.atleast_1d(ṡ.squeeze())])  # (n,)
    vJ = (S * ṡ_padded[:, None, None]).squeeze(-1)  # (n, 6)
    u = jnp.einsum("nij,nj->ni", L, vJ)  # (n, 6)
    u = u.at[0].set(W_v_WB)

    # Get the parent array λ(i) with root self-loop.
    # Note: λ(0) is set to 0 to enable root self-referencing.
    ptr = jnp.asarray(model.kin_dyn_parameters.parent_array).at[0].set(0)
    done = jnp.arange(n) == 0

    # Number of pointer-jumping rounds.
    n_levels = model.kin_dyn_parameters.level_nodes.shape[0]
    n_rounds = max(1, math.ceil(math.log2(max(n_levels, 2))))

    # ===============
    # Pointer jumping
    # ===============

    # Each round composes the node state with its current pointer target,
    # then doubles the jump distance. After ceil(log2 D) rounds every node
    # has accumulated the full root-to-node transform and velocity.

    def _pointer_jump(carry, _):
        L, u, ptr, done = carry
        need = ~done

        L_par = L[ptr]
        u_par = u[ptr]

        # Associative compose.
        L_new = jnp.where(need[:, None, None], L_par @ L, L)
        u_new = jnp.where(
            need[:, None],
            u_par + jnp.einsum("nij,nj->ni", L_par, u),
            u,
        )

        ptr_new = jnp.where(need, ptr[ptr], ptr)
        done_new = done | done[ptr]

        return (L_new, u_new, ptr_new, done_new), None

    (W_X_i, W_v_Wi, _, _), _ = (
        jax.lax.scan(
            f=_pointer_jump,
            init=(L, u, ptr, done),
            xs=jnp.arange(n_rounds),
        )
        if n > 1
        else ((L, u, ptr, done), None)
    )

    return jax.vmap(Adjoint.to_transform)(W_X_i), W_v_Wi
