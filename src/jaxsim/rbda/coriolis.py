import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Cross

from . import utils


def coriolis(
    model: js.model.JaxSimModel,
    *,
    joint_positions: jtp.VectorLike,
    joint_velocities: jtp.VectorLike,
    base_linear_velocity: jtp.VectorLike,
    base_angular_velocity: jtp.VectorLike,
) -> jtp.Matrix:
    """
    Compute the free-floating Coriolis matrix in body-fixed representation.

    Args:
        model: The model to consider.
        joint_positions: The positions of the joints.
        joint_velocities: The velocities of the joints.
        base_linear_velocity:
            The linear velocity of the base link in body-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in body-fixed representation.

    Returns:
        The free-floating Coriolis matrix in body-fixed representation.
    """

    _, _, s, B_v_WB, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
    )

    # Get the 6D spatial inertia matrices of all links.
    M = js.model.link_spatial_inertia_matrices(model=model)

    # Get the parent array λ(i).
    λ = model.kin_dyn_parameters.parent_array

    # Compute the parent-to-child adjoints of the joints.
    i_X_λi = model.kin_dyn_parameters.joint_transforms(
        joint_positions=s, base_transform=jnp.eye(4)
    )

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate the buffers.
    v = jnp.zeros(shape=(model.number_of_links(), 6))
    Ṡ = jnp.zeros_like(S)
    Mc = M
    Bc = jnp.zeros_like(M)

    # Initialize the base quantities.
    B_v_WB = B_v_WB if model.floating_base() else jnp.zeros(6)
    v = v.at[0].set(B_v_WB)
    Bc = Bc.at[0].set(Cross.vx_star(v[0]) @ M[0])

    # ====================
    # Propagate kinematics
    # ====================

    ForwardPassCarry = tuple[jtp.Matrix, jtp.Array, jtp.Array]
    forward_pass_carry: ForwardPassCarry = (v, Ṡ, Bc)

    def propagate_kinematics(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> tuple[ForwardPassCarry, None]:

        ii = i - 1
        v, Ṡ, Bc = carry

        v_i = i_X_λi[i] @ v[λ[i]] + S[i].squeeze() * ṡ[ii]
        v = v.at[i].set(v_i)

        Ṡ_i = Cross.vx(v[i]) @ S[i]
        Ṡ = Ṡ.at[i].set(Ṡ_i)

        Bc_i = Cross.vx_star(v[i]) @ M[i]
        Bc = Bc.at[i].set(Bc_i)

        return (v, Ṡ, Bc), None

    (v, Ṡ, Bc), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=forward_pass_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(v, Ṡ, Bc), None]
    )

    # =======================
    # Compute Coriolis matrix
    # =======================

    C = jnp.zeros(shape=(6 + model.dofs(), 6 + model.dofs()))

    BackwardPassCarry = tuple[jtp.Array, jtp.Array, jtp.Matrix]
    backward_pass_carry: BackwardPassCarry = (Mc, Bc, C)

    def backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> tuple[BackwardPassCarry, None]:

        ii = i - 1
        Mc, Bc, C = carry

        F1 = Mc[i] @ Ṡ[i] + Bc[i] @ S[i]
        F2 = Mc[i] @ S[i]
        F3 = Bc[i].T @ S[i]

        C_ii = S[i].T @ F1
        C = C.at[6 + ii, 6 + ii].set(C_ii.squeeze())

        Mc_λi = Mc[λ[i]] + i_X_λi[i].T @ Mc[i] @ i_X_λi[i]
        Mc = Mc.at[λ[i]].set(Mc_λi)

        Bc_λi = Bc[λ[i]] + i_X_λi[i].T @ Bc[i] @ i_X_λi[i]
        Bc = Bc.at[λ[i]].set(Bc_λi)

        j = i

        FakeWhileCarry = tuple[jtp.Int, jtp.Vector, jtp.Vector, jtp.Vector, jtp.Matrix]
        fake_while_carry: FakeWhileCarry = (j, F1, F2, F3, C)

        def fake_while_loop(
            carry: FakeWhileCarry, k: jtp.Int
        ) -> tuple[FakeWhileCarry, None]:

            def compute(carry: FakeWhileCarry) -> FakeWhileCarry:

                j, F1, F2, F3, C = carry

                F1 = i_X_λi[j].T @ F1
                F2 = i_X_λi[j].T @ F2
                F3 = i_X_λi[j].T @ F3

                j = λ[j]
                jj = j - 1

                C_ji = S[j].T @ F1
                C = C.at[6 + jj, 6 + ii].set(C_ji.squeeze())

                C_ij = Ṡ[j].T @ F2 + S[j].T @ F3
                C = C.at[6 + ii, 6 + jj].set(C_ij.squeeze())

                return j, F1, F2, F3, C

            j, _, _, _, _ = carry

            j, F1, F2, F3, C = jax.lax.cond(
                pred=jnp.logical_and(k == λ[j], λ[j] > 0),
                true_fun=compute,
                false_fun=lambda carry: carry,
                operand=carry,
            )

            return (j, F1, F2, F3, C), None

        (j, F1, F2, F3, C), _ = (
            jax.lax.scan(
                f=fake_while_loop,
                init=fake_while_carry,
                xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
            )
            if model.number_of_links() > 1
            else [(j, F1, F2, F3, C), None]
        )

        if model.floating_base():
            F1 = i_X_λi[j].T @ F1
            F2 = i_X_λi[j].T @ F2
            F3 = i_X_λi[j].T @ F3

            C = C.at[0:6, 6 + ii].set(F1.squeeze())
            C = C.at[6 + ii, 0:6].set((Cross.vx(v[0]).T @ F2 + F3).squeeze())

        return (Mc, Bc, C), None

    (Mc, Bc, C), _ = (
        jax.lax.scan(
            f=backward_pass,
            init=backward_pass_carry,
            xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
        )
        if model.number_of_links() > 1
        else [(Mc, Bc, C), None]
    )

    C_00 = Mc[0] @ Cross.vx(v[0]) + Bc[0]
    C = C.at[0:6, 0:6].set(C_00)

    return C
