import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import STANDARD_GRAVITY, Adjoint, Cross

from . import utils


def aba(
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
    Compute forward dynamics using the Articulated Body Algorithm (ABA).

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.
        base_linear_velocity:
            The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in inertial-fixed representation.
        joint_velocities: The velocities of the joints.
        joint_transforms: The parent-to-child transforms of the joints.
        joint_forces: The forces applied to the joints.
        link_forces:
            The forces applied to the links expressed in the world frame.
        standard_gravity: The standard gravity constant.

    Returns:
        A tuple containing the base acceleration in inertial-fixed representation
        and the joint accelerations that result from the applications of the given
        joint and link forces.

    Note:
        The algorithm expects a quaternion with unit norm.
    """

    W_p_B, W_Q_B, s, W_v_WB, ṡ, _, _, τ, W_f, W_g = utils.process_inputs(
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
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute the base transform.
    W_H_B = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3(wxyz=W_Q_B),
        translation=W_p_B,
    )

    # Compute 6D transforms of the base velocity.
    W_X_B = W_H_B.adjoint()
    B_X_W = W_H_B.inverse().adjoint()

    # Extract the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = jnp.asarray(joint_transforms)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate buffers.
    v = jnp.zeros(shape=(model.number_of_links(), 6, 1))
    c = jnp.zeros(shape=(model.number_of_links(), 6, 1))
    pA = jnp.zeros(shape=(model.number_of_links(), 6, 1))
    MA = jnp.zeros(shape=(model.number_of_links(), 6, 6))

    # Allocate the buffer of transforms link -> base.
    i_X_0 = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Initialize base quantities.
    if model.floating_base():

        # Base velocity v₀ in body-fixed representation.
        v_0 = B_X_W @ W_v_WB
        v = v.at[0].set(v_0)

        # Initialize the articulated-body inertia (Mᴬ) of base link.
        MA_0 = M[0]
        MA = MA.at[0].set(MA_0)

        # Initialize the articulated-body bias force (pᴬ) of the base link.
        pA_0 = Cross.vx_star(v[0]) @ MA[0] @ v[0] - W_X_B.T @ jnp.vstack(W_f[0])
        pA = pA.at[0].set(pA_0)

    # ======
    # Pass 1
    # ======

    Pass1Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]
    pass_1_carry: Pass1Carry = (v, c, MA, pA, i_X_0)

    # Propagate kinematics and initialize AB inertia and AB bias forces.
    def loop_body_pass1(carry: Pass1Carry, i: jtp.Int) -> tuple[Pass1Carry, None]:

        ii = i - 1
        v, c, MA, pA, i_X_0 = carry

        # Project the joint velocity into its motion subspace.
        vJ = S[i] * ṡ[ii]

        # Propagate the link velocity.
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        c_i = Cross.vx(v[i]) @ vJ
        c = c.at[i].set(c_i)

        # Initialize the articulated-body inertia.
        MA_i = jnp.array(M[i])
        MA = MA.at[i].set(MA_i)

        # Compute the link-to-base transform.
        i_Xi_0 = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_Xi_0)

        # Compute link-to-world transform for the 6D force.
        i_Xf_W = Adjoint.inverse(i_X_0[i] @ B_X_W).T

        # Initialize articulated-body bias force.
        pA_i = Cross.vx_star(v[i]) @ M[i] @ v[i] - i_Xf_W @ jnp.vstack(W_f[i])
        pA = pA.at[i].set(pA_i)

        return (v, c, MA, pA, i_X_0), None

    (v, c, MA, pA, i_X_0), _ = (
        jax.lax.scan(
            f=loop_body_pass1,
            init=pass_1_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(v, c, MA, pA, i_X_0), None]
    )

    # ======
    # Pass 2
    # ======

    U = jnp.zeros_like(S)
    d = jnp.zeros(shape=(model.number_of_links(), 1))
    u = jnp.zeros(shape=(model.number_of_links(), 1))

    Pass2Carry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]
    pass_2_carry: Pass2Carry = (U, d, u, MA, pA)

    def loop_body_pass2(carry: Pass2Carry, i: jtp.Int) -> tuple[Pass2Carry, None]:

        ii = i - 1
        U, d, u, MA, pA = carry

        U_i = MA[i] @ S[i]
        U = U.at[i].set(U_i)

        d_i = S[i].T @ U[i]
        d = d.at[i].set(d_i.squeeze())

        u_i = τ[ii] - S[i].T @ pA[i]
        u = u.at[i].set(u_i.squeeze())

        # Compute the articulated-body inertia and bias force of this link.
        Ma = MA[i] - U[i] / d[i] @ U[i].T
        pa = pA[i] + Ma @ c[i] + U[i] * (u[i] / d[i])

        # Propagate them to the parent, handling the base link.
        def propagate(
            MA_pA: tuple[jtp.Matrix, jtp.Matrix],
        ) -> tuple[jtp.Matrix, jtp.Matrix]:

            MA, pA = MA_pA

            MA_λi = MA[λ[i]] + i_X_λi[i].T @ Ma @ i_X_λi[i]
            MA = MA.at[λ[i]].set(MA_λi)

            pA_λi = pA[λ[i]] + i_X_λi[i].T @ pa
            pA = pA.at[λ[i]].set(pA_λi)

            return MA, pA

        MA, pA = jax.lax.cond(
            pred=jnp.logical_or(λ[i] != 0, model.floating_base()),
            true_fun=propagate,
            false_fun=lambda MA_pA: MA_pA,
            operand=(MA, pA),
        )

        return (U, d, u, MA, pA), None

    (U, d, u, MA, pA), _ = (
        jax.lax.scan(
            f=loop_body_pass2,
            init=pass_2_carry,
            xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
        )
        if model.number_of_links() > 1
        else [(U, d, u, MA, pA), None]
    )

    # ======
    # Pass 3
    # ======

    if model.floating_base():
        a0 = jnp.linalg.solve(-MA[0], pA[0])
    else:
        a0 = -B_X_W @ W_g

    s̈ = jnp.zeros_like(s)
    a = jnp.zeros_like(v).at[0].set(a0)

    Pass3Carry = tuple[jtp.Matrix, jtp.Vector]
    pass_3_carry = (a, s̈)

    def loop_body_pass3(carry: Pass3Carry, i: jtp.Int) -> tuple[Pass3Carry, None]:

        ii = i - 1
        a, s̈ = carry

        # Propagate the link acceleration.
        a_i = i_X_λi[i] @ a[λ[i]] + c[i]

        # Compute the joint acceleration.
        s̈_ii = (u[i] - U[i].T @ a_i) / d[i]
        s̈ = s̈.at[ii].set(s̈_ii.squeeze())

        # Sum the joint acceleration to the parent link acceleration.
        a_i = a_i + S[i] * s̈[ii]
        a = a.at[i].set(a_i)

        return (a, s̈), None

    (a, s̈), _ = (
        jax.lax.scan(
            f=loop_body_pass3,
            init=pass_3_carry,
            xs=jnp.arange(1, model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(a, s̈), None]
    )

    # ==============
    # Adjust outputs
    # ==============

    # TODO: remove vstack and shape=(6, 1)?
    if model.floating_base():
        # Convert the base acceleration to inertial-fixed representation,
        # and add gravity.
        B_a_WB = a[0]
        W_a_WB = W_X_B @ B_a_WB + W_g
    else:
        W_a_WB = jnp.zeros(6)

    return W_a_WB.squeeze(), jnp.atleast_1d(s̈.squeeze())
