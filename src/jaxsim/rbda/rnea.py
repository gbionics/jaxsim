import jax
import jax.numpy as jnp
import jaxlie

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import STANDARD_GRAVITY, Adjoint, Cross

from . import utils


def rnea(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.Vector,
    base_quaternion: jtp.Vector,
    joint_positions: jtp.Vector,
    base_linear_velocity: jtp.Vector,
    base_angular_velocity: jtp.Vector,
    joint_velocities: jtp.Vector,
    base_linear_acceleration: jtp.Vector | None = None,
    base_angular_acceleration: jtp.Vector | None = None,
    joint_accelerations: jtp.Vector | None = None,
    joint_transforms: jtp.Matrix,
    link_forces: jtp.Matrix | None = None,
    standard_gravity: jtp.FloatLike = STANDARD_GRAVITY,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute inverse dynamics using the Recursive Newton-Euler Algorithm (RNEA).

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
        base_linear_acceleration:
            The linear acceleration of the base link in inertial-fixed representation.
        base_angular_acceleration:
            The angular acceleration of the base link in inertial-fixed representation.
        joint_accelerations: The accelerations of the joints.
        joint_transforms: The parent-to-child transforms of the joints.
        link_forces:
            The forces applied to the links expressed in the world frame.
        standard_gravity: The standard gravity constant.

    Returns:
        A tuple containing the 6D force applied to the base link expressed in the
        world frame and the joint forces that, when applied respectively to the base
        link and joints, produce the given base and joint accelerations.
    """

    W_p_B, W_Q_B, s, W_v_WB, ṡ, W_v̇_WB, s̈, _, W_f, W_g = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        joint_velocities=joint_velocities,
        base_linear_acceleration=base_linear_acceleration,
        base_angular_acceleration=base_angular_acceleration,
        joint_accelerations=joint_accelerations,
        link_forces=link_forces,
        standard_gravity=standard_gravity,
    )

    W_g = jnp.atleast_2d(W_g).T
    W_v_WB = jnp.atleast_2d(W_v_WB).T
    W_v̇_WB = jnp.atleast_2d(W_v̇_WB).T

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
    # Ensure cached transforms are JAX arrays so they work with traced indices.
    i_X_λi = jnp.asarray(joint_transforms)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate buffers.
    v = jnp.zeros(shape=(model.number_of_links(), 6, 1))
    a = jnp.zeros(shape=(model.number_of_links(), 6, 1))
    f = jnp.zeros(shape=(model.number_of_links(), 6, 1))

    # Allocate the buffer of transforms link -> base.
    i_X_0 = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    i_X_0 = i_X_0.at[0].set(jnp.eye(6))

    # Initialize the acceleration of the base link.
    a_0 = -B_X_W @ W_g
    a = a.at[0].set(a_0)

    if model.floating_base():

        # Base velocity v₀ in body-fixed representation.
        v_0 = B_X_W @ W_v_WB
        v = v.at[0].set(v_0)

        # Base acceleration a₀ in body-fixed representation w/o gravity.
        a_0 = B_X_W @ (W_v̇_WB - W_g)
        a = a.at[0].set(a_0)

        # Force applied to the base link that produce the base acceleration w/o gravity.
        f_0 = (
            M[0] @ a[0]
            + Cross.vx_star(v[0]) @ M[0] @ v[0]
            - W_X_B.T @ jnp.vstack(W_f[0])
        )
        f = f.at[0].set(f_0)

    # ======
    # Pass 1
    # ======

    ForwardPassCarry = tuple[jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]
    forward_pass_carry: ForwardPassCarry = (v, a, i_X_0, f)

    def forward_pass(
        carry: ForwardPassCarry, i: jtp.Int
    ) -> tuple[ForwardPassCarry, None]:

        ii = i - 1
        v, a, i_X_0, f = carry

        # Project the joint velocity into its motion subspace.
        vJ = S[i] * ṡ[ii]

        # Propagate the link velocity.
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        # Propagate the link acceleration.
        a_i = i_X_λi[i] @ a[λ[i]] + S[i] * s̈[ii] + Cross.vx(v[i]) @ vJ
        a = a.at[i].set(a_i)

        # Compute the link-to-base transform.
        i_X_0_i = i_X_λi[i] @ i_X_0[λ[i]]
        i_X_0 = i_X_0.at[i].set(i_X_0_i)

        # Compute link-to-world transform for the 6D force.
        i_Xf_W = Adjoint.inverse(i_X_0[i] @ B_X_W).T

        # Compute the force acting on the link.
        f_i = (
            M[i] @ a[i]
            + Cross.vx_star(v[i]) @ M[i] @ v[i]
            - i_Xf_W @ jnp.vstack(W_f[i])
        )
        f = f.at[i].set(f_i)

        return (v, a, i_X_0, f), None

    (v, a, i_X_0, f), _ = (
        jax.lax.scan(
            f=forward_pass,
            init=forward_pass_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(v, a, i_X_0, f), None]
    )

    # ======
    # Pass 2
    # ======

    τ = jnp.zeros_like(s)

    BackwardPassCarry = tuple[jtp.Vector, jtp.Matrix]
    backward_pass_carry: BackwardPassCarry = (τ, f)

    def backward_pass(
        carry: BackwardPassCarry, i: jtp.Int
    ) -> tuple[BackwardPassCarry, None]:

        ii = i - 1
        τ, f = carry

        # Project the 6D force to the DoF of the joint.
        τ_i = S[i].T @ f[i]
        τ = τ.at[ii].set(τ_i.squeeze())

        # Propagate the force to the parent link.
        def update_f(f: jtp.Matrix) -> jtp.Matrix:

            f_λi = f[λ[i]] + i_X_λi[i].T @ f[i]
            f = f.at[λ[i]].set(f_λi)

            return f

        f = jax.lax.cond(
            pred=jnp.logical_or(λ[i] != 0, model.floating_base()),
            true_fun=update_f,
            false_fun=lambda f: f,
            operand=f,
        )

        return (τ, f), None

    (τ, f), _ = (
        jax.lax.scan(
            f=backward_pass,
            init=backward_pass_carry,
            xs=jnp.flip(jnp.arange(start=1, stop=model.number_of_links())),
        )
        if model.number_of_links() > 1
        else [(τ, f), None]
    )

    # ==============
    # Adjust outputs
    # ==============

    # Express the base 6D force in the world frame.
    W_f0 = B_X_W.T @ f[0]

    return W_f0.squeeze(), jnp.atleast_1d(τ.squeeze())
