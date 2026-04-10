import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Adjoint

from . import utils


def forward_kinematics_model(
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
    Compute the forward kinematics.

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.
        base_linear_velocity_inertial: The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity_inertial: The angular velocity of the base link in inertial-fixed representation.
        joint_velocities: The velocities of the joints.
        joint_transforms: The parent-to-child transforms of the joints.

    Returns:
        A 3D array containing the SE(3) transforms of all links belonging to the model.
    """

    _, _, _, W_v_WB, ṡ, _, _, _, _, _ = utils.process_inputs(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        base_linear_velocity=base_linear_velocity_inertial,
        base_angular_velocity=base_angular_velocity_inertial,
        joint_velocities=joint_velocities,
    )

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Extract the parent-to-child adjoints of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = jnp.asarray(joint_transforms)

    # Allocate the buffer of transforms world -> link and initialize the base pose.
    W_X_i = jnp.zeros(shape=(model.number_of_links(), 6, 6))
    W_X_i = W_X_i.at[0].set(Adjoint.inverse(i_X_λi[0]))

    # Allocate buffer of 6D inertial-fixed velocities and initialize the base velocity.
    W_v_Wi = jnp.zeros(shape=(model.number_of_links(), 6))
    W_v_Wi = W_v_Wi.at[0].set(W_v_WB)

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # ========================
    # Propagate the kinematics
    # ========================

    PropagateKinematicsCarry = tuple[jtp.Matrix, jtp.Matrix]
    propagate_kinematics_carry: PropagateKinematicsCarry = (W_X_i, W_v_Wi)

    def propagate_kinematics(
        carry: PropagateKinematicsCarry, i: jtp.Int
    ) -> tuple[PropagateKinematicsCarry, None]:

        ii = i - 1
        W_X_i, W_v_Wi = carry

        # Compute the parent to child 6D transform.
        λi_X_i = Adjoint.inverse(adjoint=i_X_λi[i])

        # Compute the world to child 6D transform.
        W_Xi_i = W_X_i[λ[i]] @ λi_X_i
        W_X_i = W_X_i.at[i].set(W_Xi_i)

        # Propagate the 6D velocity.
        W_vi_Wi = W_v_Wi[λ[i]] + W_X_i[i] @ (S[i] * ṡ[ii]).squeeze()
        W_v_Wi = W_v_Wi.at[i].set(W_vi_Wi)

        return (W_X_i, W_v_Wi), None

    (W_X_i, W_v_Wi), _ = (
        jax.lax.scan(
            f=propagate_kinematics,
            init=propagate_kinematics_carry,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(W_X_i, W_v_Wi), None]
    )

    return jax.vmap(Adjoint.to_transform)(W_X_i), W_v_Wi
