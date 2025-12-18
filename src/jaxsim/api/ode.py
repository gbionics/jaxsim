import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Quaternion, Skew
from jaxsim.rbda.kinematic_constraints import compute_constraint_wrenches

from .common import VelRepr

# ==================================
# Functions defining system dynamics
# ==================================


@functools.partial(jax.jit, static_argnames=["output_representation"])
def system_acceleration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_torques: jtp.VectorLike | None = None,
    output_representation: VelRepr | None = None,
) -> tuple[jtp.Vector, jtp.Vector, dict[str, jtp.PyTree]]:
    """
    Compute the system acceleration in the active representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the same
            velocity representation of data.
        joint_torques: The joint torques applied to the joints.
        output_representation:
            The desired output velocity representation. If None, the current
            velocity representation of `data` is used.

    Returns:
        A tuple containing the base 6D acceleration in the active representation,
        the joint accelerations, and the contact state.
    """

    output_representation = (
        output_representation
        if output_representation is not None
        else data.velocity_representation
    )

    # ====================
    # Validate input data
    # ====================

    # Build link forces if not provided.
    f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # ======================
    # Compute contact forces
    # ======================

    W_f_L_terrain = jnp.zeros_like(f_L)
    contact_state = data.contact_state

    if len(model.kin_dyn_parameters.contact_parameters.body) > 0:

        # Compute the 6D forces W_f ∈ ℝ^{n_L × 6} applied to links due to contact
        # with the terrain.
        W_f_L_terrain, contact_state_derivative = js.contact.link_contact_forces(
            model=model,
            data=data,
            link_forces=f_L,
            joint_torques=joint_torques,
        )

        # Update the contact state data. This is necessary only for the contact models
        # that require propagation and integration of contact state.
        contact_state = model.contact_model.update_contact_state(
            contact_state_derivative
        )

    # ==================================
    # Compute kinematic constraint forces
    # ==================================

    # Sum up all the forces: external + contact
    W_f_L_total = f_L + W_f_L_terrain

    # Compute the 6D forces W_f ∈ ℝ^{n_constraints × 2 × 6} applied to links due to
    # kinematic constraints.
    W_f_L_constraints = compute_constraint_wrenches(
        model=model,
        data=data,
        link_forces_inertial=W_f_L_total,
        joint_force_references=joint_torques,
    )

    # Apply constraint forces to the corresponding links
    if W_f_L_constraints.shape[0] > 0:
        # Get the constraint map from the model's kinematic parameters
        constraint_map = model.kin_dyn_parameters.constraints

        if constraint_map is not None:
            # Stack the parent link indices for both sides of each constraint
            parent_indices_flat = jnp.concatenate(
                [constraint_map.parent_link_idxs_1, constraint_map.parent_link_idxs_2],
            )

            # Flatten the constraint wrenches to match the flattened parent indices
            constraint_wrenches_flat = W_f_L_constraints.reshape(-1, 6)

            # Apply constraint wrenches using scatter_add for better performance
            W_f_L_total = W_f_L_total.at[parent_indices_flat].add(
                constraint_wrenches_flat
            )

    # Store the link forces in a references object.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        velocity_representation=output_representation,
        link_forces=W_f_L_total,
    )

    # Compute forward dynamics.
    #
    # - Joint accelerations: s̈ ∈ ℝⁿ
    # - Base acceleration: v̇_WB ∈ ℝ⁶
    #
    # Note that ABA returns the base acceleration in the velocity representation
    # stored in the `data` object.
    v̇_WB, s̈ = js.model.forward_dynamics_aba(
        model=model,
        data=data,
        joint_forces=joint_torques,
        link_forces=references.link_forces(model=model, data=data),
        output_representation=output_representation,
    )

    return v̇_WB, s̈, contact_state


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def system_position_dynamics(
    data: js.data.JaxSimModelData,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
    *,
    output_representation: VelRepr | None = None,
) -> tuple[jtp.Vector, jtp.Vector, jtp.Vector]:
    r"""
    Compute the dynamics of the system position.

    Args:
        data: The data of the considered model.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient for adjusting the quaternion norm.
        output_representation:
            The desired output velocity representation. If None, the current
            velocity representation of `data` is used.

    Returns:
        A tuple containing the derivative of the base position, the derivative of the
        base quaternion, and the derivative of the joint positions.

    Note:
        In inertial-fixed representation, the linear component of the base velocity is not
        the derivative of the base position. In fact, the base velocity is defined as:
        :math:`{} ^W v_{W, B} = \begin{bmatrix} {} ^W \dot{p}_B S({} ^W \omega_{W, B}) {} ^W p _B\\ {} ^W \omega_{W, B} \end{bmatrix}`.
        Where :math:`S(\cdot)` is the skew-symmetric matrix operator.
    """

    output_representation = (
        output_representation
        if output_representation is not None
        else data.velocity_representation
    )

    ṡ = data.joint_velocities
    W_Q_B = data.base_orientation
    W_ω_WB = data.base_velocity(output_representation)[3:6]
    W_ṗ_B = (
        data.base_velocity(output_representation)[0:3]
        + Skew.wedge(W_ω_WB) @ data.base_position
    )

    W_Q̇_B = Quaternion.derivative(
        quaternion=W_Q_B,
        omega=W_ω_WB,
        omega_in_body_fixed=False,
        K=baumgarte_quaternion_regularization,
    ).squeeze()

    return W_ṗ_B, W_Q̇_B, ṡ


@jax.jit
@js.common.named_scope
def system_dynamics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.Vector | None = None,
    joint_torques: jtp.Vector | None = None,
    output_representation: VelRepr | None = None,
    baumgarte_quaternion_regularization: jtp.FloatLike = 1.0,
) -> dict[str, jtp.Vector]:
    """
    Compute the dynamics of the system.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D forces to apply to the links expressed in the frame corresponding to
            the velocity representation of `data`.
        joint_torques: The joint torques acting on the joints.
        output_representation:
            The desired output velocity representation. If None, the current
            velocity representation of `data` is used.
        baumgarte_quaternion_regularization:
            The Baumgarte regularization coefficient used to adjust the norm of the
            quaternion (only used in integrators not operating on the SO(3) manifold).

    Returns:
        A dictionary containing the derivatives of the base position, the base quaternion,
        the joint positions, the base linear velocity, the base angular velocity, and the
        joint velocities.
    """

    output_representation = (
        output_representation
        if output_representation is not None
        else data.velocity_representation
    )

    W_v̇_WB, s̈, contact_state_derivative = system_acceleration(
        model=model,
        data=data,
        joint_torques=joint_torques,
        link_forces=link_forces,
        output_representation=VelRepr.Inertial,
    )

    W_ṗ_B, W_Q̇_B, ṡ = system_position_dynamics(
        data=data,
        baumgarte_quaternion_regularization=baumgarte_quaternion_regularization,
        output_representation=VelRepr.Inertial,
    )

    return dict(
        base_position=W_ṗ_B,
        base_quaternion=W_Q̇_B,
        joint_positions=ṡ,
        base_linear_velocity=W_v̇_WB[0:3],
        base_angular_velocity=W_v̇_WB[3:6],
        joint_velocities=s̈,
        contact_state=contact_state_derivative,
    )
