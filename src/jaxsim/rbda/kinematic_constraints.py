from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.common import ModelDataWithVelocityRepresentation, VelRepr
from jaxsim.api.kin_dyn_parameters import ConstraintMap
from jaxsim.math.adjoint import Adjoint
from jaxsim.math.rotation import Rotation
from jaxsim.math.transform import Transform

# Utility functions used for constraints computation. These functions duplicate part of the jaxsim.api.frame module for computational efficiency.
# TODO: remove these functions when jaxsim.api.frame is optimized for batched computations.
# See: https://github.com/ami-iit/jaxsim/issues/451


def _compute_constraint_transforms_batched(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraints: ConstraintMap,
) -> jtp.Matrix:
    """
    Compute the transformation matrices for kinematic constraints between pairs of frames.

    Args:
        model: The JaxSim model containing the robot description.
        data: The model data containing current state information.
        constraints: The constraint map containing frame indices and parent link information.

    Returns:
        A matrix with shape (n_constraints, 2, 4, 4) containing the transformation matrices
        for each constraint pair. The second dimension contains [W_H_F1, W_H_F2] where
        W_H_F1 and W_H_F2 are the world-to-frame transformation matrices.
    """
    W_H_L = data._link_transforms

    frame_idxs_1 = constraints.frame_idxs_1
    frame_idxs_2 = constraints.frame_idxs_2

    parent_link_idxs_1 = constraints.parent_link_idxs_1
    parent_link_idxs_2 = constraints.parent_link_idxs_2

    # Extract frame transforms
    L_H_F1 = model.kin_dyn_parameters.frame_parameters.transform[
        frame_idxs_1 - model.number_of_links()
    ]
    L_H_F2 = model.kin_dyn_parameters.frame_parameters.transform[
        frame_idxs_2 - model.number_of_links()
    ]

    # Compute the homogeneous transformation matrices for the two frames
    W_H_F1 = W_H_L[parent_link_idxs_1] @ L_H_F1
    W_H_F2 = W_H_L[parent_link_idxs_2] @ L_H_F2

    return jnp.stack([W_H_F1, W_H_F2], axis=1)


def _compute_constraint_jacobians_batched(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    constraints: ConstraintMap,
    W_H_constraint_pairs: jtp.Matrix,
) -> jtp.Matrix:
    """
    Compute the constraint Jacobian matrices for kinematic constraints in a batched manner.
    Args:
        model: The JaxSim model containing the robot description.
        data: The model data containing current state information.
        constraints: The constraint map containing frame indices and parent link information.
        W_H_constraint_pairs: Transformation matrices for constraint frame pairs with shape
                              (n_constraints, 2, 4, 4).

    Returns:
        A matrix with shape (n_constraints, 6, n_dofs) containing the constraint Jacobian
        matrices.
    """

    # Doubly-left free-floating Jacobian.
    L_J_WL_B = js.model.generalized_free_floating_jacobian(
        model=model,
        data=data,
        input_representation=VelRepr.Body,
        output_vel_repr=VelRepr.Body,
    )

    # Link transforms
    W_H_L = data._link_transforms

    def compute_frame_jacobian_mixed(L_J_WL, W_H_L, W_H_F, parent_link_index):
        """Compute the jacobian of a frame in mixed representation."""
        # Select the jacobian of the parent link
        L_J_WL = L_J_WL[parent_link_index]

        # Compute the jacobian of the frame in mixed representation
        W_H_L = W_H_L[parent_link_index]
        F_H_L = Transform.inverse(W_H_F) @ W_H_L
        FW_H_F = W_H_F.at[0:3, 3].set(jnp.zeros(3))
        FW_H_L = FW_H_F @ F_H_L
        FW_X_L = Adjoint.from_transform(transform=FW_H_L)
        FW_J_WL = FW_X_L @ L_J_WL
        O_J_WL_I = FW_J_WL

        return O_J_WL_I

    def compute_constraint_jacobian(L_J_WL, W_H_F, constraint):
        """Compute the constraint jacobian for a single constraint pair."""

        J_WF1 = compute_frame_jacobian_mixed(
            L_J_WL, W_H_L, W_H_F[0], constraint.parent_link_idxs_1
        )
        J_WF2 = compute_frame_jacobian_mixed(
            L_J_WL, W_H_L, W_H_F[1], constraint.parent_link_idxs_2
        )

        return J_WF1 - J_WF2

    # Vectorize the computation of constraint Jacobians
    constraint_jacobians = jax.vmap(compute_constraint_jacobian, in_axes=(None, 0, 0))(
        L_J_WL_B, W_H_constraint_pairs, constraints
    )

    return constraint_jacobians


def _compute_constraint_baumgarte_term(
    J_constr: jtp.Matrix,
    nu: jtp.Vector,
    W_H_F_constr: jtp.Matrix,
    constraint: ConstraintMap,
) -> jtp.Vector:
    """
    Compute the Baumgarte stabilization term for kinematic constraints.

    The Baumgarte stabilization method is used to stabilize constraint violations
    by adding proportional and derivative terms to the constraint equation. This
    helps prevent constraint drift and improves numerical stability.

    Args:
        J_constr: The constraint Jacobian matrix with shape (6, n_dofs).
        nu: The generalized velocity vector with shape (n_dofs,).
        W_H_F_constr: Array containing the homogeneous transformation matrices
                      of two frames [W_H_F1, W_H_F2] with respect to the world frame,
                      with shape (2, 4, 4).
        constraint: The constraint object containing stabilization gains K_P and K_D.

    Returns:
        The computed Baumgarte stabilization term.
    """
    W_H_F1, W_H_F2 = W_H_F_constr

    W_p_F1 = W_H_F1[0:3, 3]
    W_p_F2 = W_H_F2[0:3, 3]

    W_R_F1 = W_H_F1[0:3, 0:3]
    W_R_F2 = W_H_F2[0:3, 0:3]

    K_P = constraint.K_P
    K_D = constraint.K_D

    vel_error = J_constr @ nu
    position_error = W_p_F1 - W_p_F2
    R_error = W_R_F2.T @ W_R_F1
    orientation_error = Rotation.log_vee(R_error)

    baumgarte_term = (
        K_P * jnp.concatenate([position_error, orientation_error]) + K_D * vel_error
    )

    return baumgarte_term


@jax.jit
@js.common.named_scope
def compute_constraint_wrenches(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_force_references: jtp.VectorLike | None = None,
    link_forces_inertial: jtp.MatrixLike | None = None,
    regularization: jtp.Float = 1e-3,
) -> jtp.Matrix:
    """
    Compute the constraint wrenches for kinematic constraints.

    This function solves the constraint forces needed to satisfy kinematic constraints
    between pairs of frames. It uses the Baumgarte stabilization method and computes
    the constraint wrenches in inertial representation.

    Args:
        model: The JaxSim model.
        data: The model data.
        joint_force_references: Optional joint force/torque references to apply. If None,
                               zero forces are used.
        link_forces_inertial: Optional link forces applied in inertial representation.
                             If None, zero forces are used.
        regularization: Regularization parameter for the constraint solver to improve
                       numerical stability. Default is 1e-3.

    Returns:
        Array with shape (n_constraints, 2, 6) containing constraint wrench pairs
        in inertial representation. Each constraint produces two equal and opposite
        wrenches applied to the constrained frames.
    """

    # Retrieve the kinematic constraints, if any.
    kin_constraints = model.kin_dyn_parameters.constraints

    n_kin_constraints = (
        6 * kin_constraints.frame_idxs_1.shape[0]
        if kin_constraints is not None and kin_constraints.frame_idxs_1.shape[0] > 0
        else 0
    )

    # Return empty results if no constraints exist
    if n_kin_constraints == 0:
        return jnp.zeros((0, 2, 6))

    # Build joint forces if not provided
    τ_references = (
        jnp.asarray(joint_force_references, dtype=float)
        if joint_force_references is not None
        else jnp.zeros_like(data.joint_positions)
    )

    # Build link forces if not provided
    W_f_L = (
        jnp.atleast_2d(jnp.array(link_forces_inertial).squeeze())
        if link_forces_inertial is not None
        else jnp.zeros((model.number_of_links(), 6))
    ).astype(float)

    # Create references object for handling different velocity representations
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=τ_references,
        link_forces=W_f_L,
        velocity_representation=VelRepr.Inertial,
    )

    BW_ν = data.generalized_velocity(VelRepr.Mixed)

    # Compute free acceleration without constraints
    BW_ν̇_free = jnp.hstack(
        js.model.forward_dynamics_aba(
            model=model,
            data=data,
            link_forces=references.link_forces(model=model, data=data),
            joint_forces=references.joint_force_references(model=model),
            output_representation=VelRepr.Mixed,
        )
    )

    # Compute mass matrix
    M_inv = js.model.free_floating_mass_matrix_inverse(
        model=model, data=data, output_representation=VelRepr.Mixed
    )

    W_H_constr_pairs = _compute_constraint_transforms_batched(
        model=model,
        data=data,
        constraints=kin_constraints,
    )

    # Compute constraint jacobians
    J_constr = _compute_constraint_jacobians_batched(
        model=model,
        data=data,
        constraints=kin_constraints,
        W_H_constraint_pairs=W_H_constr_pairs,
    )

    # Compute Baumgarte stabilization term
    constr_baumgarte_term = jnp.ravel(
        jax.vmap(
            _compute_constraint_baumgarte_term,
            in_axes=(0, None, 0, 0),
        )(
            J_constr,
            BW_ν,
            W_H_constr_pairs,
            kin_constraints,
        ),
    )

    # Stack constraint jacobians
    J_constr = jnp.vstack(J_constr)

    # Compute Delassus matrix for constraints
    G_constraints = J_constr @ M_inv @ J_constr.T

    # Compute constraint acceleration
    # TODO: add J̇_constr with efficient computation
    CW_al_free_constr = J_constr @ BW_ν̇_free

    # Setup constraint optimization problem
    constraint_regularization = regularization * jnp.ones(n_kin_constraints)
    R = jnp.diag(constraint_regularization)
    A = G_constraints + R
    b = CW_al_free_constr + constr_baumgarte_term

    # Solve for constraint forces
    kin_constr_wrench_mixed = jnp.linalg.solve(A, -b).reshape(-1, 6)

    def transform_wrenches_to_inertial(wrench, transform_pair):
        """
        Transform wrench pairs in inertial representation.

        Args:
            wrench: Wrench vector with shape (6,).
            transform_pair: Pair of transformation matrices [W_H_F1, W_H_F2]

        Returns:
            Stack of transformed wrenches with shape (2, 6).
        """
        W_H_F1, W_H_F2 = transform_pair[0], transform_pair[1]
        wrench_F1 = wrench
        wrench_F2 = -wrench

        # Create wrench pair directly
        # Transform both at once
        wrench_F1_inertial = (
            ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=wrench_F1,
                transform=W_H_F1,
                other_representation=VelRepr.Mixed,
                is_force=True,
            )
        )
        wrench_F2_inertial = (
            ModelDataWithVelocityRepresentation.other_representation_to_inertial(
                array=wrench_F2,
                transform=W_H_F2,
                other_representation=VelRepr.Mixed,
                is_force=True,
            )
        )

        return jnp.stack([wrench_F1_inertial, wrench_F2_inertial])

    kin_constr_wrench_pairs_inertial = jax.vmap(transform_wrenches_to_inertial)(
        kin_constr_wrench_mixed, W_H_constr_pairs
    )

    return kin_constr_wrench_pairs_inertial
