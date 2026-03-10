from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.exceptions
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import Adjoint, Cross, Skew
from jaxsim.rbda.contacts import SoftContacts

from .common import VelRepr


def _contact_layout(
    model: js.model.JaxSimModel, contact_mode: str = "enabled"
) -> tuple[jtp.Vector, jtp.Matrix, jtp.Vector]:
    contact_parameters = model.kin_dyn_parameters.contact_parameters
    enabled_mask = jnp.array(contact_parameters.enabled, dtype=bool)
    parent_link_indices = jnp.array(contact_parameters.body, dtype=int)
    collidable_points = contact_parameters.point

    match contact_mode:
        case "enabled":
            indices = contact_parameters.indices_of_enabled_collidable_points
            return (
                parent_link_indices[indices],
                collidable_points[indices],
                enabled_mask[indices],
            )
        case "masked":
            return parent_link_indices, collidable_points, enabled_mask
        case _:
            raise ValueError(contact_mode)


def _contact_point_pose_data(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_mode: str,
) -> tuple[jtp.Vector, jtp.Matrix, jtp.Matrix, jtp.Matrix, jtp.Matrix]:
    parent_link_indices, L_p_Ci, _ = _contact_layout(
        model=model, contact_mode=contact_mode
    )
    W_H_L = data._link_transforms[parent_link_indices]
    W_R_C = W_H_L[..., 0:3, 0:3]
    W_p_C = W_H_L[..., 0:3, 3] + jnp.einsum("...ij,...j->...i", W_R_C, L_p_Ci)
    return parent_link_indices, L_p_Ci, W_H_L, W_R_C, W_p_C


def _inverse_adjoint_from_rotation_translation(
    rotation: jtp.Matrix,
    translation: jtp.Vector,
) -> jtp.Matrix:
    R_T = jnp.swapaxes(rotation, -1, -2)
    zeros = jnp.zeros_like(R_T)
    top_right = -jnp.einsum("...ij,...jk->...ik", R_T, Skew.wedge(translation))
    return jnp.concatenate(
        [
            jnp.concatenate([R_T, top_right], axis=-1),
            jnp.concatenate([zeros, R_T], axis=-1),
        ],
        axis=-2,
    )


def _inverse_adjoint_from_translation(translation: jtp.Vector) -> jtp.Matrix:
    batch_shape = translation.shape[:-1]
    identity = jnp.broadcast_to(jnp.eye(3), batch_shape + (3, 3)).astype(
        translation.dtype
    )
    zeros = jnp.zeros_like(identity)
    top_right = -Skew.wedge(translation)
    return jnp.concatenate(
        [
            jnp.concatenate([identity, top_right], axis=-1),
            jnp.concatenate([zeros, identity], axis=-1),
        ],
        axis=-2,
    )


def _apply_input_representation(J: jtp.Matrix, X: jtp.Matrix) -> jtp.Matrix:
    transformed_base = jnp.einsum("...ij,...jk->...ik", J[..., :, 0:6], X)
    return jnp.concatenate([transformed_base, J[..., :, 6:]], axis=-1)


def _apply_input_representation_derivative(J: jtp.Matrix, X_dot: jtp.Matrix) -> jtp.Matrix:
    transformed_base = jnp.einsum("...ij,...jk->...ik", J[..., :, 0:6], X_dot)
    return jnp.concatenate([transformed_base, jnp.zeros_like(J[..., :, 6:])], axis=-1)


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def collidable_point_enabled_mask(
    model: js.model.JaxSimModel,
    *,
    contact_mode: str = "enabled",
) -> jtp.Vector:
    """
    Return the enabled-mask matching the selected collidable-point layout.
    """

    *_, enabled_mask = _contact_layout(model=model, contact_mode=contact_mode)
    return enabled_mask


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def collidable_point_indices(
    model: js.model.JaxSimModel,
    *,
    contact_mode: str = "enabled",
) -> jtp.Vector:
    """
    Return the original collidable-point indices matching the selected layout.
    """

    n_collidable_points = len(model.kin_dyn_parameters.contact_parameters.body)

    match contact_mode:
        case "enabled":
            return jnp.array(
                model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points,
                dtype=int,
            )
        case "masked":
            return jnp.arange(n_collidable_points, dtype=int)
        case _:
            raise ValueError(contact_mode)


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def collidable_point_kinematics(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_mode: str = "enabled",
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """
    Compute the position and 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position and velocity of the collidable points in the world frame.

    Note:
        The collidable point velocity is the plain coordinate derivative of the position.
        If we attach a frame C = (p_C, [C]) to the collidable point, it corresponds to
        the linear component of the mixed 6D frame velocity.
    """

    W_p_Ci, W_ṗ_Ci = jaxsim.rbda.collidable_points.collidable_points_pos_vel(
        model=model,
        link_transforms=data._link_transforms,
        link_velocities=data._link_velocities,
        contact_mode=contact_mode,
    )

    return W_p_Ci, W_ṗ_Ci


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def collidable_point_positions(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_mode: str = "enabled",
) -> jtp.Matrix:
    """
    Compute the position of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the collidable points in the world frame.
    """

    W_p_Ci, _ = collidable_point_kinematics(
        model=model, data=data, contact_mode=contact_mode
    )

    return W_p_Ci


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def collidable_point_velocities(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_mode: str = "enabled",
) -> jtp.Matrix:
    """
    Compute the 3D velocity of the collidable points in the world frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The 3D velocity of the collidable points.
    """

    _, W_ṗ_Ci = collidable_point_kinematics(
        model=model, data=data, contact_mode=contact_mode
    )

    return W_ṗ_Ci


@functools.partial(jax.jit, static_argnames=["link_names"])
@js.common.named_scope
def in_contact(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_names: tuple[str, ...] | None = None,
) -> jtp.Vector:
    """
    Return whether the links are in contact with the terrain.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_names:
            The names of the links to consider. If None, all links are considered.

    Returns:
        A boolean vector indicating whether the links are in contact with the terrain.
    """

    if link_names is not None and set(link_names).difference(model.link_names()):
        raise ValueError("One or more link names are not part of the model")

    # Get the indices of the enabled collidable points.
    indices_of_enabled_collidable_points = (
        model.kin_dyn_parameters.contact_parameters.indices_of_enabled_collidable_points
    )

    parent_link_idx_of_enabled_collidable_points = jnp.array(
        model.kin_dyn_parameters.contact_parameters.body, dtype=int
    )[indices_of_enabled_collidable_points]

    W_p_Ci = collidable_point_positions(model=model, data=data)

    terrain_height = jax.vmap(lambda x, y: model.terrain.height(x=x, y=y))(
        W_p_Ci[:, 0], W_p_Ci[:, 1]
    )

    below_terrain = W_p_Ci[:, 2] <= terrain_height

    link_idxs = (
        js.link.names_to_idxs(link_names=link_names, model=model)
        if link_names is not None
        else jnp.arange(model.number_of_links())
    )

    links_in_contact = jax.vmap(
        lambda link_index: jnp.where(
            parent_link_idx_of_enabled_collidable_points == link_index,
            below_terrain,
            jnp.zeros_like(below_terrain, dtype=bool),
        ).any()
    )(link_idxs)

    return links_in_contact


def estimate_good_soft_contacts_parameters(
    *args, **kwargs
) -> jaxsim.rbda.contacts.ContactParamsTypes:
    """
    Estimate good soft contacts parameters. Deprecated, use `estimate_good_contact_parameters` instead.
    """

    msg = "This method is deprecated, please use `{}`."
    logging.warning(msg.format(estimate_good_contact_parameters.__name__))
    return estimate_good_contact_parameters(*args, **kwargs)


def estimate_good_contact_parameters(
    model: js.model.JaxSimModel,
    *,
    standard_gravity: jtp.FloatLike = jaxsim.math.STANDARD_GRAVITY,
    static_friction_coefficient: jtp.FloatLike = 0.5,
    number_of_active_collidable_points_steady_state: jtp.IntLike = 1,
    damping_ratio: jtp.FloatLike = 1.0,
    max_penetration: jtp.FloatLike | None = None,
) -> jaxsim.rbda.contacts.ContactParamsTypes:
    """
    Estimate good contact parameters.

    Args:
        model: The model to consider.
        standard_gravity: The standard gravity acceleration.
        static_friction_coefficient: The static friction coefficient.
        number_of_active_collidable_points_steady_state:
            The number of active collidable points in steady state.
        damping_ratio: The damping ratio.
        max_penetration: The maximum penetration allowed.

    Returns:
        The estimated good contacts parameters.

    Note:
        This is primarily a convenience function for soft-like contact models.
        However, it provides with some good default parameters also for the other ones.

    Note:
        This method provides a good set of contacts parameters.
        The user is encouraged to fine-tune the parameters based on the
        specific application.
    """
    if max_penetration is None:
        zero_data = js.data.JaxSimModelData.build(model=model)
        W_pz_CoM = js.com.com_position(model=model, data=zero_data)[2]
        if model.floating_base():
            W_pz_C = collidable_point_positions(model=model, data=zero_data)[:, -1]
            W_pz_CoM = W_pz_CoM - W_pz_C.min()

        # Consider as default a 1% of the model center of mass height.
        max_penetration = 0.01 * W_pz_CoM

    nc = number_of_active_collidable_points_steady_state
    return model.contact_model._parameters_class().build_default_from_jaxsim_model(
        model=model,
        standard_gravity=standard_gravity,
        static_friction_coefficient=static_friction_coefficient,
        max_penetration=max_penetration,
        number_of_active_collidable_points_steady_state=nc,
        damping_ratio=damping_ratio,
    )


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def transforms(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    contact_mode: str = "enabled",
) -> jtp.Array:
    r"""
    Return the pose of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The stacked SE(3) matrices of all enabled collidable points.

    Note:
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    _, _, W_H_L, _, W_p_C = _contact_point_pose_data(
        model=model,
        data=data,
        contact_mode=contact_mode,
    )

    return W_H_L.at[..., 0:3, 3].set(W_p_C)


@functools.partial(
    jax.jit, static_argnames=["output_vel_repr", "contact_mode"]
)
@js.common.named_scope
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
    contact_mode: str = "enabled",
) -> jtp.Array:
    r"""
    Return the free-floating Jacobian of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The stacked :math:`6 \times (6+n)` free-floating jacobians of the frames associated to the
        enabled collidable points.

    Note:
        Each collidable point is implicitly associated with a frame
        :math:`C = ({}^W p_C, [L])`, where :math:`{}^W p_C` is the position of the
        collidable point and :math:`[L]` is the orientation frame of the link it is
        rigidly attached to.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    parent_link_idx_of_enabled_collidable_points, _, _, W_R_C, W_p_C = (
        _contact_point_pose_data(
            model=model,
            data=data,
            contact_mode=contact_mode,
        )
    )

    # Compute the Jacobians of all links.
    W_J_WL = js.model.generalized_free_floating_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Inertial
    )

    # Compute the contact Jacobian.
    # In inertial-fixed output representation, the Jacobian of the parent link is also
    # the Jacobian of the frame C implicitly associated with the collidable point.
    W_J_WC = W_J_WL[parent_link_idx_of_enabled_collidable_points]

    # Adjust the output representation.
    match output_vel_repr:

        case VelRepr.Inertial:
            O_J_WC = W_J_WC

        case VelRepr.Body:
            C_X_W = _inverse_adjoint_from_rotation_translation(
                rotation=W_R_C,
                translation=W_p_C,
            )
            O_J_WC = jnp.einsum("...ij,...jk->...ik", C_X_W, W_J_WC)

        case VelRepr.Mixed:
            CW_X_W = _inverse_adjoint_from_translation(translation=W_p_C)
            O_J_WC = jnp.einsum("...ij,...jk->...ik", CW_X_W, W_J_WC)

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WC


@functools.partial(
    jax.jit, static_argnames=["output_vel_repr", "contact_mode"]
)
@js.common.named_scope
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
    contact_mode: str = "enabled",
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the enabled collidable points.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivative.

    Returns:
        The derivative of the :math:`6 \times (6+n)` free-floating jacobian of the enabled collidable points.

    Note:
        The input representation of the free-floating jacobian derivative is the active
        velocity representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    (
        parent_link_idx_of_enabled_collidable_points,
        _L_p_Ci,
        _W_H_L,
        W_R_C,
        W_p_C,
    ) = _contact_point_pose_data(
        model=model,
        data=data,
        contact_mode=contact_mode,
    )

    # Get the link velocities of the parent links.
    W_v_WC = data._link_velocities[parent_link_idx_of_enabled_collidable_points]

    # Compute the operator to change the representation of ν, and its
    # time derivative.
    match data.velocity_representation:
        case VelRepr.Inertial:
            X = jnp.eye(6)
            Ẋ = jnp.zeros((6, 6))

        case VelRepr.Body:
            W_H_B = data._base_transform
            X = W_X_B = Adjoint.from_transform(transform=W_H_B)  # noqa: F841
            B_v_WB = data.base_velocity
            B_vx_WB = Cross.vx(B_v_WB)
            Ẋ = W_Ẋ_B = W_X_B @ B_vx_WB  # noqa: F841

        case VelRepr.Mixed:
            W_H_B = data._base_transform
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            X = W_X_BW = Adjoint.from_transform(transform=W_H_BW)  # noqa: F841
            BW_v_WB = data.base_velocity
            BW_v_W_BW = BW_v_WB.at[3:6].set(jnp.zeros(3))
            BW_vx_W_BW = Cross.vx(BW_v_W_BW)
            Ẋ = W_Ẋ_BW = W_X_BW @ BW_vx_W_BW  # noqa: F841

        case _:
            raise ValueError(data.velocity_representation)

    # =====================================================
    # Compute quantities to adjust the output representation
    # =====================================================

    with data.switch_velocity_representation(VelRepr.Inertial):
        # Compute the Jacobian of the parent link in inertial representation.
        W_J_WL_W = js.model.generalized_free_floating_jacobian(
            model=model,
            data=data,
        )
        # Compute the Jacobian derivative of the parent link in inertial representation.
        W_J̇_WL_W = js.model.generalized_free_floating_jacobian_derivative(
            model=model,
            data=data,
        )
    W_J_WC_W = W_J_WL_W[parent_link_idx_of_enabled_collidable_points]
    W_J̇_WC_W = W_J̇_WL_W[parent_link_idx_of_enabled_collidable_points]

    W_J_WC_I = _apply_input_representation(W_J_WC_W, X)
    W_J̇_WC_input = _apply_input_representation(W_J̇_WC_W, X)
    W_J̇_WC_repr = _apply_input_representation_derivative(W_J_WC_W, Ẋ)

    match output_vel_repr:
        case VelRepr.Inertial:
            batch_shape = W_p_C.shape[:-1]
            O_X_W = jnp.broadcast_to(jnp.eye(6), batch_shape + (6, 6))
            O_Ẋ_W = jnp.zeros_like(O_X_W)

        case VelRepr.Body:
            O_X_W = _inverse_adjoint_from_rotation_translation(
                rotation=W_R_C,
                translation=W_p_C,
            )
            O_Ẋ_W = -jnp.einsum("...ij,...jk->...ik", O_X_W, Cross.vx(W_v_WC))

        case VelRepr.Mixed:
            O_X_W = _inverse_adjoint_from_translation(translation=W_p_C)
            CW_v_WC = jnp.einsum("...ij,...j->...i", O_X_W, W_v_WC)
            W_v_W_CW = jnp.zeros_like(CW_v_WC).at[..., 0:3].set(CW_v_WC[..., 0:3])
            O_Ẋ_W = -jnp.einsum("...ij,...jk->...ik", O_X_W, Cross.vx(W_v_W_CW))

        case _:
            raise ValueError(output_vel_repr)

    O_J̇_WC = jnp.einsum("...ij,...jk->...ik", O_Ẋ_W, W_J_WC_I)
    O_J̇_WC += jnp.einsum("...ij,...jk->...ik", O_X_W, W_J̇_WC_input)
    O_J̇_WC += jnp.einsum("...ij,...jk->...ik", O_X_W, W_J̇_WC_repr)

    return O_J̇_WC


@functools.partial(
    jax.jit, static_argnames=["contact_mode", "precision_policy"]
)
@js.common.named_scope
def link_contact_forces(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_torques: jtp.VectorLike | None = None,
    contact_mode: str = "enabled",
    precision_policy=None,
) -> tuple[jtp.Matrix, dict[str, jtp.Matrix]]:
    """
    Compute the 6D contact forces of all links of the model in inertial representation.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_forces:
            The 6D external forces to apply to the links expressed in inertial representation
        joint_torques:
            The joint torques acting on the joints.

    Returns:
        A `(nL, 6)` array containing the stacked 6D contact forces of the links,
        expressed in inertial representation.
    """

    # Compute the contact forces for each collidable point with the active contact model.
    W_f_C, aux_dict = model.contact_model.compute_contact_forces(
        model=model,
        data=data,
        contact_mode=contact_mode,
        precision_policy=precision_policy,
        **(
            dict(link_forces=link_forces, joint_force_references=joint_torques)
            if not isinstance(model.contact_model, SoftContacts)
            else {}
        ),
    )

    # Compute the 6D forces applied to the links equivalent to the forces applied
    # to the frames associated to the collidable points.
    W_f_L = link_forces_from_contact_forces(
        model=model,
        contact_forces=W_f_C,
        contact_mode=contact_mode,
    )

    return W_f_L, aux_dict


@functools.partial(jax.jit, static_argnames=["contact_mode"])
@js.common.named_scope
def link_forces_from_contact_forces(
    model: js.model.JaxSimModel,
    *,
    contact_forces: jtp.MatrixLike,
    contact_mode: str = "enabled",
) -> jtp.Matrix:
    """
    Compute the link forces from the contact forces.

    Args:
        model: The robot model considered by the contact model.
        contact_forces: The contact forces computed by the contact model.

    Returns:
        The 6D contact forces applied to the links and expressed in the frame of
        the velocity representation of data.
    """

    # Convert the contact forces to a JAX array.
    W_f_C = jnp.atleast_2d(jnp.array(contact_forces, dtype=float).squeeze())

    # Construct the vector defining the parent link index of each collidable point.
    # We use this vector to sum the 6D forces of all collidable points rigidly
    # attached to the same link.
    parent_link_index_of_collidable_points, _, _ = _contact_layout(
        model=model, contact_mode=contact_mode
    )

    # Sum the forces of collidable points rigidly attached to the same link.
    # Since contact forces are already expressed in the world frame, we only
    # need an indexed accumulation over parent-link ids.
    return jnp.zeros((model.number_of_links(), 6), dtype=W_f_C.dtype).at[
        parent_link_index_of_collidable_points
    ].add(W_f_C)
