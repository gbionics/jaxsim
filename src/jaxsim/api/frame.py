import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.math import Cross

from .common import VelRepr

# =======================
# Index-related functions
# =======================


@jax.jit
@js.common.named_scope
def idx_of_parent_link(
    model: js.model.JaxSimModel, *, frame_index: jtp.IntLike
) -> jtp.Int:
    """
    Get the index of the link to which the frame is rigidly attached.

    Args:
        model: The model to consider.
        frame_index: The index of the frame.

    Returns:
        The index of the frame's parent link.
    """

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    return jnp.array(model.kin_dyn_parameters.frame_parameters.body)[
        frame_index - model.number_of_links()
    ]


@functools.partial(jax.jit, static_argnames="frame_name")
@js.common.named_scope
def name_to_idx(model: js.model.JaxSimModel, *, frame_name: str) -> jtp.Int:
    """
    Convert the name of a frame to its index.

    Args:
        model: The model to consider.
        frame_name: The name of the frame.

    Returns:
        The index of the frame.
    """

    if frame_name not in model.frame_names():
        raise ValueError(f"Frame '{frame_name}' not found in the model.")

    return (
        jnp.array(
            model.number_of_links()
            + model.kin_dyn_parameters.frame_parameters.name.index(frame_name)
        )
        .astype(int)
        .squeeze()
    )


def idx_to_name(model: js.model.JaxSimModel, *, frame_index: jtp.IntLike) -> str:
    """
    Convert the index of a frame to its name.

    Args:
        model: The model to consider.
        frame_index: The index of the frame.

    Returns:
        The name of the frame.
    """

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    return model.kin_dyn_parameters.frame_parameters.name[
        frame_index - model.number_of_links()
    ]


@functools.partial(jax.jit, static_argnames=["frame_names"])
@js.common.named_scope
def names_to_idxs(
    model: js.model.JaxSimModel, *, frame_names: Sequence[str]
) -> jax.Array:
    """
    Convert a sequence of frame names to their corresponding indices.

    Args:
        model: The model to consider.
        frame_names: The names of the frames.

    Returns:
        The indices of the frames.
    """

    return jnp.array(
        [name_to_idx(model=model, frame_name=name) for name in frame_names]
    ).astype(int)


def idxs_to_names(
    model: js.model.JaxSimModel, *, frame_indices: Sequence[jtp.IntLike]
) -> tuple[str, ...]:
    """
    Convert a sequence of frame indices to their corresponding names.

    Args:
        model: The model to consider.
        frame_indices: The indices of the frames.

    Returns:
        The names of the frames.
    """

    return tuple(idx_to_name(model=model, frame_index=idx) for idx in frame_indices)


# ==========
# Frame APIs
# ==========


@jax.jit
@js.common.named_scope
def transform(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
) -> jtp.Matrix:
    """
    Compute the SE(3) transform from the world frame to the specified frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame for which the transform is requested.

    Returns:
        The 4x4 matrix representing the transform.
    """

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    # Compute the necessary transforms.
    L = idx_of_parent_link(model=model, frame_index=frame_index)
    W_H_L = js.link.transform(model=model, data=data, link_index=L)

    # Get the static frame pose wrt the parent link.
    L_H_F = model.kin_dyn_parameters.frame_parameters.transform[
        frame_index - model.number_of_links()
    ]

    # Combine the transforms computing the frame pose.
    return W_H_L @ L_H_F


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
@js.common.named_scope
def velocity(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Vector:
    """
    Compute the 6D velocity of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        output_vel_repr:
            The output velocity representation of the frame velocity.

    Returns:
        The 6D velocity of the frame in the specified velocity representation.
    """
    n_l = model.number_of_links()
    n_f = model.number_of_frames()

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the frame jacobian having I as input representation (taken from data)
    # and O as output representation, specified by the user (or taken from data).
    O_J_WF_I = jacobian(
        model=model,
        data=data,
        frame_index=frame_index,
        output_vel_repr=output_vel_repr,
    )

    # Get the generalized velocity in the input velocity representation.
    I_ν = data.generalized_velocity

    # Compute the frame velocity in the output velocity representation.
    return O_J_WF_I @ I_ν


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
@js.common.named_scope
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the free-floating jacobian of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The :math:`6 \times (6+n)` free-floating jacobian of the frame.

    Note:
        The input representation of the free-floating jacobian is the active
        velocity representation.
    """

    n_l = model.number_of_links()
    n_f = model.number_of_frames()

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the index of the parent link.
    L = idx_of_parent_link(model=model, frame_index=frame_index)

    # Index the model-level body Jacobian so vmapped frame queries reuse the
    # shared parent-link computation rather than rebuilding it per frame.
    L_J_WL = js.model.generalized_free_floating_jacobian(
        model=model,
        data=data,
        output_vel_repr=VelRepr.Body,
    )[L]
    W_H_L = data._link_transforms[L]
    L_H_F = model.kin_dyn_parameters.frame_parameters.transform[
        frame_index - model.number_of_links()
    ]
    L_p_F = L_H_F[0:3, 3]

    # Adjust the output representation.
    match output_vel_repr:
        case VelRepr.Inertial:
            W_X_L = js.model._adjoint_from_rotation_translation(
                rotation=W_H_L[0:3, 0:3],
                translation=W_H_L[0:3, 3],
            )
            W_J_WL = W_X_L @ L_J_WL
            O_J_WL_I = W_J_WL

        case VelRepr.Body:
            F_X_L = js.model._inverse_adjoint_from_rotation_translation(
                rotation=L_H_F[0:3, 0:3],
                translation=L_p_F,
            )
            F_J_WL = F_X_L @ L_J_WL
            O_J_WL_I = F_J_WL

        case VelRepr.Mixed:
            W_R_L = W_H_L[0:3, 0:3]
            FW_X_L = js.model._adjoint_from_rotation_translation(
                rotation=W_R_L,
                translation=-W_R_L @ L_p_F,
            )
            FW_J_WL = FW_X_L @ L_J_WL
            O_J_WL_I = FW_J_WL

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
@js.common.named_scope
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivative.

    Returns:
        The derivative of the :math:`6 \times (6+n)` free-floating jacobian of the frame.

    Note:
        The input representation of the free-floating jacobian derivative is the active
        velocity representation.
    """

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    exceptions.raise_value_error_if(
        condition=jnp.array([frame_index < n_l, frame_index >= n_l + n_f]).any(),
        msg="Invalid frame index '{idx}'",
        idx=frame_index,
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the index of the parent link.
    L = idx_of_parent_link(model=model, frame_index=frame_index)
    W_J_WL_I = js.model.generalized_free_floating_jacobian(
        model=model,
        data=data,
        output_vel_repr=VelRepr.Inertial,
    )[L]
    W_J̇_WL_I = js.model.generalized_free_floating_jacobian_derivative(
        model=model,
        data=data,
        output_vel_repr=VelRepr.Inertial,
    )[L]

    W_H_L = data._link_transforms[L]
    L_H_F = model.kin_dyn_parameters.frame_parameters.transform[
        frame_index - model.number_of_links()
    ]
    W_H_F = W_H_L @ L_H_F

    # =====================================================
    # Compute quantities to adjust the output representation
    # =====================================================

    W_v_WF = W_J_WL_I @ data.generalized_velocity

    match output_vel_repr:
        case VelRepr.Inertial:
            O_X_W = jnp.eye(6, dtype=W_H_F.dtype)
            O_Ẋ_W = jnp.zeros((6, 6), dtype=W_H_F.dtype)

        case VelRepr.Body:
            O_X_W = js.model._inverse_adjoint_from_rotation_translation(
                rotation=W_H_F[0:3, 0:3],
                translation=W_H_F[0:3, 3],
            )
            O_Ẋ_W = -O_X_W @ Cross.vx(W_v_WF)

        case VelRepr.Mixed:
            O_X_W = js.model._inverse_adjoint_from_rotation_translation(
                rotation=jnp.eye(3, dtype=W_H_F.dtype),
                translation=W_H_F[0:3, 3],
            )
            FW_v_WF = O_X_W @ W_v_WF
            W_v_W_FW = FW_v_WF.at[3:6].set(jnp.zeros_like(FW_v_WF[3:6]))
            O_Ẋ_W = -O_X_W @ Cross.vx(W_v_W_FW)

        case _:
            raise ValueError(output_vel_repr)

    O_J̇_WF_I = O_Ẋ_W @ W_J_WL_I
    O_J̇_WF_I += O_X_W @ W_J̇_WL_I

    return O_J̇_WF_I
