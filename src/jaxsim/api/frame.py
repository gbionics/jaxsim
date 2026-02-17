import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.math import Adjoint, Cross, Transform

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
    I_ν = data.generalized_velocity()

    # Compute the frame velocity in the output velocity representation.
    return O_J_WF_I @ I_ν


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
@js.common.named_scope
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the free-floating jacobian of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        input_representation:
            The input velocity representation of the free-floating jacobian.
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

    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    # Get the index of the parent link.
    L = idx_of_parent_link(model=model, frame_index=frame_index)

    # Compute the Jacobian of the parent link using body-fixed output representation.
    L_J_WL = js.link.jacobian(
        model=model,
        data=data,
        link_index=L,
        input_representation=input_representation,
        output_vel_repr=VelRepr.Body,
    )

    # Adjust the output representation.
    match output_vel_repr:
        case VelRepr.Inertial:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_X_L = Adjoint.from_transform(transform=W_H_L)
            W_J_WL = W_X_L @ L_J_WL
            O_J_WL_I = W_J_WL

        case VelRepr.Body:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            F_H_L = Transform.inverse(W_H_F) @ W_H_L
            F_X_L = Adjoint.from_transform(transform=F_H_L)
            F_J_WL = F_X_L @ L_J_WL
            O_J_WL_I = F_J_WL

        case VelRepr.Mixed:
            W_H_L = js.link.transform(model=model, data=data, link_index=L)
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            F_H_L = Transform.inverse(W_H_F) @ W_H_L
            FW_H_F = W_H_F.at[0:3, 3].set(jnp.zeros(3))
            FW_H_L = FW_H_F @ F_H_L
            FW_X_L = Adjoint.from_transform(transform=FW_H_L)
            FW_J_WL = FW_X_L @ L_J_WL
            O_J_WL_I = FW_J_WL

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
@js.common.named_scope
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    frame_index: jtp.IntLike,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        frame_index: The index of the frame.
        input_representation:
            The input velocity representation of data.
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

    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    # Get the index of the parent link.
    L = idx_of_parent_link(model=model, frame_index=frame_index)

    # Compute the Jacobian of the parent link in inertial representation.
    W_J_WL_W = js.link.jacobian(
        model=model,
        data=data,
        link_index=L,
        input_representation=VelRepr.Inertial,
        output_vel_repr=VelRepr.Inertial,
    )

    # Compute the Jacobian derivative of the parent link in inertial representation.
    W_J̇_WL_W = js.link.jacobian_derivative(
        model=model,
        data=data,
        link_index=L,
        input_representation=VelRepr.Inertial,
        output_vel_repr=VelRepr.Inertial,
    )

    # =====================================================
    # Compute quantities to adjust the input representation
    # =====================================================

    def compute_T(model: js.model.JaxSimModel, X: jtp.Matrix) -> jtp.Matrix:
        In = jnp.eye(model.dofs())
        T = jax.scipy.linalg.block_diag(X, In)
        return T

    def compute_Ṫ(model: js.model.JaxSimModel, Ẋ: jtp.Matrix) -> jtp.Matrix:
        On = jnp.zeros(shape=(model.dofs(), model.dofs()))
        Ṫ = jax.scipy.linalg.block_diag(Ẋ, On)
        return Ṫ

    # Compute the operator to change the representation of ν, and its
    # time derivative.
    match input_representation:
        case VelRepr.Inertial:
            W_H_W = jnp.eye(4)
            W_X_W = Adjoint.from_transform(transform=W_H_W)
            W_Ẋ_W = jnp.zeros((6, 6))

            T = compute_T(model=model, X=W_X_W)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_W)

        case VelRepr.Body:
            W_H_B = data._base_transform
            W_X_B = Adjoint.from_transform(transform=W_H_B)
            B_v_WB = data.base_velocity(input_representation)
            B_vx_WB = Cross.vx(B_v_WB)
            W_Ẋ_B = W_X_B @ B_vx_WB

            T = compute_T(model=model, X=W_X_B)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_B)

        case VelRepr.Mixed:
            W_H_B = data._base_transform
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_X_BW = Adjoint.from_transform(transform=W_H_BW)
            BW_v_WB = data.base_velocity(input_representation)
            BW_v_W_BW = BW_v_WB.at[3:6].set(jnp.zeros(3))
            BW_vx_W_BW = Cross.vx(BW_v_W_BW)
            W_Ẋ_BW = W_X_BW @ BW_vx_W_BW

            T = compute_T(model=model, X=W_X_BW)
            Ṫ = compute_Ṫ(model=model, Ẋ=W_Ẋ_BW)

        case _:
            raise ValueError(input_representation)

    # =====================================================
    # Compute quantities to adjust the output representation
    # =====================================================

    match output_vel_repr:
        case VelRepr.Inertial:
            O_X_W = W_X_W = Adjoint.from_transform(transform=jnp.eye(4))
            O_Ẋ_W = W_Ẋ_W = jnp.zeros((6, 6))

        case VelRepr.Body:
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            O_X_W = F_X_W = Adjoint.from_transform(transform=W_H_F, inverse=True)
            W_nu = data.generalized_velocity(VelRepr.Inertial)
            W_v_WF = W_J_WL_W @ W_nu
            W_vx_WF = Cross.vx(W_v_WF)
            O_Ẋ_W = F_Ẋ_W = -F_X_W @ W_vx_WF  # noqa: F841

        case VelRepr.Mixed:
            W_H_F = transform(model=model, data=data, frame_index=frame_index)
            W_H_FW = W_H_F.at[0:3, 0:3].set(jnp.eye(3))
            FW_H_W = Transform.inverse(W_H_FW)
            O_X_W = FW_X_W = Adjoint.from_transform(transform=FW_H_W)
            FW_J_WF_FW = jacobian(
                model=model,
                data=data,
                frame_index=frame_index,
                output_vel_repr=VelRepr.Mixed,
            )
            FW_v_WF = FW_J_WF_FW @ data.generalized_velocity(VelRepr.Mixed)
            W_v_W_FW = jnp.zeros(6).at[0:3].set(FW_v_WF[0:3])
            W_vx_W_FW = Cross.vx(W_v_W_FW)
            O_Ẋ_W = FW_Ẋ_W = -FW_X_W @ W_vx_W_FW  # noqa: F841

        case _:
            raise ValueError(output_vel_repr)

    O_J̇_WF_I = jnp.zeros(shape=(6, 6 + model.dofs()))
    O_J̇_WF_I += O_Ẋ_W @ W_J_WL_W @ T
    O_J̇_WF_I += O_X_W @ W_J̇_WL_W @ T
    O_J̇_WF_I += O_X_W @ W_J_WL_W @ Ṫ

    return O_J̇_WF_I
