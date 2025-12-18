import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.math import Adjoint

from .common import VelRepr

# =======================
# Index-related functions
# =======================


@functools.partial(jax.jit, static_argnames="link_name")
def name_to_idx(model: js.model.JaxSimModel, *, link_name: str) -> jtp.Int:
    """
    Convert the name of a link to its index.

    Args:
        model: The model to consider.
        link_name: The name of the link.

    Returns:
        The index of the link.
    """

    if link_name not in model.link_names():
        raise ValueError(f"Link '{link_name}' not found in the model.")

    return (
        jnp.array(model.kin_dyn_parameters.link_names.index(link_name))
        .astype(int)
        .squeeze()
    )


def idx_to_name(model: js.model.JaxSimModel, *, link_index: jtp.IntLike) -> str:
    """
    Convert the index of a link to its name.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The name of the link.
    """

    exceptions.raise_value_error_if(
        condition=link_index < 0,
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    return model.kin_dyn_parameters.link_names[link_index]


@functools.partial(jax.jit, static_argnames="link_names")
def names_to_idxs(
    model: js.model.JaxSimModel, *, link_names: Sequence[str]
) -> jax.Array:
    """
    Convert a sequence of link names to their corresponding indices.

    Args:
        model: The model to consider.
        link_names: The names of the links.

    Returns:
        The indices of the links.
    """

    return jnp.array(
        [name_to_idx(model=model, link_name=name) for name in link_names],
    ).astype(int)


def idxs_to_names(
    model: js.model.JaxSimModel, *, link_indices: Sequence[jtp.IntLike] | jtp.VectorLike
) -> tuple[str, ...]:
    """
    Convert a sequence of link indices to their corresponding names.

    Args:
        model: The model to consider.
        link_indices: The indices of the links.

    Returns:
        The names of the links.
    """

    return tuple(np.array(model.kin_dyn_parameters.link_names)[list(link_indices)])


# =========
# Link APIs
# =========


@jax.jit
def mass(model: js.model.JaxSimModel, *, link_index: jtp.IntLike) -> jtp.Float:
    """
    Return the mass of the link.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The mass of the link.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    return model.kin_dyn_parameters.link_parameters.mass[link_index].astype(float)


@jax.jit
def spatial_inertia(
    model: js.model.JaxSimModel, *, link_index: jtp.IntLike
) -> jtp.Matrix:
    r"""
    Compute the 6D spatial inertial of the link.

    Args:
        model: The model to consider.
        link_index: The index of the link.

    Returns:
        The :math:`6 \times 6` matrix representing the spatial inertia of the link expressed in
        the link frame (body-fixed representation).
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    link_parameters = jax.tree.map(
        lambda l: l[link_index], model.kin_dyn_parameters.link_parameters
    )

    return js.kin_dyn_parameters.LinkParameters.spatial_inertia(link_parameters)


@jax.jit
def transform(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
) -> jtp.Matrix:
    """
    Compute the SE(3) transform from the world frame to the link frame.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.

    Returns:
        The 4x4 matrix representing the transform.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    return data._link_transforms[link_index]


@jax.jit
def com_position(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    in_link_frame: jtp.BoolLike = True,
) -> jtp.Vector:
    """
    Compute the position of the center of mass of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        in_link_frame:
            Whether to return the position in the link frame or in the world frame.

    Returns:
        The 3D position of the center of mass of the link.
    """

    from jaxsim.math.inertia import Inertia

    _, L_p_CoM, _ = Inertia.to_params(
        M=spatial_inertia(model=model, link_index=link_index)
    )

    def com_in_link_frame():
        return L_p_CoM.squeeze()

    def com_in_inertial_frame():
        W_H_L = transform(link_index=link_index, model=model, data=data)
        W_p̃_CoM = W_H_L @ jnp.hstack([L_p_CoM.squeeze(), 1])

        return W_p̃_CoM[0:3].squeeze()

    return jax.lax.select(
        pred=in_link_frame,
        on_true=com_in_link_frame(),
        on_false=com_in_inertial_frame(),
    )


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
def jacobian(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the free-floating jacobian of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        input_representation:
            The input velocity representation of the free-floating jacobian.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian.

    Returns:
        The :math:`6 \times (6+n)` free-floating jacobian of the link.

    Note:
        The input representation of the free-floating jacobian is the active
        velocity representation.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    # Compute the doubly-left free-floating full jacobian.
    B_J_full_WX_B, B_H_Li = jaxsim.rbda.jacobian_full_doubly_left(
        model=model,
        joint_positions=data.joint_positions,
    )

    # Compute the actual doubly-left free-floating jacobian of the link.
    κb = model.kin_dyn_parameters.support_body_array_bool[link_index]
    B_J_WL_B = jnp.hstack([jnp.ones(5), κb]) * B_J_full_WX_B

    # Adjust the input representation such that `J_WL_I @ I_ν`.
    match input_representation:
        case VelRepr.Inertial:
            W_H_B = data._base_transform
            B_X_W = Adjoint.from_transform(transform=W_H_B, inverse=True)
            B_J_WL_I = B_J_WL_W = B_J_WL_B @ jax.scipy.linalg.block_diag(  # noqa: F841
                B_X_W, jnp.eye(model.dofs())
            )

        case VelRepr.Body:
            B_J_WL_I = B_J_WL_B

        case VelRepr.Mixed:
            W_R_B = jaxsim.math.Quaternion.to_dcm(data.base_orientation)
            BW_H_B = jnp.eye(4).at[0:3, 0:3].set(W_R_B)
            B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)
            B_J_WL_I = B_J_WL_BW = B_J_WL_B @ jax.scipy.linalg.block_diag(  # noqa: F841
                B_X_BW, jnp.eye(model.dofs())
            )

        case _:
            raise ValueError(input_representation)

    B_H_L = B_H_Li[link_index]

    # Adjust the output representation such that `O_v_WL_I = O_J_WL_I @ I_ν`.
    match output_vel_repr:
        case VelRepr.Inertial:
            W_H_B = data._base_transform
            W_X_B = Adjoint.from_transform(transform=W_H_B)
            O_J_WL_I = W_J_WL_I = W_X_B @ B_J_WL_I  # noqa: F841

        case VelRepr.Body:
            L_X_B = Adjoint.from_transform(transform=B_H_L, inverse=True)
            L_J_WL_I = L_X_B @ B_J_WL_I
            O_J_WL_I = L_J_WL_I

        case VelRepr.Mixed:
            W_H_B = data._base_transform
            W_H_L = W_H_B @ B_H_L
            LW_H_L = W_H_L.at[0:3, 3].set(jnp.zeros(3))
            LW_H_B = LW_H_L @ jaxsim.math.Transform.inverse(B_H_L)
            LW_X_B = Adjoint.from_transform(transform=LW_H_B)
            LW_J_WL_I = LW_X_B @ B_J_WL_I
            O_J_WL_I = LW_J_WL_I

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def velocity(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Vector:
    """
    Compute the 6D velocity of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        output_vel_repr:
            The output velocity representation of the link velocity.

    Returns:
        The 6D velocity of the link in the specified velocity representation.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Get the link jacobian having I as input representation (taken from data)
    # and O as output representation, specified by the user (or taken from data).
    O_J_WL_I = jacobian(
        model=model,
        data=data,
        link_index=link_index,
        output_vel_repr=output_vel_repr,
    )

    # Get the generalized velocity in the input velocity representation.
    I_ν = data.generalized_velocity(output_vel_repr)

    # Compute the link velocity in the output velocity representation.
    return O_J_WL_I @ I_ν


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
def jacobian_derivative(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    r"""
    Compute the derivative of the free-floating jacobian of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.
        input_representation:
            The input velocity representation of the free-floating jacobian.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivative.

    Returns:
        The derivative of the :math:`6 \times (6+n)` free-floating jacobian of the link.

    Note:
        The input representation of the free-floating jacobian derivative is the active
        velocity representation.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    O_J̇_WL_I = js.model.generalized_free_floating_jacobian_derivative(
        model=model,
        data=data,
        input_representation=input_representation,
        output_vel_repr=output_vel_repr,
    )[link_index]

    return O_J̇_WL_I


@jax.jit
def bias_acceleration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_index: jtp.IntLike,
) -> jtp.Vector:
    """
    Compute the bias acceleration of the link.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        link_index: The index of the link.

    Returns:
        The 6D bias acceleration of the link.
    """

    exceptions.raise_value_error_if(
        condition=jnp.array(
            [link_index < 0, link_index >= model.number_of_links()]
        ).any(),
        msg="Invalid link index '{idx}'",
        idx=link_index,
    )

    # Compute the bias acceleration of all links in the active representation.
    O_v̇_WL = js.model.link_bias_accelerations(model=model, data=data)[link_index]
    return O_v̇_WL
