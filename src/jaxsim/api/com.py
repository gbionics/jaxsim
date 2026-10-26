import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.math
import jaxsim.typing as jtp

from .common import VelRepr


@jax.jit
@js.common.named_scope
def com_position(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    """
    Compute the position of the center of mass of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The position of the center of mass of the model w.r.t. the world frame.
    """

    m = js.model.total_mass(model=model)

    W_H_L = data._link_transforms
    W_H_B = data._base_transform
    B_H_W = jaxsim.math.Transform.inverse(transform=W_H_B)

    def B_p̃_LCoM(i) -> jtp.Vector:
        m = js.link.mass(model=model, link_index=i)
        L_p_LCoM = js.link.com_position(
            model=model, data=data, link_index=i, in_link_frame=True
        )
        return m * B_H_W @ W_H_L[i] @ jnp.hstack([L_p_LCoM, 1])

    com_links = jax.vmap(B_p̃_LCoM)(jnp.arange(model.number_of_links()))

    B_p̃_CoM = (1 / m) * com_links.sum(axis=0)
    B_p̃_CoM = B_p̃_CoM.at[3].set(1)

    return (W_H_B @ B_p̃_CoM)[0:3].astype(float)


@jax.jit
@js.common.named_scope
def com_linear_velocity(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the linear velocity of the center of mass of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The linear velocity of the center of mass of the model in the
        active representation.

    Note:
        The linear velocity of the center of mass  is expressed in the mixed frame
        :math:`G = ({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`[C] = [W]` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`[C] = [B]` if the active velocity representation is body-fixed.
    """

    # Extract the linear component of the 6D average centroidal velocity.
    # This is expressed in G[B] in body-fixed representation, and in G[W] in
    # inertial-fixed or mixed representation.
    G_vl_WG = average_centroidal_velocity(model=model, data=data)[0:3]

    return G_vl_WG


@jax.jit
@js.common.named_scope
def centroidal_momentum(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the centroidal momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The centroidal momentum of the model.

    Note:
        The centroidal momentum is expressed in the mixed frame
        :math:`({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`C = W` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`C = B` if the active velocity representation is body-fixed.
    """

    ν = data.generalized_velocity
    G_J = centroidal_momentum_jacobian(model=model, data=data)

    return G_J @ ν


@jax.jit
@js.common.named_scope
def centroidal_momentum_jacobian(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    r"""
    Compute the Jacobian of the centroidal momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The Jacobian of the centroidal momentum of the model.

    Note:
        The frame corresponding to the output representation of this Jacobian is either
        :math:`G[W]`, if the active velocity representation is inertial-fixed or mixed,
        or :math:`G[B]`, if the active velocity representation is body-fixed.

    Note:
        This Jacobian is also known in the literature as Centroidal Momentum Matrix.
    """

    # Compute the Jacobian of the total momentum with body-fixed output representation.
    # We convert the output representation either to G[W] or G[B] below.
    B_Jh = js.model.total_momentum_jacobian(
        model=model, data=data, output_vel_repr=VelRepr.Body
    )

    W_H_B = data._base_transform
    B_H_W = jaxsim.math.Transform.inverse(W_H_B)

    W_p_CoM = com_position(model=model, data=data)

    match data.velocity_representation:
        case VelRepr.Inertial | VelRepr.Mixed:
            W_H_G = W_H_GW = (  # ruff: ignore[unused-variable]
                jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            )
        case VelRepr.Body:
            W_H_G = W_H_GB = W_H_B.at[0:3, 3].set(  # ruff: ignore[unused-variable]
                W_p_CoM
            )
        case _:
            raise ValueError(data.velocity_representation)

    # Compute the transform for 6D forces.
    G_Xf_B = jaxsim.math.Adjoint.from_transform(transform=B_H_W @ W_H_G).T

    return G_Xf_B @ B_Jh


@jax.jit
@js.common.named_scope
def locked_centroidal_spatial_inertia(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
):
    """
    Compute the locked centroidal spatial inertia of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The locked centroidal spatial inertia of the model.
    """

    with data.switch_velocity_representation(VelRepr.Body):
        B_Mbb_B = js.model.locked_spatial_inertia(model=model, data=data)

    W_H_B = data._base_transform
    W_p_CoM = com_position(model=model, data=data)

    match data.velocity_representation:
        case VelRepr.Inertial | VelRepr.Mixed:
            W_H_G = W_H_GW = (  # ruff: ignore[unused-variable]
                jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            )
        case VelRepr.Body:
            W_H_G = W_H_GB = W_H_B.at[0:3, 3].set(  # ruff: ignore[unused-variable]
                W_p_CoM
            )
        case _:
            raise ValueError(data.velocity_representation)

    B_H_G = jaxsim.math.Transform.inverse(W_H_B) @ W_H_G

    B_Xv_G = jaxsim.math.Adjoint.from_transform(transform=B_H_G)
    G_Xf_B = B_Xv_G.transpose()

    return G_Xf_B @ B_Mbb_B @ B_Xv_G


@jax.jit
@js.common.named_scope
def average_centroidal_velocity(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the average centroidal velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The average centroidal velocity of the model.

    Note:
        The average velocity is expressed in the mixed frame
        :math:`G = ({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`[C] = [W]` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`[C] = [B]` if the active velocity representation is body-fixed.
    """

    ν = data.generalized_velocity
    G_J = average_centroidal_velocity_jacobian(model=model, data=data)

    return G_J @ ν


@jax.jit
@js.common.named_scope
def average_centroidal_velocity_jacobian(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Matrix:
    r"""
    Compute the Jacobian of the average centroidal velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The Jacobian of the average centroidal velocity of the model.

    Note:
        The frame corresponding to the output representation of this Jacobian is either
        :math:`G[W]`, if the active velocity representation is inertial-fixed or mixed,
        or :math:`G[B]`, if the active velocity representation is body-fixed.
    """

    G_J = centroidal_momentum_jacobian(model=model, data=data)
    G_Mbb = locked_centroidal_spatial_inertia(model=model, data=data)

    return jnp.linalg.inv(G_Mbb) @ G_J


@jax.jit
@js.common.named_scope
def bias_acceleration(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    r"""
    Compute the bias linear acceleration of the center of mass.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The bias linear acceleration of the center of mass in the active representation.

    Note:
        The bias acceleration is expressed in the mixed frame
        :math:`G = ({}^W \mathbf{p}_{\text{CoM}}, [C])`, where :math:`[C] = [W]` if the
        active velocity representation is either inertial-fixed or mixed,
        and :math:`[C] = [B]` if the active velocity representation is body-fixed.
    """

    # Compute the pose of all links with forward kinematics.
    W_H_L = data._link_transforms

    # Compute the bias acceleration of all links by zeroing the generalized velocity
    # in the active representation.
    v̇_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

    def other_representation_to_body(
        C_v̇_WL: jtp.Vector, C_v_WC: jtp.Vector, L_H_C: jtp.Matrix, L_v_LC: jtp.Vector
    ) -> jtp.Vector:
        """
        Convert the body-fixed representation of the link bias acceleration
        C_v̇_WL expressed in a generic frame C to the body-fixed representation L_v̇_WL.
        """

        L_X_C = jaxsim.math.Adjoint.from_transform(transform=L_H_C)
        C_X_L = jaxsim.math.Adjoint.inverse(L_X_C)

        L_v̇_WL = L_X_C @ (C_v̇_WL + jaxsim.math.Cross.vx(C_X_L @ L_v_LC) @ C_v_WC)
        return L_v̇_WL

    # We need here to get the body-fixed bias acceleration of the links.
    # Since it's computed in the active representation, we need to convert it to body.
    match data.velocity_representation:

        case VelRepr.Body:
            L_a_bias_WL = v̇_bias_WL

        case VelRepr.Inertial:

            C_v̇_WL = W_v̇_bias_WL = v̇_bias_WL  # ruff: ignore[unused-variable]
            C_v_WC = W_v_WW = jnp.zeros(6)  # ruff: ignore[unused-variable]

            L_H_C = L_H_W = jax.vmap(  # ruff: ignore[unused-variable]
                jaxsim.math.Transform.inverse
            )(W_H_L)

            L_v_LC = L_v_LW = jax.vmap(  # ruff: ignore[unused-variable]
                lambda i: -js.link.velocity(
                    model=model, data=data, link_index=i, output_vel_repr=VelRepr.Body
                )
            )(jnp.arange(model.number_of_links()))

            L_a_bias_WL = jax.vmap(
                lambda i: other_representation_to_body(
                    C_v̇_WL=C_v̇_WL[i],
                    C_v_WC=C_v_WC,
                    L_H_C=L_H_C[i],
                    L_v_LC=L_v_LC[i],
                )
            )(jnp.arange(model.number_of_links()))

        case VelRepr.Mixed:

            C_v̇_WL = LW_v̇_bias_WL = v̇_bias_WL  # ruff: ignore[unused-variable]

            C_v_WC = LW_v_W_LW = jax.vmap(  # ruff: ignore[unused-variable]
                lambda i: js.link.velocity(
                    model=model, data=data, link_index=i, output_vel_repr=VelRepr.Mixed
                )
                .at[3:6]
                .set(jnp.zeros(3))
            )(jnp.arange(model.number_of_links()))

            L_H_C = L_H_LW = jax.vmap(  # ruff: ignore[unused-variable]
                lambda W_H_L: jaxsim.math.Transform.inverse(
                    W_H_L.at[0:3, 3].set(jnp.zeros(3))
                )
            )(W_H_L)

            L_v_LC = L_v_L_LW = jax.vmap(  # ruff: ignore[unused-variable]
                lambda i: -js.link.velocity(
                    model=model, data=data, link_index=i, output_vel_repr=VelRepr.Body
                )
                .at[0:3]
                .set(jnp.zeros(3))
            )(jnp.arange(model.number_of_links()))

            L_a_bias_WL = jax.vmap(
                lambda i: other_representation_to_body(
                    C_v̇_WL=C_v̇_WL[i],
                    C_v_WC=C_v_WC[i],
                    L_H_C=L_H_C[i],
                    L_v_LC=L_v_LC[i],
                )
            )(jnp.arange(model.number_of_links()))

        case _:
            raise ValueError(data.velocity_representation)

    # Compute the bias of the 6D momentum derivative.
    def bias_momentum_derivative_term(
        link_index: jtp.Int, L_a_bias_WL: jtp.Vector
    ) -> jtp.Vector:

        # Get the body-fixed 6D inertia matrix.
        L_M_L = js.link.spatial_inertia(model=model, link_index=link_index)

        # Compute the body-fixed 6D velocity.
        L_v_WL = js.link.velocity(
            model=model, data=data, link_index=link_index, output_vel_repr=VelRepr.Body
        )

        # Compute the world-to-link transformations for 6D forces.
        W_Xf_L = jaxsim.math.Adjoint.from_transform(
            transform=W_H_L[link_index], inverse=True
        ).T

        # Compute the contribution of the link to the bias acceleration of the CoM.
        W_ḣ_bias_link_contribution = W_Xf_L @ (
            L_M_L @ L_a_bias_WL + jaxsim.math.Cross.vx_star(L_v_WL) @ L_M_L @ L_v_WL
        )

        return W_ḣ_bias_link_contribution

    # Sum the contributions of all links to the bias acceleration of the CoM.
    W_ḣ_bias = jax.vmap(bias_momentum_derivative_term)(
        jnp.arange(model.number_of_links()), L_a_bias_WL
    ).sum(axis=0)

    # Compute the total mass of the model.
    m = js.model.total_mass(model=model)

    # Compute the position of the CoM.
    W_p_CoM = com_position(model=model, data=data)

    match data.velocity_representation:

        # G := G[W] = (W_p_CoM, [W])
        case VelRepr.Inertial | VelRepr.Mixed:

            W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            GW_Xf_W = jaxsim.math.Adjoint.from_transform(W_H_GW).T

            GW_ḣ_bias = GW_Xf_W @ W_ḣ_bias
            GW_v̇l_com_bias = GW_ḣ_bias[0:3] / m

            return GW_v̇l_com_bias

        # G := G[B] = (W_p_CoM, [B])
        case VelRepr.Body:

            GB_Xf_W = jaxsim.math.Adjoint.from_transform(
                transform=data._base_transform.at[0:3].set(W_p_CoM)
            ).T

            GB_ḣ_bias = GB_Xf_W @ W_ḣ_bias
            GB_v̇l_com_bias = GB_ḣ_bias[0:3] / m

            return GB_v̇l_com_bias

        case _:
            raise ValueError(data.velocity_representation)
