import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


def collidable_points_pos_vel(
    model: js.model.JaxSimModel,
    *,
    link_transforms: jtp.Matrix,
    link_velocities: jtp.Matrix,
    contact_mode: str = "enabled",
) -> tuple[jtp.Matrix, jtp.Matrix]:
    """

    Compute the position and linear velocity of the enabled collidable points in the world frame.

    Args:
        model: The model to consider.
        link_transforms: The transforms from the world frame to each link.
        link_velocities: The linear and angular velocities of each link.

    Returns:
        A tuple containing the position and linear velocity of the enabled collidable points.
    """

    contact_parameters = model.kin_dyn_parameters.contact_parameters
    enabled_mask = jnp.array(contact_parameters.enabled, dtype=bool)
    parent_link_indices = jnp.array(contact_parameters.body, dtype=int)
    collidable_points = contact_parameters.point

    if len(parent_link_indices) == 0:
        return jnp.array(0).astype(float), jnp.empty(0).astype(float)

    parent_link_transforms = link_transforms[parent_link_indices]
    parent_link_velocities = link_velocities[parent_link_indices]

    W_p_L = parent_link_transforms[..., 0:3, 3]
    W_R_L = parent_link_transforms[..., 0:3, 0:3]
    W_p_Ci = W_p_L + jnp.einsum("...ij,...j->...i", W_R_L, collidable_points)

    W_v_WL = parent_link_velocities[..., 0:3]
    W_ω_WL = parent_link_velocities[..., 3:6]

    # The collidable-point coordinate derivative is the mixed linear velocity of
    # the implicit point frame rigidly attached to the parent link.
    CW_vl_WC = W_v_WL + jnp.cross(W_ω_WL, W_p_Ci)

    match contact_mode:
        case "enabled":
            indices = contact_parameters.indices_of_enabled_collidable_points
            return W_p_Ci[indices], CW_vl_WC[indices]
        case "masked":
            return W_p_Ci, jnp.where(enabled_mask[:, None], CW_vl_WC, 0.0)
        case _:
            raise ValueError(contact_mode)
