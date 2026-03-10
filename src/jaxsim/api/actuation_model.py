import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp


def _tn_curve_from_joint_velocities(
    model: js.model.JaxSimModel,
    *,
    joint_velocities: jtp.Vector,
) -> jtp.Vector:
    τ_max = model.actuation_params.torque_max
    ω_th = model.actuation_params.omega_th
    ω_max = model.actuation_params.omega_max
    abs_vel = jnp.abs(joint_velocities)
    return jnp.where(
        abs_vel <= ω_th,
        τ_max,
        jnp.where(
            abs_vel <= ω_max, τ_max * (1 - (abs_vel - ω_th) / (ω_max - ω_th)), 0.0
        ),
    )


def _compute_resultant_torques_from_state(
    model: js.model.JaxSimModel,
    *,
    joint_positions: jtp.Vector,
    joint_velocities: jtp.Vector,
    joint_force_references: jtp.Vector | None = None,
) -> jtp.Vector:
    """
    Internal helper operating directly on minimal state quantities.
    """

    τ_references = (
        jnp.atleast_1d(joint_force_references.squeeze())
        if joint_force_references is not None
        else jnp.zeros_like(joint_positions)
    ).astype(float)

    τ_position_limit = jnp.zeros_like(τ_references).astype(float)

    if model.dofs() > 0:
        k_j = jnp.array(
            model.kin_dyn_parameters.joint_parameters.position_limit_spring
        ).astype(float)
        d_j = jnp.array(
            model.kin_dyn_parameters.joint_parameters.position_limit_damper
        ).astype(float)

        lower_violation = jnp.clip(
            joint_positions - model.kin_dyn_parameters.joint_parameters.position_limits_min,
            max=0.0,
        )
        upper_violation = jnp.clip(
            joint_positions - model.kin_dyn_parameters.joint_parameters.position_limits_max,
            min=0.0,
        )

        τ_position_limit -= jnp.diag(k_j) @ (lower_violation + upper_violation)
        τ_position_limit -= jnp.positive(τ_position_limit) * jnp.diag(d_j) @ joint_velocities

    τ_friction = jnp.zeros_like(τ_references).astype(float)

    if model.dofs() > 0 and model.actuation_params.enable_friction:
        kc = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_static
        ).astype(float)
        kv = jnp.array(
            model.kin_dyn_parameters.joint_parameters.friction_viscous
        ).astype(float)
        τ_friction = -(jnp.diag(kc) @ jnp.sign(joint_velocities) + jnp.diag(kv) @ joint_velocities)

    τ_total = τ_references + τ_friction + τ_position_limit
    τ_lim = _tn_curve_from_joint_velocities(
        model=model,
        joint_velocities=joint_velocities,
    )
    return jnp.clip(τ_total, -τ_lim, τ_lim)


def compute_resultant_torques(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_force_references: jtp.Vector | None = None,
) -> jtp.Vector:
    """
    Compute the resultant torques acting on the joints.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_force_references: The joint force references to apply.

    Returns:
        The resultant torques acting on the joints.
    """

    return _compute_resultant_torques_from_state(
        model=model,
        joint_positions=data.joint_positions,
        joint_velocities=data.joint_velocities,
        joint_force_references=joint_force_references,
    )


def tn_curve_fn(
    model: js.model.JaxSimModel, data: js.data.JaxSimModelData
) -> jtp.Vector:
    """
    Compute the torque limits using the tn curve.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The torque limits.
    """

    return _tn_curve_from_joint_velocities(
        model=model,
        joint_velocities=data.joint_velocities,
    )
