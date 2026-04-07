import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.math import STANDARD_GRAVITY


def process_inputs(
    model: js.model.JaxSimModel,
    *,
    base_position: jtp.VectorLike | None = None,
    base_quaternion: jtp.VectorLike | None = None,
    joint_positions: jtp.VectorLike | None = None,
    base_linear_velocity: jtp.VectorLike | None = None,
    base_angular_velocity: jtp.VectorLike | None = None,
    joint_velocities: jtp.VectorLike | None = None,
    base_linear_acceleration: jtp.VectorLike | None = None,
    base_angular_acceleration: jtp.VectorLike | None = None,
    joint_accelerations: jtp.VectorLike | None = None,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    standard_gravity: jtp.ScalarLike | None = None,
) -> tuple[
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Vector,
    jtp.Matrix,
    jtp.Vector,
]:
    """
    Adjust the inputs to rigid-body dynamics algorithms.

    Args:
        model: The model to consider.
        base_position: The position of the base link.
        base_quaternion: The quaternion of the base link.
        joint_positions: The positions of the joints.
        base_linear_velocity: The linear velocity of the base link.
        base_angular_velocity: The angular velocity of the base link.
        joint_velocities: The velocities of the joints.
        base_linear_acceleration: The linear acceleration of the base link.
        base_angular_acceleration: The angular acceleration of the base link.
        joint_accelerations: The accelerations of the joints.
        joint_forces: The forces applied to the joints.
        link_forces: The forces applied to the links.
        standard_gravity: The standard gravity constant.

    Returns:
        The adjusted inputs.
    """

    dofs = model.dofs()
    nl = model.number_of_links()

    # Floating-base position.
    W_p_B = base_position
    W_Q_B = base_quaternion
    s = joint_positions

    # Floating-base velocity in inertial-fixed representation.
    W_vl_WB = base_linear_velocity
    W_ω_WB = base_angular_velocity
    ṡ = joint_velocities

    # Floating-base acceleration in inertial-fixed representation.
    W_v̇l_WB = base_linear_acceleration
    W_ω̇_WB = base_angular_acceleration
    s̈ = joint_accelerations

    # System dynamics inputs.
    f = link_forces
    τ = joint_forces

    # Fill missing data and adjust dimensions.
    s = jnp.atleast_1d(s.squeeze()) if s is not None else jnp.zeros(dofs)
    ṡ = jnp.atleast_1d(ṡ.squeeze()) if ṡ is not None else jnp.zeros(dofs)
    s̈ = jnp.atleast_1d(s̈.squeeze()) if s̈ is not None else jnp.zeros(dofs)
    τ = jnp.atleast_1d(τ.squeeze()) if τ is not None else jnp.zeros(dofs)
    W_vl_WB = jnp.atleast_1d(W_vl_WB.squeeze()) if W_vl_WB is not None else jnp.zeros(3)
    W_v̇l_WB = (
        jnp.atleast_1d(W_v̇l_WB.squeeze()) if W_v̇l_WB is not None else jnp.zeros(3)
    )
    W_p_B = jnp.atleast_1d(W_p_B.squeeze()) if W_p_B is not None else jnp.zeros(3)
    W_ω_WB = jnp.atleast_1d(W_ω_WB.squeeze()) if W_ω_WB is not None else jnp.zeros(3)
    W_ω̇_WB = jnp.atleast_1d(W_ω̇_WB.squeeze()) if W_ω̇_WB is not None else jnp.zeros(3)
    f = jnp.atleast_2d(f.squeeze()) if f is not None else jnp.zeros(shape=(nl, 6))
    W_Q_B = (
        jnp.atleast_1d(W_Q_B.squeeze())
        if W_Q_B is not None
        else jnp.array([1.0, 0, 0, 0])
    )
    standard_gravity = (
        jnp.array(standard_gravity).squeeze()
        if standard_gravity is not None
        else STANDARD_GRAVITY
    )

    if s.shape != (dofs,):
        raise ValueError(s.shape, dofs)

    if ṡ.shape != (dofs,):
        raise ValueError(ṡ.shape, dofs)

    if s̈.shape != (dofs,):
        raise ValueError(s̈.shape, dofs)

    if τ.shape != (dofs,):
        raise ValueError(τ.shape, dofs)

    if W_p_B.shape != (3,):
        raise ValueError(W_p_B.shape, (3,))

    if W_vl_WB.shape != (3,):
        raise ValueError(W_vl_WB.shape, (3,))

    if W_ω_WB.shape != (3,):
        raise ValueError(W_ω_WB.shape, (3,))

    if W_v̇l_WB.shape != (3,):
        raise ValueError(W_v̇l_WB.shape, (3,))

    if W_ω̇_WB.shape != (3,):
        raise ValueError(W_ω̇_WB.shape, (3,))

    if f.shape != (nl, 6):
        raise ValueError(f.shape, (nl, 6))

    if W_Q_B.shape != (4,):
        raise ValueError(W_Q_B.shape, (4,))

    # Check that the quaternion does not contain NaN values.
    exceptions.raise_value_error_if(
        condition=jnp.isnan(W_Q_B).any(),
        msg="A RBDA received a quaternion that contains NaN values.",
    )

    # Check that the quaternion is unary since our RBDAs make this assumption in order
    # to prevent introducing additional normalizations that would affect AD.
    exceptions.raise_value_error_if(
        condition=~jnp.allclose(W_Q_B.dot(W_Q_B), 1.0),
        msg="A RBDA received a quaternion that is not normalized.",
    )

    # Pack the 6D base velocity and acceleration.
    W_v_WB = jnp.hstack([W_vl_WB, W_ω_WB])
    W_v̇_WB = jnp.hstack([W_v̇l_WB, W_ω̇_WB])

    # Create the 6D gravity acceleration.
    W_g = jnp.array([0, 0, standard_gravity, 0, 0, 0])

    return (
        W_p_B.astype(float),
        W_Q_B.astype(float),
        s.astype(float),
        W_v_WB.astype(float),
        ṡ.astype(float),
        W_v̇_WB.astype(float),
        s̈.astype(float),
        τ.astype(float),
        f.astype(float),
        W_g.astype(float),
    )
