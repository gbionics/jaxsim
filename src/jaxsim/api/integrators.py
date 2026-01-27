import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.api.data import JaxSimModelData
from jaxsim.math import Skew


def semi_implicit_euler_integration(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    link_forces: jtp.Vector,
    joint_torques: jtp.Vector,
) -> JaxSimModelData:
    """Integrate the system state using the semi-implicit Euler method."""

    with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

        # Compute the system acceleration
        W_v̇_WB, s̈, contact_state_derivative = js.ode.system_acceleration(
            model=model,
            data=data,
            link_forces=link_forces,
            joint_torques=joint_torques,
        )

        dt = model.time_step

        # Compute the new generalized velocity.
        new_generalized_acceleration = jnp.hstack([W_v̇_WB, s̈])
        new_generalized_velocity = (
            data.generalized_velocity + dt * new_generalized_acceleration
        )

        # Extract the new base and joint velocities.
        W_v_B = new_generalized_velocity[0:6]
        ṡ = new_generalized_velocity[6:]

        # Compute the new base position and orientation.
        W_ω_WB = new_generalized_velocity[3:6]

        # To obtain the derivative of the base position, we need to subtract
        # the skew-symmetric matrix of the base angular velocity times the base position.
        # See: S. Traversaro and A. Saccon, “Multibody Dynamics Notation (Version 2), pg.9
        W_ṗ_B = new_generalized_velocity[0:3] + Skew.wedge(W_ω_WB) @ data.base_position

        W_Q̇_B = jaxsim.math.Quaternion.derivative(
            quaternion=data.base_orientation,
            omega=W_ω_WB,
        ).squeeze()

        W_p_B = data.base_position + dt * W_ṗ_B
        W_Q_B = data.base_orientation + dt * W_Q̇_B

        base_quaternion_norm = jaxsim.math.safe_norm(W_Q_B, axis=-1)

        W_Q_B = W_Q_B / jnp.where(base_quaternion_norm == 0, 1.0, base_quaternion_norm)

        s = data.joint_positions + dt * ṡ

        integrated_contact_state = jax.tree.map(
            lambda x, x_dot: x + dt * x_dot,
            data.contact_state,
            contact_state_derivative,
        )

    # TODO: Avoid double replace, e.g. by computing cached value here
    data = dataclasses.replace(
        data,
        _base_quaternion=W_Q_B,
        _base_position=W_p_B,
        _joint_positions=s,
        _joint_velocities=ṡ,
        _base_linear_velocity=W_v_B[0:3],
        _base_angular_velocity=W_ω_WB,
        contact_state=integrated_contact_state,
    )

    # Update the cached computations.
    data = data.replace(model=model)

    return data


def rk4_integration(
    model: js.model.JaxSimModel,
    data: JaxSimModelData,
    link_forces: jtp.Vector,
    joint_torques: jtp.Vector,
) -> JaxSimModelData:
    """Integrate the system state using the Runge-Kutta 4 method."""

    dt = model.time_step

    def f(x) -> dict[str, jtp.Matrix]:

        with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

            data_ti = data.replace(model=model, **x)

            return js.ode.system_dynamics(
                model=model,
                data=data_ti,
                link_forces=link_forces,
                joint_torques=joint_torques,
            )

    base_quaternion_norm = jaxsim.math.safe_norm(data._base_quaternion, axis=-1)
    base_quaternion = data._base_quaternion / jnp.where(
        base_quaternion_norm == 0, 1.0, base_quaternion_norm
    )

    x_t0 = dict(
        base_position=data._base_position,
        base_quaternion=base_quaternion,
        joint_positions=data._joint_positions,
        base_linear_velocity=data._base_linear_velocity,
        base_angular_velocity=data._base_angular_velocity,
        joint_velocities=data._joint_velocities,
        contact_state=data.contact_state,
    )

    euler_mid = lambda x, dxdt: x + (0.5 * dt) * dxdt
    euler_fin = lambda x, dxdt: x + dt * dxdt

    k1 = f(x_t0)
    k2 = f(jax.tree.map(euler_mid, x_t0, k1))
    k3 = f(jax.tree.map(euler_mid, x_t0, k2))
    k4 = f(jax.tree.map(euler_fin, x_t0, k3))

    # Average the slopes and compute the RK4 state derivative.
    average = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) / 6

    dxdt = jax.tree_util.tree_map(average, k1, k2, k3, k4)

    # Integrate the dynamics
    x_tf = jax.tree_util.tree_map(euler_fin, x_t0, dxdt)

    data_tf = dataclasses.replace(
        data,
        _base_position=x_tf["base_position"],
        _base_quaternion=x_tf["base_quaternion"],
        _joint_positions=x_tf["joint_positions"],
        _base_linear_velocity=x_tf["base_linear_velocity"],
        _base_angular_velocity=x_tf["base_angular_velocity"],
        _joint_velocities=x_tf["joint_velocities"],
        contact_state=x_tf["contact_state"],
    )

    return data_tf.replace(model=model)


def rk4fast_integration(
    model: js.model.JaxSimModel,
    data: JaxSimModelData,
    link_forces: jtp.Vector,
    joint_torques: jtp.Vector,
) -> JaxSimModelData:
    """
    Integrate the system state using the Runge-Kutta 4 fast method.

    Note:
        This method is a faster version of the RK4 method, but it may not be as accurate.
        It computes the contact forces only once at the beginning of the integration step.
    """

    dt = model.time_step

    if len(model.kin_dyn_parameters.contact_parameters.body) > 0:

        # Compute the 6D forces W_f ∈ ℝ^{n_L × 6} applied to links due to contact
        # with the terrain.
        W_f_L_terrain, contact_state_derivative = js.contact.link_contact_forces(
            model=model,
            data=data,
            link_forces=link_forces,
            joint_torques=joint_torques,
        )

    W_f_L_total = link_forces + W_f_L_terrain

    # Update the contact state data. This is necessary only for the contact models
    # that require propagation and integration of contact state.
    contact_state = model.contact_model.update_contact_state(contact_state_derivative)

    def f(x) -> dict[str, jtp.Matrix]:

        with data.switch_velocity_representation(jaxsim.VelRepr.Inertial):

            data_ti = data.replace(model=model, **x)

            W_v̇_WB, s̈ = js.model.forward_dynamics_aba(
                model=model,
                data=data_ti,
                joint_forces=joint_torques,
                link_forces=W_f_L_total,
            )

            W_ṗ_B, W_Q̇_B, ṡ = js.ode.system_position_dynamics(
                data=data,
                baumgarte_quaternion_regularization=1.0,
            )

        return dict(
            base_position=W_ṗ_B,
            base_quaternion=W_Q̇_B,
            joint_positions=ṡ,
            base_linear_velocity=W_v̇_WB[0:3],
            base_angular_velocity=W_v̇_WB[3:6],
            joint_velocities=s̈,
            # The contact state is not updated here, as it is assumed to be constant.
            contact_state=data_ti.contact_state,
        )

    base_quaternion_norm = jaxsim.math.safe_norm(data._base_quaternion, axis=-1)
    base_quaternion = data._base_quaternion / jnp.where(
        base_quaternion_norm == 0, 1.0, base_quaternion_norm
    )

    x_t0 = dict(
        base_position=data._base_position,
        base_quaternion=base_quaternion,
        joint_positions=data._joint_positions,
        base_linear_velocity=data._base_linear_velocity,
        base_angular_velocity=data._base_angular_velocity,
        joint_velocities=data._joint_velocities,
        contact_state=contact_state,
    )

    euler_mid = lambda x, dxdt: x + (0.5 * dt) * dxdt
    euler_fin = lambda x, dxdt: x + dt * dxdt

    k1 = f(x_t0)
    k2 = f(jax.tree.map(euler_mid, x_t0, k1))
    k3 = f(jax.tree.map(euler_mid, x_t0, k2))
    k4 = f(jax.tree.map(euler_fin, x_t0, k3))

    # Average the slopes and compute the RK4 state derivative.
    average = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) / 6

    dxdt = jax.tree_util.tree_map(average, k1, k2, k3, k4)

    # Integrate the dynamics
    x_tf = jax.tree_util.tree_map(euler_fin, x_t0, dxdt)

    data_tf = dataclasses.replace(
        data,
        _base_position=x_tf["base_position"],
        _base_quaternion=x_tf["base_quaternion"],
        _joint_positions=x_tf["joint_positions"],
        _base_linear_velocity=x_tf["base_linear_velocity"],
        _base_angular_velocity=x_tf["base_angular_velocity"],
        _joint_velocities=x_tf["joint_velocities"],
        contact_state=x_tf["contact_state"],
    )

    return data_tf.replace(model=model)


_INTEGRATORS_MAP: dict[
    js.model.IntegratorType, Callable[..., js.data.JaxSimModelData]
] = {
    js.model.IntegratorType.SemiImplicitEuler: semi_implicit_euler_integration,
    js.model.IntegratorType.RungeKutta4: rk4_integration,
    js.model.IntegratorType.RungeKutta4Fast: rk4fast_integration,
}
