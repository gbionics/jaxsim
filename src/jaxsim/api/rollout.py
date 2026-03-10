from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim.math import Quaternion, Skew
from jaxsim.config import PrecisionPolicy


_TARGETED_CHECKPOINT_POLICIES = frozenset({"contact", "dynamics", "contact_dynamics"})


def _as_state(
    state: js.data.JaxSimModelData | js.data.JaxSimModelState,
) -> js.data.JaxSimModelState:
    if isinstance(state, js.data.JaxSimModelData):
        return state.state
    if isinstance(state, js.data.JaxSimModelState):
        return state
    raise TypeError(type(state))


def _infer_horizon(
    *,
    controls: jtp.ArrayLike | None,
    link_forces: jtp.ArrayLike | None,
    joint_force_references: jtp.ArrayLike | None,
) -> int:
    for sequence in (controls, joint_force_references, link_forces):
        if sequence is not None:
            return int(jnp.asarray(sequence).shape[0])

    raise ValueError(
        "A rollout horizon could not be inferred. Provide `controls`, "
        "`joint_force_references`, or `link_forces` with a leading time dimension."
    )


def _has_constraints(model: js.model.JaxSimModel) -> bool:
    constraints = model.kin_dyn_parameters.constraints
    return constraints is not None and constraints.frame_idxs_1.shape[0] > 0


def _can_use_state_native_step(
    model: js.model.JaxSimModel,
    *,
    validate: bool,
) -> bool:
    return (
        not validate
        and model.integrator is js.model.IntegratorType.SemiImplicitEuler
        and isinstance(model.contact_model, jaxsim.rbda.contacts.SoftContacts)
        and not _has_constraints(model)
    )


def _fallback_step(
    model: js.model.JaxSimModel,
    carry_state: js.data.JaxSimModelState,
    *,
    link_force_t: jtp.ArrayLike,
    joint_force_reference_t: jtp.ArrayLike,
    contact_mode: str,
    precision_policy: PrecisionPolicy,
    validate: bool,
) -> js.data.JaxSimModelState:
    data_t = js.data.JaxSimModelData.from_state(
        model=model,
        state=carry_state,
        validate=validate,
    )
    data_tf = js.model.step(
        model=model,
        data=data_t,
        link_forces=link_force_t,
        joint_force_references=joint_force_reference_t,
        contact_mode=contact_mode,
        precision_policy=precision_policy,
    )
    return data_tf.state


def _compute_native_contact_forces(
    model: js.model.JaxSimModel,
    carry_state: js.data.JaxSimModelState,
    kinematics: js.data.JaxSimModelKinematicCache,
    *,
    contact_mode: str,
) -> tuple[jtp.Matrix, dict[str, jtp.PyTree]]:
    return jaxsim.rbda.contacts.SoftContacts.compute_contact_forces_from_kinematics(
        model=model,
        state=carry_state,
        kinematics=kinematics,
        contact_mode=contact_mode,
    )


def _compute_native_forward_dynamics(
    model: js.model.JaxSimModel,
    carry_state: js.data.JaxSimModelState,
    *,
    joint_forces: jtp.Vector,
    link_forces: jtp.Matrix,
) -> tuple[jtp.Vector, jtp.Vector]:
    return jaxsim.rbda.aba(
        model=model,
        base_position=carry_state.base_position,
        base_quaternion=carry_state.base_orientation,
        joint_positions=carry_state.joint_positions,
        base_linear_velocity=carry_state._base_linear_velocity,
        base_angular_velocity=carry_state._base_angular_velocity,
        joint_velocities=carry_state.joint_velocities,
        joint_forces=joint_forces,
        link_forces=link_forces,
        standard_gravity=model.gravity,
    )


def _state_native_step(
    model: js.model.JaxSimModel,
    carry_state: js.data.JaxSimModelState,
    *,
    link_force_t: jtp.ArrayLike,
    joint_force_reference_t: jtp.ArrayLike,
    contact_mode: str,
    checkpoint_policy: str | None = None,
) -> js.data.JaxSimModelState:
    kinematics = js.data.JaxSimModelKinematicCache.from_state(
        model=model,
        state=carry_state,
    )

    τ_total = js.actuation_model._compute_resultant_torques_from_state(
        model=model,
        joint_positions=carry_state.joint_positions,
        joint_velocities=carry_state.joint_velocities,
        joint_force_references=jnp.asarray(joint_force_reference_t),
    )

    W_f_L_external = js.model._active_link_forces_to_inertial(
        model=model,
        link_forces=link_force_t,
        velocity_representation=carry_state.velocity_representation,
        link_transforms=kinematics.link_transforms,
    )

    W_f_L_terrain = jnp.zeros_like(W_f_L_external)
    contact_state_derivative = carry_state.contact_state
    contact_force_fn = lambda state, cache: _compute_native_contact_forces(
        model=model,
        carry_state=state,
        kinematics=cache,
        contact_mode=contact_mode,
    )

    if checkpoint_policy in {"contact", "contact_dynamics"}:
        contact_force_fn = jax.checkpoint(contact_force_fn)

    if len(model.kin_dyn_parameters.contact_parameters.body) > 0:
        W_f_C, aux_dict = contact_force_fn(carry_state, kinematics)
        W_f_L_terrain = js.contact.link_forces_from_contact_forces(
            model=model,
            contact_forces=W_f_C,
            contact_mode=contact_mode,
        )
        contact_state_derivative = model.contact_model.update_contact_state(aux_dict)

    W_f_L_total = W_f_L_external + W_f_L_terrain

    forward_dynamics_fn = lambda state, joint_forces, link_forces: (
        _compute_native_forward_dynamics(
            model=model,
            carry_state=state,
            joint_forces=joint_forces,
            link_forces=link_forces,
        )
    )
    if checkpoint_policy in {"dynamics", "contact_dynamics"}:
        forward_dynamics_fn = jax.checkpoint(forward_dynamics_fn)

    W_v̇_WB, s̈ = forward_dynamics_fn(
        carry_state,
        joint_forces=τ_total,
        link_forces=W_f_L_total,
    )

    dt = model.time_step
    new_generalized_velocity = jnp.hstack(
        [
            carry_state._base_linear_velocity,
            carry_state._base_angular_velocity,
            carry_state.joint_velocities,
        ]
    ) + dt * jnp.hstack([W_v̇_WB, s̈])

    W_v_WB = new_generalized_velocity[0:6]
    ṡ = new_generalized_velocity[6:]
    W_ω_WB = W_v_WB[3:6]
    W_ṗ_B = W_v_WB[0:3] + Skew.wedge(W_ω_WB) @ carry_state.base_position
    W_Q̇_B = Quaternion.derivative(
        quaternion=carry_state.base_orientation,
        omega=W_ω_WB,
        omega_in_body_fixed=False,
    ).squeeze()

    integrated_contact_state = jax.tree.map(
        lambda x, x_dot: x + dt * x_dot,
        carry_state.contact_state,
        contact_state_derivative,
    )

    return dataclasses.replace(
        carry_state,
        _base_quaternion=jaxsim.math.normalize_quaternion(
            carry_state.base_orientation + dt * W_Q̇_B
        ),
        _base_position=carry_state.base_position + dt * W_ṗ_B,
        _joint_positions=carry_state.joint_positions + dt * ṡ,
        _joint_velocities=ṡ,
        _base_linear_velocity=W_v_WB[0:3],
        _base_angular_velocity=W_ω_WB,
        contact_state=integrated_contact_state,
    )


def rollout_scan(
    model: js.model.JaxSimModel,
    state: js.data.JaxSimModelData | js.data.JaxSimModelState,
    controls: jtp.ArrayLike | None = None,
    *,
    link_forces: jtp.ArrayLike | None = None,
    joint_force_references: jtp.ArrayLike | None = None,
    return_trajectory: bool = False,
    checkpoint_policy: str | None = None,
    precision_policy: PrecisionPolicy | None = None,
    contact_mode: str = "masked",
    validate: bool = False,
) -> js.data.JaxSimModelState | tuple[js.data.JaxSimModelState, js.data.JaxSimModelState]:
    """
    Roll out the simulator over a fixed horizon using `jax.lax.scan`.

    Args:
        model: The model to roll out.
        state: The initial state or cached data object.
        controls:
            Optional alias for `joint_force_references`. When provided, it must have
            shape `(T, dofs)`.
        link_forces:
            Optional sequence of link forces with shape `(T, n_links, 6)`.
        joint_force_references:
            Optional sequence of joint-force references with shape `(T, dofs)`.
        return_trajectory:
            If `True`, return the sequence of minimal states in addition to the final
            state. If `False`, avoid materializing trajectory outputs.
        checkpoint_policy:
            Optional rematerialization policy. Supported values are `None`,
            `"step"`, `"contact"`, `"dynamics"`, and `"contact_dynamics"`.
        precision_policy:
            Optional precision policy used by the rollout and contact solve path.
        contact_mode:
            Contact layout mode. `"masked"` is the TPU-oriented static-shape path.
        validate:
            Whether to validate reconstructed cached data during the rollout.

    Returns:
        The final minimal state, or a tuple of final state and state trajectory.
    """

    if controls is not None and joint_force_references is not None:
        raise ValueError(
            "Pass either `controls` or `joint_force_references`, not both."
        )

    precision_policy = PrecisionPolicy.resolve(precision_policy)
    state0 = _as_state(state)
    horizon = _infer_horizon(
        controls=controls,
        link_forces=link_forces,
        joint_force_references=joint_force_references,
    )

    joint_force_references = (
        controls if joint_force_references is None else joint_force_references
    )
    use_state_native_step = _can_use_state_native_step(model=model, validate=validate)

    if checkpoint_policy in _TARGETED_CHECKPOINT_POLICIES and not use_state_native_step:
        raise ValueError(
            "Targeted checkpoint policies require the native rollout step path."
        )

    joint_sequence = (
        precision_policy.cast(
            jnp.zeros((horizon, model.dofs())),
            kind="simulation",
        )
        if joint_force_references is None
        else precision_policy.cast(jnp.asarray(joint_force_references), kind="simulation")
    )
    link_force_sequence = (
        precision_policy.cast(
            jnp.zeros((horizon, model.number_of_links(), 6)),
            kind="simulation",
        )
        if link_forces is None
        else precision_policy.cast(jnp.asarray(link_forces), kind="simulation")
    )

    def advance(
        carry_state: js.data.JaxSimModelState,
        inputs: tuple[jtp.ArrayLike, jtp.ArrayLike],
    ) -> tuple[js.data.JaxSimModelState, js.data.JaxSimModelState | None]:
        joint_force_reference_t, link_force_t = inputs

        next_state = (
            _state_native_step(
                model=model,
                carry_state=carry_state,
                link_force_t=link_force_t,
                joint_force_reference_t=joint_force_reference_t,
                contact_mode=contact_mode,
                checkpoint_policy=(
                    None if checkpoint_policy == "step" else checkpoint_policy
                ),
            )
            if use_state_native_step
            else _fallback_step(
                model=model,
                carry_state=carry_state,
                link_force_t=link_force_t,
                joint_force_reference_t=joint_force_reference_t,
                contact_mode=contact_mode,
                precision_policy=precision_policy,
                validate=validate,
            )
        )

        return next_state, next_state if return_trajectory else None

    match checkpoint_policy:
        case None:
            scan_step = advance
        case "step":
            scan_step = jax.checkpoint(advance)
        case "contact" | "dynamics" | "contact_dynamics":
            scan_step = advance
        case _:
            raise ValueError(checkpoint_policy)

    final_state, trajectory = jax.lax.scan(
        scan_step,
        init=state0,
        xs=(joint_sequence, link_force_sequence),
    )

    return (final_state, trajectory) if return_trajectory else final_state
