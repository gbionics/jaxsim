import jax
import jax.numpy as jnp
import numpy as np

import jaxsim
import jaxsim.api as js
from jaxsim import VelRepr
from jaxsim.config import PrecisionPolicy

from .utils import assert_allclose


def _build_ballistic_box(model: js.model.JaxSimModel) -> js.model.JaxSimModel:
    with model.editable(validate=False) as edited_model:
        edited_model.terrain = jaxsim.terrain.FlatTerrain.build(height=-1e9)
        edited_model.gravity = 0.0
        edited_model.integrator = js.model.IntegratorType.SemiImplicitEuler

    return edited_model


def _build_contact_box(model: js.model.JaxSimModel) -> js.model.JaxSimModel:
    with model.editable(validate=False) as edited_model:
        edited_model.integrator = js.model.IntegratorType.SemiImplicitEuler

    return edited_model


def _configure_relaxed_rigid_box(
    model: js.model.JaxSimModel,
    *,
    solver_mode: str,
) -> js.model.JaxSimModel:
    with model.editable(validate=False) as edited_model:
        edited_model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts.build(
            solver_mode=solver_mode,
            solver_options={"tol": 1e-6, "maxiter": 8},
        )
        edited_model.contact_params = edited_model.contact_model._parameters_class()

        enabled_mask = np.zeros(
            len(edited_model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_mask[[0, 1, 2, 3]] = True
        edited_model.kin_dyn_parameters.contact_parameters.enabled = tuple(
            enabled_mask.tolist()
        )

    return edited_model


def test_data_state_roundtrip(
    jaxsim_model_box: js.model.JaxSimModel,
    prng_key: jax.Array,
):
    model = _build_ballistic_box(jaxsim_model_box)
    data = js.data.random_model_data(
        model=model,
        key=prng_key,
        velocity_representation=VelRepr.Mixed,
    )

    rebuilt = js.data.JaxSimModelData.from_state(
        model=model,
        state=data.state,
        validate=True,
    )

    assert_allclose(rebuilt.base_position, data.base_position)
    assert_allclose(rebuilt.base_velocity, data.base_velocity)
    assert_allclose(rebuilt.joint_positions, data.joint_positions)
    assert_allclose(rebuilt.joint_velocities, data.joint_velocities)
    assert_allclose(rebuilt._link_transforms, data._link_transforms)
    assert_allclose(rebuilt._link_velocities, data._link_velocities)


def test_rollout_scan_matches_step_loop(
    jaxsim_model_box: js.model.JaxSimModel,
):
    model = _build_ballistic_box(jaxsim_model_box)
    policy = PrecisionPolicy.safe_default()

    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.1, -0.2, 0.3]),
        base_linear_velocity=jnp.array([0.3, 0.2, -0.1]),
        base_angular_velocity=jnp.array([0.0, 0.0, 0.1]),
        velocity_representation=VelRepr.Inertial,
    )

    horizon = 6
    link_force_sequence = jnp.linspace(
        jnp.zeros((model.number_of_links(), 6)),
        jnp.ones((model.number_of_links(), 6)),
        num=horizon,
    )
    controls = jnp.zeros((horizon, model.dofs()))

    data_tf = data_t0
    for step_idx in range(horizon):
        data_tf = js.model.step(
            model=model,
            data=data_tf,
            link_forces=link_force_sequence[step_idx],
            joint_force_references=controls[step_idx],
            contact_mode="masked",
            precision_policy=policy,
        )

    rollout_tf = js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=controls,
        link_forces=link_force_sequence,
        contact_mode="masked",
        precision_policy=policy,
    )

    assert_allclose(rollout_tf.base_position, data_tf.base_position)
    assert_allclose(rollout_tf.base_velocity, data_tf.base_velocity)
    assert_allclose(rollout_tf.joint_positions, data_tf.joint_positions)
    assert_allclose(rollout_tf.joint_velocities, data_tf.joint_velocities)


def test_rollout_scan_returns_trajectory(
    jaxsim_model_box: js.model.JaxSimModel,
):
    model = _build_ballistic_box(jaxsim_model_box)
    horizon = 4

    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.4]),
        velocity_representation=VelRepr.Inertial,
    )

    final_state, trajectory = js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=jnp.zeros((horizon, model.dofs())),
        link_forces=jnp.zeros((horizon, model.number_of_links(), 6)),
        contact_mode="masked",
        return_trajectory=True,
        checkpoint_policy="step",
    )

    assert trajectory._base_position.shape == (horizon, 3)
    assert trajectory._base_quaternion.shape == (horizon, 4)
    assert trajectory._joint_positions.shape == (horizon, model.dofs())
    assert_allclose(final_state.base_position, trajectory.base_position[-1])
    assert_allclose(final_state.base_velocity, trajectory.base_velocity[-1])


def test_rollout_scan_targeted_checkpoint_policies_match_contact_baseline(
    jaxsim_model_box: js.model.JaxSimModel,
):
    model = _build_contact_box(jaxsim_model_box)
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.04]),
        velocity_representation=VelRepr.Inertial,
    )

    horizon = 4
    controls = jnp.zeros((horizon, model.dofs()))
    link_forces = jnp.zeros((horizon, model.number_of_links(), 6))

    baseline = js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=controls,
        link_forces=link_forces,
        contact_mode="masked",
        checkpoint_policy=None,
    )

    for checkpoint_policy in ("contact", "dynamics", "contact_dynamics"):
        result = js.rollout.rollout_scan(
            model=model,
            state=data_t0,
            controls=controls,
            link_forces=link_forces,
            contact_mode="masked",
            checkpoint_policy=checkpoint_policy,
        )

        assert_allclose(result.base_position, baseline.base_position)
        assert_allclose(result.base_velocity, baseline.base_velocity)
        assert_allclose(
            result.contact_state["tangential_deformation"],
            baseline.contact_state["tangential_deformation"],
        )


def test_rollout_scan_matches_step_loop_in_mixed_representation(
    jaxsim_model_box: js.model.JaxSimModel,
):
    model = _build_ballistic_box(jaxsim_model_box)
    policy = PrecisionPolicy.safe_default()

    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.05, -0.1, 0.25]),
        base_linear_velocity=jnp.array([0.2, -0.05, 0.1]),
        base_angular_velocity=jnp.array([0.0, 0.1, -0.2]),
        velocity_representation=VelRepr.Mixed,
    )

    horizon = 5
    link_force_sequence = jnp.linspace(
        jnp.zeros((model.number_of_links(), 6)),
        jnp.array([[1.0, -0.5, 0.25, 0.1, -0.2, 0.3]]),
        num=horizon,
    )
    controls = jnp.zeros((horizon, model.dofs()))

    data_tf = data_t0
    for step_idx in range(horizon):
        data_tf = js.model.step(
            model=model,
            data=data_tf,
            link_forces=link_force_sequence[step_idx],
            joint_force_references=controls[step_idx],
            contact_mode="masked",
            precision_policy=policy,
        )

    rollout_tf = js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=controls,
        link_forces=link_force_sequence,
        contact_mode="masked",
        precision_policy=policy,
    )

    assert_allclose(rollout_tf.base_position, data_tf.base_position)
    assert_allclose(rollout_tf.base_velocity, data_tf.base_velocity)
    assert_allclose(rollout_tf.joint_positions, data_tf.joint_positions)
    assert_allclose(rollout_tf.joint_velocities, data_tf.joint_velocities)


def test_masked_contact_layout_preserves_full_shape(
    jaxsim_model_box: js.model.JaxSimModel,
):
    with jaxsim_model_box.editable(validate=False) as model:
        enabled_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_mask[[0, 1]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(enabled_mask.tolist())

    data = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.2]),
        velocity_representation=VelRepr.Inertial,
    )

    W_p_enabled, _ = js.contact.collidable_point_kinematics(
        model=model,
        data=data,
        contact_mode="enabled",
    )
    W_p_masked, W_v_masked = js.contact.collidable_point_kinematics(
        model=model,
        data=data,
        contact_mode="masked",
    )
    mask = js.contact.collidable_point_enabled_mask(model=model, contact_mode="masked")

    assert W_p_enabled.shape[0] == int(enabled_mask.sum())
    assert W_p_masked.shape[0] == len(enabled_mask)
    assert np.array_equal(np.array(mask), enabled_mask)
    assert_allclose(W_p_masked[np.array(enabled_mask)], W_p_enabled)
    assert_allclose(
        W_v_masked[np.logical_not(np.array(enabled_mask))],
        jnp.zeros((np.logical_not(enabled_mask).sum(), 3)),
    )


def test_link_forces_from_contact_forces_matches_parent_accumulation(
    jaxsim_model_box: js.model.JaxSimModel,
):
    with jaxsim_model_box.editable(validate=False) as model:
        enabled_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_mask[[0, 1, 3, 5]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(enabled_mask.tolist())

    parent_indices = np.array(model.kin_dyn_parameters.contact_parameters.body)
    masked_contact_forces = jnp.arange(
        len(parent_indices) * 6, dtype=float
    ).reshape(len(parent_indices), 6)

    masked_link_forces = js.contact.link_forces_from_contact_forces(
        model=model,
        contact_forces=masked_contact_forces,
        contact_mode="masked",
    )
    enabled_link_forces = js.contact.link_forces_from_contact_forces(
        model=model,
        contact_forces=masked_contact_forces[np.array(enabled_mask)],
        contact_mode="enabled",
    )

    expected_masked = np.zeros((model.number_of_links(), 6))
    expected_enabled = np.zeros((model.number_of_links(), 6))

    for point_idx, parent_idx in enumerate(parent_indices):
        expected_masked[parent_idx] += np.asarray(masked_contact_forces[point_idx])
        if enabled_mask[point_idx]:
            expected_enabled[parent_idx] += np.asarray(masked_contact_forces[point_idx])

    assert_allclose(masked_link_forces, expected_masked)
    assert_allclose(enabled_link_forces, expected_enabled)


def test_relaxed_rigid_fixed_iterations_matches_enabled_layout(
    jaxsim_model_box: js.model.JaxSimModel,
):
    adaptive_model = _configure_relaxed_rigid_box(
        jaxsim_model_box, solver_mode="adaptive"
    )
    fixed_model = _configure_relaxed_rigid_box(
        jaxsim_model_box, solver_mode="fixed_iterations"
    )

    adaptive_data = js.data.JaxSimModelData.build(
        model=adaptive_model,
        base_position=jnp.array([0.0, 0.0, 0.04]),
        velocity_representation=VelRepr.Inertial,
    )
    fixed_data = js.data.JaxSimModelData.build(
        model=fixed_model,
        base_position=jnp.array([0.0, 0.0, 0.04]),
        velocity_representation=VelRepr.Inertial,
    )

    adaptive_link_forces, _ = js.contact.link_contact_forces(
        model=adaptive_model,
        data=adaptive_data,
        contact_mode="enabled",
    )
    fixed_link_forces, _ = js.contact.link_contact_forces(
        model=fixed_model,
        data=fixed_data,
        contact_mode="masked",
        precision_policy=PrecisionPolicy.tpu_fast(),
    )

    assert jnp.all(jnp.isfinite(fixed_link_forces))
    assert_allclose(fixed_link_forces, adaptive_link_forces, atol=1e-2, rtol=1e-2)


def test_safe_math_and_precision_policy():
    vector_grad = jax.grad(lambda x: jnp.sum(jaxsim.math.safe_normalize(x)[0]))(
        jnp.zeros(3)
    )

    assert jnp.all(jnp.isfinite(vector_grad))
    assert_allclose(
        jaxsim.math.normalize_quaternion(jnp.zeros(4)),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
    )

    default_policy = PrecisionPolicy.resolve(None)
    assert default_policy == PrecisionPolicy.safe_default()
    assert PrecisionPolicy.tpu_fast().simulation_dtype == jnp.float32
    assert PrecisionPolicy.reference().matmul_precision == jax.lax.Precision.HIGHEST
