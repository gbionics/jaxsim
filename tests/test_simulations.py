import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxsim.api as js
import jaxsim.rbda
import jaxsim.typing as jtp
from jaxsim import VelRepr
from jaxsim.api.kin_dyn_parameters import ConstraintType

from .utils import assert_allclose


def test_box_with_external_forces(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
):
    """
    Simulate a box falling due to gravity.

    We apply to its CoM a 6D force that balances exactly the gravitational force.
    The box should not fall.
    """

    model = jaxsim_model_box

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.5]),
        velocity_representation=velocity_representation,
    )

    # Compute the force due to gravity at the CoM.
    mg = -model.gravity * js.model.total_mass(model=model)
    G_f = jnp.array([0.0, 0.0, mg, 0, 0, 0])

    # Compute the position of the CoM expressed in the coordinates of the link frame L.
    L_p_CoM = js.link.com_position(
        model=model, data=data0, link_index=0, in_link_frame=True
    )

    # Compute the transform of 6D forces from the CoM to the link frame.
    L_H_G = jaxsim.math.Transform.from_quaternion_and_translation(translation=L_p_CoM)
    G_Xv_L = jaxsim.math.Adjoint.from_transform(transform=L_H_G, inverse=True)
    L_Xf_G = G_Xv_L.T
    L_f = L_Xf_G @ G_f

    # Initialize a references object that simplifies handling external forces.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data0,
        velocity_representation=velocity_representation,
    )

    # Apply a link forces to the base link.
    references = references.apply_link_forces(
        forces=jnp.atleast_2d(L_f),
        link_names=model.link_names()[0:1],
        model=model,
        data=data0,
        additive=False,
        input_representation=VelRepr.Body,
    )

    # Initialize the simulation horizon.
    tf = 0.5
    T_ns = jnp.arange(start=0, stop=tf * 1e9, step=model.time_step * 1e9, dtype=int)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T_ns:

        data = js.model.step(
            model=model,
            data=data,
            link_forces=references.link_forces(model, data),
        )

    # Check that the box didn't move.
    assert_allclose(data.base_position, data0.base_position)
    assert_allclose(data.base_orientation, data0.base_orientation)


def test_box_with_zero_gravity(
    jaxsim_model_box: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jnp.ndarray,
):

    model = jaxsim_model_box

    # Move the terrain (almost) infinitely far away from the box.
    with model.editable(validate=False) as model:
        model.terrain = jaxsim.terrain.FlatTerrain.build(height=-1e9)
        model.gravity = 0.0

    # Split the PRNG key.
    _, subkey = jax.random.split(prng_key, num=2)

    # Build the data of the model.
    data0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jax.random.uniform(subkey, shape=(3,)),
        velocity_representation=velocity_representation,
    )

    # Initialize a references object that simplifies handling external forces.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data0,
        velocity_representation=velocity_representation,
    )

    # Generate a random linear force.
    # We enforce them to be the same for all velocity representations so that
    # we can compare their outcomes.
    LW_f = 10.0 * (
        jax.random.uniform(jax.random.key(0), shape=(model.number_of_links(), 6))
        .at[:, 3:]
        .set(jnp.zeros(3))
    )

    # Note that the context manager does not switch back the newly created
    # `references` (that is not the yielded object) to the original representation.
    # In the simulation loop below, we need to make sure that we switch both `data`
    # and `references` to the same representation before extracting the information
    # passed to the step function.
    references = references.apply_link_forces(
        forces=jnp.atleast_2d(LW_f),
        link_names=model.link_names(),
        model=model,
        data=data0,
        additive=False,
        input_representation=VelRepr.Mixed,
    )

    tf = 0.01
    T = jnp.arange(start=0, stop=tf * 1e9, step=model.time_step * 1e9, dtype=int)

    # Copy the initial data...
    data = data0.copy()

    # ... and step the simulation.
    for _ in T:
        data = js.model.step(
            model=model,
            data=data,
            link_forces=references.link_forces(
                model=model, data=data, output_representation=velocity_representation
            ),
            output_representation=velocity_representation,
        )

    # Check that the box moved as expected.
    assert_allclose(
        data.base_position,
        data0.base_position
        + 0.5 * LW_f[:, :3].squeeze() / js.model.total_mass(model=model) * tf**2,
        atol=1e-3,
    )


def run_simulation(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    tf: jtp.FloatLike,
) -> js.data.JaxSimModelData:

    # Initialize the integration horizon.
    T_ns = jnp.arange(
        start=0.0, stop=int(tf * 1e9), step=int(model.time_step * 1e9)
    ).astype(int)

    # Initialize the simulation data.
    data = data_t0.copy()

    for _ in T_ns:

        data = js.model.step(
            model=model,
            data=data,
        )

    return data


def test_simulation_with_soft_contacts(
    jaxsim_model_box: js.model.JaxSimModel, integrator
):

    model = jaxsim_model_box

    # Define the maximum penetration of each collidable point at steady state.
    max_penetration = 0.001

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.SoftContacts.build()
        model.contact_params = js.contact.estimate_good_contact_parameters(
            model=model,
            number_of_active_collidable_points_steady_state=4,
            static_friction_coefficient=1.0,
            damping_ratio=1.0,
            max_penetration=max_penetration,
        )

        # Enable a subset of the collidable points.
        enabled_collidable_points_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_collidable_points_mask[[0, 1, 2, 3]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(
            enabled_collidable_points_mask.tolist()
        )

    assert np.sum(model.kin_dyn_parameters.contact_parameters.enabled) == 4

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, tf=1.0)

    assert_allclose(data_tf.base_position[0:2], data_t0.base_position[0:2])
    assert_allclose(data_tf.base_position[2] + max_penetration, box_height / 2)


def test_simulation_with_rigid_contacts(
    jaxsim_model_box: js.model.JaxSimModel, integrator
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        # In order to achieve almost no penetration, we need to use a fairly large
        # Baumgarte stabilization term.
        model.contact_model = jaxsim.rbda.contacts.RigidContacts.build(
            solver_options={"solver_tol": 1e-3}
        )
        model.contact_params = model.contact_model._parameters_class(K=1e5)

        # Enable a subset of the collidable points.
        enabled_collidable_points_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_collidable_points_mask[[0, 1, 2, 3]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(
            enabled_collidable_points_mask.tolist()
        )

    assert np.sum(model.kin_dyn_parameters.contact_parameters.enabled) == 4

    # Initialize the maximum penetration of each collidable point at steady state.
    # This model is rigid, so we expect (almost) no penetration.
    max_penetration = 0.000

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, tf=1.0)

    assert_allclose(data_tf.base_position[0:2], data_t0.base_position[0:2])
    assert_allclose(data_tf.base_position[2] + max_penetration, box_height / 2)


def test_simulation_with_relaxed_rigid_contacts(
    jaxsim_model_box: js.model.JaxSimModel, integrator
):

    model = jaxsim_model_box

    with model.editable(validate=False) as model:

        model.contact_model = jaxsim.rbda.contacts.RelaxedRigidContacts.build(
            solver_options={"tol": 1e-3},
        )
        model.contact_params = model.contact_model._parameters_class()

        # Enable a subset of the collidable points.
        enabled_collidable_points_mask = np.zeros(
            len(model.kin_dyn_parameters.contact_parameters.body), dtype=bool
        )
        enabled_collidable_points_mask[[0, 1, 2, 3]] = True
        model.kin_dyn_parameters.contact_parameters.enabled = tuple(
            enabled_collidable_points_mask.tolist()
        )
        model.integrator = integrator

    assert np.sum(model.kin_dyn_parameters.contact_parameters.enabled) == 4

    # Initialize the maximum penetration of each collidable point at steady state.
    # This model is quasi-rigid, so we expect (almost) no penetration.
    max_penetration = 0.000

    # Check jaxsim_model_box@conftest.py.
    box_height = 0.1

    # Build the data of the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, box_height * 2]),
        velocity_representation=VelRepr.Inertial,
    )

    # ===========================================
    # Run the simulation and test the final state
    # ===========================================

    data_tf = run_simulation(model=model, data_t0=data_t0, tf=1.0)

    # With this contact model, we need to slightly increase the tolerances.
    assert_allclose(data_tf.base_position[0:2], data_t0.base_position[0:2], atol=1e-5)
    assert_allclose(
        data_tf.base_position[2] + max_penetration, box_height / 2, atol=1e-4
    )


def test_joint_limits(
    jaxsim_model_single_pendulum: js.model.JaxSimModel,
):

    model = jaxsim_model_single_pendulum

    with model.editable(validate=False) as model:
        model.kin_dyn_parameters.joint_parameters.position_limits_max = jnp.atleast_1d(
            jnp.array(1.5708)
        )
        model.kin_dyn_parameters.joint_parameters.position_limits_min = jnp.atleast_1d(
            jnp.array(-1.5708)
        )
        model.kin_dyn_parameters.joint_parameters.position_limit_spring = (
            jnp.atleast_1d(jnp.array(75.0))
        )
        model.kin_dyn_parameters.joint_parameters.position_limit_damper = (
            jnp.atleast_1d(jnp.array(0.1))
        )

    position_limits_min, position_limits_max = js.joint.position_limits(model=model)

    data = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
    )

    theta = 10 * np.pi / 180

    # Define a tolerance since the spring-damper model does
    # not guarantee that the joint position will be exactly
    # below the limit.
    tolerance = theta * 0.10

    # Test minimum joint position limits.
    data_t0 = data.replace(model=model, joint_positions=position_limits_min - theta)

    model = model.replace(time_step=0.005, validate=False)
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=3.0)

    assert (
        np.min(np.array(data_tf.joint_positions), axis=0) + tolerance
        >= position_limits_min
    )

    # Test maximum joint position limits.
    data_t0 = data.replace(model=model, joint_positions=position_limits_max - theta)

    model = model.replace(time_step=0.001)
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=3.0)

    assert (
        np.max(np.array(data_tf.joint_positions), axis=0) - tolerance
        <= position_limits_max
    )


@pytest.mark.parametrize(
    "initial_joint_positions",
    [
        jnp.array([0, 0]),
        np.pi / 180 * jnp.array([5, 0]),
    ],
)
def test_simulation_with_kinematic_constraints_double_pendulum(
    jaxsim_model_double_pendulum: js.model.JaxSimModel,
    initial_joint_positions: jtp.Array,
):

    # ========
    # Arrange
    # ========

    tf = 1.0  # Final simulation time in seconds.

    model = jaxsim_model_double_pendulum

    frame_1_name = "right_link_extremity_frame"
    frame_2_name = "left_link_extremity_frame"
    frame_1_idx = js.frame.name_to_idx(model=model, frame_name=frame_1_name)
    frame_2_idx = js.frame.name_to_idx(model=model, frame_name=frame_2_name)

    # Define the kinematic constraints.
    constraints = js.kin_dyn_parameters.ConstraintMap()
    constraints = constraints.add_constraint(
        model=model,
        frame_idx_1=frame_1_idx,
        frame_idx_2=frame_2_idx,
        constraint_type=ConstraintType.Weld,
    )

    # Set the constraints in the model.
    with model.editable(validate=False) as model:
        model.kin_dyn_parameters.constraints = constraints
        model.gravity = 0.0

    # Build the initial data for the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
        joint_positions=initial_joint_positions,
    )

    # ====
    # Act
    # ====

    # Simulate the model for a given time and time step.
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=tf)

    # =========
    # Assert
    # =========

    # Assert that the chosen frames exist in the model
    assert frame_1_name in model.frame_names()
    assert frame_2_name in model.frame_names()

    # Assert that the joint positions are now equal
    actual_delta_s_tf = jnp.abs(data_tf.joint_positions[0] - data_tf.joint_positions[1])
    expected_delta_s_tf = 0.0

    assert_allclose(
        expected_delta_s_tf,
        actual_delta_s_tf,
        atol=1e-2,
        err_msg=f"Position difference [deg]: {actual_delta_s_tf * 180 / np.pi}",
    )


def test_simulation_with_kinematic_constraints_cartpole(
    jaxsim_model_cartpole: js.model.JaxSimModel,
):
    # ========
    # Arrange
    # ========

    tf = 1.0  # Final simulation time in seconds.

    model = jaxsim_model_cartpole

    frame_1_name = "cart_frame"
    frame_2_name = "rail_frame"
    frame_1_idx = js.frame.name_to_idx(model=model, frame_name=frame_1_name)
    frame_2_idx = js.frame.name_to_idx(model=model, frame_name=frame_2_name)

    # Define the kinematic constraints.
    constraints = js.kin_dyn_parameters.ConstraintMap()
    constraints = constraints.add_constraint(
        model,
        frame_1_idx,
        frame_2_idx,
        ConstraintType.Weld,
    )

    # Set the initial joint positions with the cart displaced from the rail zero position.
    initial_joint_positions = jnp.array([0.05, 0.0])

    # Set the constraints in the model.
    with model.editable(validate=False) as model:
        model.kin_dyn_parameters.constraints = constraints

    # Build the initial data for the model.
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
        joint_positions=initial_joint_positions,
    )

    # ====
    # Act
    # ====

    # Simulate the model for a given time and time step.
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=tf)

    H_frame1 = js.frame.transform(
        model=model,
        data=data_tf,
        frame_index=frame_1_idx,
    )
    H_frame2 = js.frame.transform(
        model=model,
        data=data_tf,
        frame_index=frame_2_idx,
    )

    # =========
    # Assert
    # =========

    # Assert that the chosen frames exist in the model
    assert frame_1_name in model.frame_names()
    assert frame_2_name in model.frame_names()

    # Assert that the two frames are in the same pose
    actual_frame_error = jnp.linalg.inv(H_frame1) @ H_frame2
    expected_frame_error = jnp.eye(4)

    assert_allclose(actual_frame_error, expected_frame_error, atol=1e-3)


def test_simulation_with_kinematic_constraints_4_bar_linkage(
    jaxsim_model_4_bar_linkage: js.model.JaxSimModel,
):
    """Test kinematic weld constraint on 4-bar linkage model."""

    # ========
    # Arrange
    # ========

    tf = 1.0  # Final simulation time in seconds.
    model = jaxsim_model_4_bar_linkage

    frame_1_name = "BC1_frame"
    frame_2_name = "BC2_frame"
    frame_1_idx = js.frame.name_to_idx(model=model, frame_name=frame_1_name)
    frame_2_idx = js.frame.name_to_idx(model=model, frame_name=frame_2_name)

    # Define the kinematic constraints.
    constraints = js.kin_dyn_parameters.ConstraintMap()
    constraints = constraints.add_constraint(
        model=model,
        frame_idx_1=frame_1_idx,
        frame_idx_2=frame_2_idx,
        constraint_type=ConstraintType.Weld,
        K_P=1e4,
    )

    # Set the constraints in the model.
    with model.editable(validate=False) as model:
        model.kin_dyn_parameters.constraints = constraints

    # Build the initial data for the model (default base pose is fine).
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=VelRepr.Inertial,
        base_position=jnp.array([0.0, 0.0, 0.10]),
    )

    # ====
    # Act
    # ====
    data_tf = run_simulation(model=model, data_t0=data_t0, tf=tf)

    H_frame1 = js.frame.transform(
        model=model,
        data=data_tf,
        frame_index=frame_1_idx,
    )
    H_frame2 = js.frame.transform(
        model=model,
        data=data_tf,
        frame_index=frame_2_idx,
    )

    # =========
    # Assert
    # =========
    assert frame_1_name in model.frame_names()
    assert frame_2_name in model.frame_names()

    # Position check
    pos1 = H_frame1[:3, 3]
    pos2 = H_frame2[:3, 3]
    assert_allclose(pos1, pos2, atol=1e-5)

    # Orientation check
    R1 = H_frame1[:3, :3]
    R2 = H_frame2[:3, :3]
    R_err = R1.T @ R2
    assert_allclose(R_err, jnp.eye(3), atol=1e-3)
