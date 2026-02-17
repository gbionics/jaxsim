import pathlib
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import pytest
import rod

import jaxsim.api as js
from jaxsim.api.kin_dyn_parameters import HwLinkMetadata, ScalingFactors
from jaxsim.rbda.contacts import SoftContactsParams

from .utils import assert_allclose


def test_update_hw_link_parameters(jaxsim_model_garpez: js.model.JaxSimModel):
    """
    Test that the hardware parameters of the model are updated correctly.
    """

    model = jaxsim_model_garpez

    # Store initial hardware parameters
    initial_metadata = model.kin_dyn_parameters.hw_link_metadata

    # Create the scaling factors
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [2.0, 1.5, 1.0],  # Scale x, y, z for link1
                [1.2, 1.0, 1.0],  # Scale r for link2
                [1.5, 0.8, 1.0],  # Scale r, l for link3
                [1.5, 1.0, 0.8],  # Scale x, y, z for link4
            ]
        ),
        density=jnp.ones(4),
    )

    # Update the model using the scaling factors
    updated_model = js.model.update_hw_parameters(model, scaling_parameters)

    # Compare updated hardware parameters
    for link_idx, link_name in enumerate(model.link_names()):
        updated_metadata = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx],
            updated_model.kin_dyn_parameters.hw_link_metadata,
        )
        initial_metadata_link = jax.tree_util.tree_map(
            lambda x, link_idx=link_idx: x[link_idx], initial_metadata
        )

        # TODO: Compute the 3D scaling vector
        # scale_vector = HwLinkMetadata._convert_scaling_to_3d_vector(
        #     initial_metadata_link.shape, scaling_parameters.dims[link_idx]
        # )

        expected_link_dimensions = (
            initial_metadata_link.geometry * scaling_parameters.dims[link_idx]
        )

        # Compare shape dimensions
        assert_allclose(
            updated_metadata.geometry,
            expected_link_dimensions,
            atol=1e-6,
            err_msg=f"Mismatch in dimensions for link {link_name}",
        )


@pytest.mark.parametrize(
    "jaxsim_model_garpez_scaled",
    [
        {
            "link1_scale": 4.0,
            "link2_scale": 3.0,
            "link3_scale": 2.0,
            "link4_scale": 1.5,
        }
    ],
    indirect=True,
)
def test_model_scaling_against_rod(
    jaxsim_model_garpez: js.model.JaxSimModel,
    jaxsim_model_garpez_scaled: js.model.JaxSimModel,
):
    """
    Test that scaling the HW parameters of JaxSim model matches the kin/dyn quantities of a JaxSim model obtained from a pre-scaled rod model.
    """

    # Define scaling parameters
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [4.0, 1.0, 1.0],  # Scale only x-dimension for link1
                [3.0, 1.0, 1.0],  # Scale only r-dimension for link2
                [1.0, 2.0, 1.0],  # Scale l dimension for link3
                [1.5, 1.0, 1.0],  # Scale only x-dimension for link4
            ]
        ),
        density=jnp.ones(4),
    )

    # Apply scaling to the original JaxSim model
    updated_model = js.model.update_hw_parameters(
        jaxsim_model_garpez, scaling_parameters
    )

    # Compare hardware parameters of the scaled JaxSim model with the pre-scaled JaxSim model
    scaled_metadata = updated_model.kin_dyn_parameters.hw_link_metadata

    pre_scaled_metadata = jaxsim_model_garpez_scaled.kin_dyn_parameters.hw_link_metadata

    # Compare shape dimensions
    assert_allclose(scaled_metadata.geometry, pre_scaled_metadata.geometry, atol=1e-6)

    # Compare mass
    scaled_mass, _ = HwLinkMetadata.compute_mass_and_inertia(scaled_metadata)
    pre_scaled_mass, _ = HwLinkMetadata.compute_mass_and_inertia(pre_scaled_metadata)

    assert_allclose(scaled_mass, pre_scaled_mass, atol=1e-6)

    # Compare inertia tensors
    _, scaled_inertia = HwLinkMetadata.compute_mass_and_inertia(scaled_metadata)
    _, pre_scaled_inertia = HwLinkMetadata.compute_mass_and_inertia(pre_scaled_metadata)

    assert_allclose(scaled_inertia, pre_scaled_inertia, atol=1e-6)

    # Compare transformations
    assert_allclose(scaled_metadata.L_H_G, pre_scaled_metadata.L_H_G, atol=1e-6)
    assert_allclose(scaled_metadata.L_H_vis, pre_scaled_metadata.L_H_vis, atol=1e-6)

    # Compare collidable points positions
    assert_allclose(
        jaxsim_model_garpez_scaled.kin_dyn_parameters.contact_parameters.point,
        updated_model.kin_dyn_parameters.contact_parameters.point,
        atol=1e-6,
    )


def test_update_hw_parameters_vmap(
    jaxsim_model_garpez: js.model.JaxSimModel,
):
    """
    Test that the hardware parameters of the model are updated correctly using vmap
    to create a set of n updated models.
    """

    model_nominal = jaxsim_model_garpez
    dofs = model_nominal.dofs()

    # Define a set of scaling factors for n models
    n = 10  # Number of updated models to create
    scaling_factors = [
        ScalingFactors(
            dims=(scale * jnp.ones((model_nominal.number_of_links(), 3))),
            density=(scale * jnp.ones(model_nominal.number_of_links())),
        )
        for scale in jnp.linspace(2.0, 2.0 + n - 1, n)
    ]

    # Convert the list of ScalingFactors to a JAX array of pytrees
    scaling_factors = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scaling_factors)

    # Generate a batch of updated models using vmap
    updated_models = jax.vmap(js.model.update_hw_parameters, in_axes=(None, 0))(
        model_nominal,
        scaling_factors,
    )

    def validate_model(updated_model):
        assert updated_model is not None

        # Compute forward kinematics for the "link3" link
        H_link3 = js.link.transform(
            model=updated_model,
            data=js.data.JaxSimModelData.build(model=updated_model),
            link_index=js.link.name_to_idx(model=updated_model, link_name="link3"),
        )

        # Compute the mass matrix
        M = js.model.free_floating_mass_matrix(
            model=updated_model,
            data=js.data.JaxSimModelData.build(model=updated_model),
        )

        assert H_link3 is not None
        assert H_link3.shape == (4, 4)
        assert M is not None
        assert isinstance(M, jnp.ndarray)
        assert M.shape == (6 + dofs, 6 + dofs)

    # Use vmap to validate all updated models
    jax.vmap(validate_model)(updated_models)


@pytest.mark.parametrize(
    "jaxsim_model_garpez_scaled",
    [
        {
            "link1_scale": 4.0,
            "link2_scale": 3.0,
            "link3_scale": 2.0,
            "link4_scale": 1.5,
        }
    ],
    indirect=True,
)
def test_export_updated_model(
    jaxsim_model_garpez: js.model.JaxSimModel,
    jaxsim_model_garpez_scaled: js.model.JaxSimModel,
):
    """
    Test the export of an updated model using JaxSimModel.export_updated_model.
    """

    model: js.model.JaxSimModel = jaxsim_model_garpez

    # Define scaling parameters
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [4.0, 1.0, 1.0],  # Scale x-dimension for link1
                [3.0, 1.0, 1.0],  # Scale r-dimension for link2
                [1.0, 2.0, 1.0],  # Scale l-dimension for link3
                [1.5, 1.0, 1.0],  # Scale x-dimension for link4
            ]
        ),
        density=jnp.ones(4),
    )
    identity_scaling = ScalingFactors(
        dims=jnp.ones((model.number_of_links(), 3)),
        density=jnp.ones(model.number_of_links()),
    )

    def get_link_by_name(model, name):
        try:
            return next(link for link in model.links() if link.name == name)
        except StopIteration as err:
            raise ValueError(
                f"Link '{name}' not found. Available links: {[l.name for l in model.links()]}"
            ) from err

    def compare_geometries(exported_link, ref_link, label=""):
        exported_geom = exported_link.visual.geometry.geometry()
        ref_geom = ref_link.visual.geometry.geometry()

        attrs = [attr for attr in vars(exported_geom) if hasattr(ref_geom, attr)]
        exported_vals = jnp.array([getattr(exported_geom, attr) for attr in attrs])
        ref_vals = jnp.array([getattr(ref_geom, attr) for attr in attrs])
        assert_allclose(
            exported_vals,
            ref_vals,
            err_msg=f"Geometry mismatch in {label} model.",
            atol=1e-6,
        )

    def compare_mass_and_inertia(exported_link, ref_link, label=""):
        assert_allclose(
            exported_link.inertial.mass,
            ref_link.inertial.mass,
            atol=1e-4,
            err_msg=f"Mass mismatch in {label} model.",
        )
        assert_allclose(
            exported_link.inertial.inertia.matrix(),
            ref_link.inertial.inertia.matrix(),
            atol=1e-4,
            err_msg=f"Inertia matrix mismatch in {label} model.",
        )

    def compare_collisions(exported_link, ref_link, label=""):
        geom_types = ["box", "sphere", "cylinder"]
        for geom_type in geom_types:
            exp_geom = getattr(exported_link.collision.geometry, geom_type)
            ref_geom = getattr(ref_link.collision.geometry, geom_type)
            if ref_geom is not None:
                if geom_type == "box":
                    assert_allclose(
                        jnp.array(exp_geom.size), jnp.array(ref_geom.size), atol=1e-6
                    )
                elif geom_type == "sphere":
                    assert_allclose(exp_geom.radius, ref_geom.radius, atol=1e-6)
                elif geom_type == "cylinder":
                    assert_allclose(exp_geom.radius, ref_geom.radius, atol=1e-6)
                    assert_allclose(exp_geom.length, ref_geom.length, atol=1e-6)
                return
        pytest.skip(
            f"Collision geometry type for link {exported_link.name} not supported."
        )

    def validate_model(updated_model, ref_model, label):

        urdf = updated_model.export_updated_model()
        assert isinstance(urdf, str), f"{label}: Exported URDF is not a string."

        exported_sdf = rod.Sdf.load(urdf, is_urdf=True)

        assert (
            len(exported_sdf.models()) == 1
        ), f"{label}: Exported model does not contain exactly one ROD model."

        exported_model = exported_sdf.models()[0]

        for link_name in model.link_names():

            exported_link = get_link_by_name(exported_model, link_name)
            ref_link = get_link_by_name(ref_model, link_name)

            compare_geometries(exported_link, ref_link, label=label)

            compare_mass_and_inertia(exported_link, ref_link, label=label)

            compare_collisions(exported_link, ref_link, label=label)

    # Test both scaled and identity-scaled updates
    for scaling, label in [
        (scaling_parameters, "SCALED"),
        (identity_scaling, "IDENTITY SCALED"),
    ]:
        # Load reference ROD model
        if label == "IDENTITY SCALED":
            ref_model = rod.Sdf.load(jaxsim_model_garpez.built_from).models()[0]
        else:
            ref_model = rod.Sdf.load(jaxsim_model_garpez_scaled.built_from).models()[0]

        updated_model = js.model.update_hw_parameters(model, scaling)
        validate_model(updated_model, ref_model, label)


def test_hw_parameters_optimization(jaxsim_model_garpez: js.model.JaxSimModel):
    """
    Test that updating hardware parameters allows optimizing the position of a link
    to match a target value along a specific world axis.
    """

    model = jaxsim_model_garpez
    data = js.data.JaxSimModelData.build(model=model)

    # Define the target height for the link.
    target_height = 3.0

    # Get the index of the link to optimize (e.g., "torso").
    link_idx = js.link.name_to_idx(model, link_name="link4")

    # Define the initial hardware parameters (scaling factors).
    initial_dims = jnp.ones(
        (model.number_of_links(), 3)
    )  # Initial dimensions (1.0 for all links).
    initial_density = jnp.ones(
        (model.number_of_links(),)
    )  # Initial density (1.0 for all links).
    scaling_factors = js.kin_dyn_parameters.ScalingFactors(
        dims=initial_dims, density=initial_density
    )

    # Define the loss function.
    def loss(scaling_factors):
        # Update the model with the new hardware parameters.
        updated_model = js.model.update_hw_parameters(
            model=model, scaling_factors=scaling_factors
        )

        # Compute forward kinematics for the link.
        W_H_L = js.model.forward_kinematics(model=updated_model, data=data)[link_idx]

        # Extract the height (z-axis position) of the link.
        link4_height = W_H_L[2, 3]  # Assuming z-axis is the third row.

        # Compute the loss as the squared difference from the target height.
        return (link4_height - target_height) ** 2

    # Compute the gradient of the loss function with respect to the scaling factors.
    loss_grad = jax.grad(loss)

    # Perform gradient descent.
    alpha = 0.01  # Learning rate.
    num_iterations = 1000  # Number of gradient descent steps.
    for _ in range(num_iterations):
        # Compute the gradient.
        grad_scaling_factors = loss_grad(scaling_factors)

        # Update the scaling factors.
        scaling_factors = js.kin_dyn_parameters.ScalingFactors(
            dims=scaling_factors.dims - alpha * grad_scaling_factors.dims,
            density=scaling_factors.density - alpha * grad_scaling_factors.density,
        )

        # Compute the current loss value.
        current_loss = loss(scaling_factors)

        # Optionally, print the progress.
        if _ % 100 == 0:
            print(f"Iteration {_}: Loss = {current_loss}")

    # Assert that the final loss is close to zero.
    assert current_loss < 1e-3, "Optimization did not converge to the target height."


def test_hw_parameters_collision_scaling(
    jaxsim_model_box: js.model.JaxSimModel, prng_key: jax.Array
):
    """
    Test that the collision elements of the model are updated correctly during the scaling of the model hw parameters.
    """

    _, subkey = jax.random.split(prng_key, num=2)

    # TODO: the jaxsim_model_box has an additional frame, which is handled wrongly
    # during the export of the updated model. For this reason, we recreate the model
    # from scratch here.
    del jaxsim_model_box

    import rod.builder.primitives

    # Create on-the-fly a ROD model of a box.
    rod_model = (
        rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=1.0, name="box")
        .build_model()
        .add_link(name="box_link")
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    model = js.model.JaxSimModel.build_from_model_description(
        model_description=rod_model
    )

    # Define the scaling factor for the model
    scaling_factor = 5.0

    # Recompute K and D, since the mass is scaled by scaling_factor^3
    # and the expected static compression of the terrain is approximately
    # proportional to mass/K and divided by the 4 contact points.
    K = model.contact_params.K * (scaling_factor**2)

    # Strongly overdamped, to avoid oscillations due to the high mass
    # and the low penetration allowed.
    D = 8 * jnp.sqrt(K)

    with model.editable(validate=False) as model:
        model.contact_params = SoftContactsParams(K=K, D=D)

    # Define the nominal radius of the sphere
    nominal_height = model.kin_dyn_parameters.hw_link_metadata.geometry[0, 2]

    # Define scaling parameters
    scaling_parameters = ScalingFactors(
        dims=jnp.ones((model.number_of_links(), 3)) * scaling_factor,
        density=jnp.array([1.0]),
    )

    # Update the model with the scaling parameters
    updated_model = js.model.update_hw_parameters(model, scaling_parameters)

    # Compute the expected height (nominal radius * scaling factor)
    expected_height = nominal_height * scaling_factor / 2

    # Simulate the box falling under gravity
    data = js.data.JaxSimModelData.build(
        model=updated_model,
        # Set the initial position of the box's base to be slightly above the ground
        # to allow it to settle at the expected height after scaling.
        # The base position is set to the nominal height of the box scaled by the scaling factor,
        # plus a small offset to avoid immediate collision with the ground.
        # This ensures that the box has enough space to fall and settle at the expected height.
        base_position=jnp.array(
            [
                *jax.random.uniform(subkey, shape=(2,)),
                expected_height + 0.05,
            ]
        ),
    )

    num_steps = 1000  # Number of simulation steps

    for _ in range(num_steps):
        data = js.model.step(
            model=updated_model,
            data=data,
        )

    # Get the final height of the box's base
    updated_base_height = data.base_position[2]

    # Assert that the box settles at the expected height
    assert jnp.isclose(
        updated_base_height, expected_height, atol=1e-3
    ), f"model base height mismatch: expected {expected_height}, got {updated_base_height}"


def test_unsupported_link_cases():
    """
    Test that unsupported link cases are handled correctly.
    """
    import rod.builder.primitives

    from jaxsim.api.kin_dyn_parameters import LinkParametrizableShape

    # Test unsupported (no visual)
    no_visual_model = js.model.JaxSimModel.build_from_model_description(
        rod.builder.primitives.BoxBuilder(x=1, y=1, z=1, mass=1, name="no_vis_box")
        .build_model()
        .add_link(name="no_visual_link")
        .add_inertial()
        .build()  # No .add_visual()
    )
    no_visual_metadata = no_visual_model.kin_dyn_parameters.hw_link_metadata
    empty_metadata = HwLinkMetadata.empty()
    comparison = jax.tree.map(
        lambda a, b: jnp.allclose(a, b),
        no_visual_metadata,
        empty_metadata,
    )
    assert jax.tree.reduce(
        lambda acc, value: acc and bool(value), comparison, True
    ), "No links should be supported."

    # Create a simple multi-link URDF and add collision to ensure links are kept
    multi_link_urdf = """
        <?xml version="1.0"?>
        <robot name="two_link_test">

        <!-- Link 1: Supported (with box visual) -->
        <link name="supported_link">
            <inertial>
            <mass value="1.0"/>
            <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
            <visual>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
            </visual>
            <collision>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
            </collision>
        </link>

        <!-- Link 2: Unsupported (no visual but has collision) -->
        <link name="unsupported_link">
            <inertial>
            <mass value="1.0"/>
            <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
            <!-- No visual element - this makes it unsupported -->
            <collision>
            <geometry>
                <box size="0.5 0.5 0.5"/>
            </geometry>
            </collision>
        </link>

        <!-- Link 3: Two visuals (first should be picked) -->
        <link name="double_visual_link">
            <inertial>
            <mass value="1.0"/>
            <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
            <visual name="primary_sphere">
            <geometry>
                <sphere radius="0.4"/>
            </geometry>
            </visual>
            <visual name="secondary_box">
            <geometry>
                <box size="0.8 0.2 0.2"/>
            </geometry>
            </visual>
            <collision>
            <geometry>
                <sphere radius="0.4"/>
            </geometry>
            </collision>
        </link>

        <!-- Joint connecting the links -->
        <joint name="connecting_joint" type="revolute">
            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
            <parent link="supported_link"/>
            <child link="unsupported_link"/>
            <axis xyz="1 0 0"/>
            <limit effort="3.4028235e+38" velocity="3.4028235e+38"/>
        </joint>

        <!-- Joint for double visual link -->
        <joint name="double_visual_joint" type="revolute">
            <origin xyz="0.1 0.0 0.0" rpy="0.0 0.0 0.0"/>
            <parent link="unsupported_link"/>
            <child link="double_visual_link"/>
            <axis xyz="0 1 0"/>
            <limit effort="3.4028235e+38" velocity="3.4028235e+38"/>
        </joint>

        </robot>
    """

    # Build JaxSim model from the URDF
    multi_link_model = js.model.JaxSimModel.build_from_model_description(
        multi_link_urdf, is_urdf=True
    )
    multi_link_metadata = multi_link_model.kin_dyn_parameters.hw_link_metadata

    # Verify array consistency for the model
    num_links = multi_link_model.number_of_links()
    assert num_links == 3, f"Expected 3 links in the URDF model, got {num_links}"
    assert (
        len(multi_link_metadata.link_shape)
        == len(multi_link_metadata.geometry)
        == len(multi_link_metadata.density)
        == num_links
    )

    # Count verification in single model
    supported_count = sum(
        1
        for s in multi_link_metadata.link_shape
        if s != LinkParametrizableShape.Unsupported
    )
    unsupported_count = sum(
        1
        for s in multi_link_metadata.link_shape
        if s == LinkParametrizableShape.Unsupported
    )

    assert (
        supported_count == 2
    ), f"Expected 2 supported links in single model, got {supported_count}"
    assert (
        unsupported_count == 1
    ), f"Expected 1 unsupported link in single model, got {unsupported_count}"

    # Ensure shapes match expectations by name
    link_indices = {name: idx for idx, name in enumerate(multi_link_model.link_names())}

    assert (
        multi_link_metadata.link_shape[link_indices["supported_link"]]
        == LinkParametrizableShape.Box
    ), "Supported link should remain a box"
    assert (
        multi_link_metadata.link_shape[link_indices["unsupported_link"]]
        == LinkParametrizableShape.Unsupported
    ), "Unsupported link should remain unsupported"

    double_visual_idx = link_indices["double_visual_link"]
    assert (
        multi_link_metadata.link_shape[double_visual_idx]
        == LinkParametrizableShape.Sphere
    ), "Double visual link should pick the first (sphere) visual"
    assert_allclose(
        multi_link_metadata.geometry[double_visual_idx, 0],
        0.4,
        err_msg="Sphere radius must match the first visual",
    )

    # Test selective parametrization: only 'supported_link' and 'double_visual_link' should be parametrized
    selective_model = js.model.JaxSimModel.build_from_model_description(
        multi_link_urdf, is_urdf=True, parametrized_links=("double_visual_link")
    )
    selective_metadata = selective_model.kin_dyn_parameters.hw_link_metadata

    # Check that only the selected links are parametrized
    link_indices = {name: idx for idx, name in enumerate(selective_model.link_names())}
    assert (
        selective_metadata.link_shape[link_indices["supported_link"]]
        == LinkParametrizableShape.Unsupported
    ), "Selected supported_link should be parametrized as Box"
    assert (
        selective_metadata.link_shape[link_indices["double_visual_link"]]
        == LinkParametrizableShape.Sphere
    ), "Selected double_visual_link should be parametrized as Sphere"
    assert (
        selective_metadata.link_shape[link_indices["unsupported_link"]]
        == LinkParametrizableShape.Unsupported
    ), "Non-selected unsupported_link should be marked as Unsupported"


def test_export_continuous_joint_handling():
    """
    Test that continuous joints are correctly exported with type="continuous"
    and without position limits, while preserving effort and velocity limits.
    """

    # Load cartpole model which has a continuous joint (pivot)
    cartpole_path = (
        pathlib.Path(__file__).parent.parent / "examples" / "assets" / "cartpole.urdf"
    )
    model = js.model.JaxSimModel.build_from_model_description(cartpole_path)

    # Define some simple scaling parameters (identity scaling)
    scaling_parameters = ScalingFactors(
        dims=jnp.ones((model.number_of_links(), 3)),
        density=jnp.ones(model.number_of_links()),
    )

    # Update the model with scaling parameters
    updated_model = js.model.update_hw_parameters(model, scaling_parameters)

    # Export the updated model
    exported_urdf = updated_model.export_updated_model()

    # Parse the URDF XML directly (not through rod, which would convert continuous back to revolute)
    root = ET.fromstring(exported_urdf)

    # Find the pivot joint (continuous joint)
    pivot_joint = None
    for joint_elem in root.findall(".//joint"):
        if joint_elem.get("name") == "pivot":
            pivot_joint = joint_elem
            break

    assert pivot_joint is not None, "pivot joint should exist in exported model"

    # Verify that the joint type is "continuous"
    assert (
        pivot_joint.get("type") == "continuous"
    ), f"pivot joint should have type='continuous', got '{pivot_joint.get('type')}'"

    # Verify that position limits are not present for continuous joints
    limit_elem = pivot_joint.find("limit")
    assert limit_elem is not None, "pivot joint should have limits element"

    assert (
        limit_elem.get("lower") is None
    ), f"continuous joint should not have lower position limit, got {limit_elem.get('lower')}"
    assert (
        limit_elem.get("upper") is None
    ), f"continuous joint should not have upper position limit, got {limit_elem.get('upper')}"

    # Verify that effort and velocity limits are preserved
    assert (
        limit_elem.get("effort") is not None
    ), "continuous joint should preserve effort limit"
    assert (
        limit_elem.get("velocity") is not None
    ), "continuous joint should preserve velocity limit"

    # Verify that the linear joint (prismatic) is NOT changed to continuous
    linear_joint = None
    for joint_elem in root.findall(".//joint"):
        if joint_elem.get("name") == "linear":
            linear_joint = joint_elem
            break

    assert linear_joint is not None, "linear joint should exist in exported model"
    assert (
        linear_joint.get("type") == "prismatic"
    ), f"linear joint should remain prismatic, got '{linear_joint.get('type')}'"

    # Prismatic joint should keep its limits
    linear_limit = linear_joint.find("limit")
    assert linear_limit is not None, "prismatic joint should have limits"
    assert (
        linear_limit.get("lower") is not None
    ), "prismatic joint should have lower limit"
    assert (
        linear_limit.get("upper") is not None
    ), "prismatic joint should have upper limit"


def test_export_model_with_missing_collision(
    jaxsim_model_missing_collision: js.model.JaxSimModel,
):
    """
    Test that export_updated_model() works correctly when a link has a visual
    but is missing a collision element.

    This validates the skip logic that handles None collision elements.
    """

    model = jaxsim_model_missing_collision

    # Define scaling parameters to modify the model
    scaling_parameters = ScalingFactors(
        dims=jnp.array(
            [
                [1.5, 1.0, 1.0],  # Scale x-dimension for link1 (has collision)
                [2.0, 1.0, 1.0],  # Scale radius for link2 (missing collision)
            ]
        ),
        density=jnp.ones(2),
    )

    # Update the model with scaling parameters
    updated_model = js.model.update_hw_parameters(model, scaling_parameters)

    # Export the updated model - this should NOT fail even though link2 is missing collision
    exported_urdf = updated_model.export_updated_model()

    # Verify basic structure of exported URDF
    assert isinstance(exported_urdf, str), "Exported URDF should be a string"
    assert len(exported_urdf) > 0, "Exported URDF should not be empty"

    # Parse the exported URDF to verify it's valid XML
    root = ET.fromstring(exported_urdf)
    assert root.tag == "robot", "Root element should be 'robot'"

    # Find both links in the exported model
    links = {link.get("name"): link for link in root.findall(".//link")}
    assert "link1" in links, "link1 should exist in exported model"
    assert "link2" in links, "link2 should exist in exported model"

    # Verify link1 has both visual and collision
    link1 = links["link1"]
    link1_visual = link1.find("visual")
    link1_collision = link1.find("collision")
    assert link1_visual is not None, "link1 should have a visual element"
    assert link1_collision is not None, "link1 should have a collision element"

    # Verify link1's geometry was updated
    link1_visual_box = link1_visual.find(".//box")
    assert link1_visual_box is not None, "link1 visual should have box geometry"
    link1_size = [float(x) for x in link1_visual_box.get("size").split()]
    # First dimension should be scaled by 1.5
    assert_allclose(
        link1_size[0],
        0.3 * 1.5,
        atol=1e-6,
        err_msg="link1 x-dimension should be scaled",
    )

    # Verify link2 has visual but no collision
    link2 = links["link2"]
    link2_visual = link2.find("visual")
    link2_collision = link2.find("collision")
    assert link2_visual is not None, "link2 should have a visual element"
    assert link2_collision is None, "link2 should NOT have a collision element"

    # Verify link2's visual geometry was updated despite missing collision
    link2_visual_sphere = link2_visual.find(".//sphere")
    assert link2_visual_sphere is not None, "link2 visual should have sphere geometry"
    link2_radius = float(link2_visual_sphere.get("radius"))
    # Radius should be scaled by 2.0
    assert_allclose(
        link2_radius,
        0.1 * 2.0,
        atol=1e-6,
        err_msg="link2 radius should be scaled despite missing collision",
    )

    # Load the exported model to verify it can be parsed correctly
    exported_sdf = rod.Sdf.load(exported_urdf, is_urdf=True)
    assert (
        len(exported_sdf.models()) == 1
    ), "Exported model should contain exactly one ROD model"

    exported_model = exported_sdf.models()[0]
    assert exported_model.name == model.name(), "Exported model name should match"

    # Verify we can build a JaxSim model from the exported URDF
    reimported_jaxsim_model = js.model.JaxSimModel.build_from_model_description(
        model_description=exported_urdf, is_urdf=True
    )
    assert (
        reimported_jaxsim_model is not None
    ), "Should be able to build model from exported URDF"
    assert (
        reimported_jaxsim_model.number_of_links() == model.number_of_links()
    ), "Reimported model should have same number of links"
