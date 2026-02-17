import os

os.environ["JAXSIM_ENABLE_EXCEPTIONS"] = "1"

import pathlib
import subprocess

import jax
import numpy as np
import pytest
import rod
import rod.urdf.exporter

import jaxsim
import jaxsim.api as js
from jaxsim.api.model import IntegratorType


def pytest_addoption(parser):
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Run tests only if GPU is available and utilized",
    )

    parser.addoption(
        "--batch-size",
        action="store",
        default="None",
        help="Batch size for vectorized benchmarks (only applies to benchmark tests)",
    )


def pytest_generate_tests(metafunc):
    if (
        "batch_size" in metafunc.fixturenames
        and (batch_size := metafunc.config.getoption("--batch-size")) != "None"
    ):
        metafunc.parametrize("batch_size", [1, int(batch_size)])


def check_gpu_usage():
    # Set environment variable to prioritize GPU.
    os.environ["JAX_PLATFORM_NAME"] = "gpu"

    # Run a simple JAX operation
    x = jax.device_put(jax.numpy.ones((512, 512)))
    y = jax.device_put(jax.numpy.ones((512, 512)))
    _ = jax.numpy.dot(x, y).block_until_ready()

    # Check GPU memory usage with nvidia-smi.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.exit(
            "Failed to query GPU usage. Ensure nvidia-smi is installed and accessible."
        )

    gpu_memory_usage = [
        int(line.strip().split()[0]) for line in result.stdout.splitlines()
    ]
    if all(usage == 0 for usage in gpu_memory_usage):
        pytest.exit(
            "GPU is available but not utilized during computations. Check your JAX installation."
        )


def pytest_configure(config) -> None:
    """Pytest configuration hook."""

    # This is a global variable that is updated by the `prng_key` fixture.
    pytest.prng_key = jax.random.PRNGKey(
        seed=int(os.environ.get("JAXSIM_TEST_SEED", 0))
    )

    # Check if GPU is available and utilized.
    if config.getoption("--gpu-only"):
        devices = jax.devices()
        if not any(device.platform == "gpu" for device in devices):
            pytest.exit("No GPU devices found. Check your JAX installation.")

        # Ensure GPU is being used during computation
        check_gpu_usage()


def load_model_from_file(file_path: pathlib.Path, is_urdf=False) -> rod.Sdf:
    """
    Load an SDF or URDF model from a file.

    Args:
        file_path: The path to the model file.
        is_urdf: Whether the file is in URDF or SDF format.

    Returns:
        The corresponding rod model.
    """

    return rod.Sdf.load(file_path, is_urdf=is_urdf)


# ================
# Generic fixtures
# ================


@pytest.fixture(scope="function")
def prng_key() -> jax.Array:
    """
    Fixture to generate a new PRNG key for each test function.

    Returns:
        The new PRNG key passed to the test.

    Note:
        This fixture operates on a global variable initialized in the
        `pytest_configure` hook.
    """

    pytest.prng_key, subkey = jax.random.split(pytest.prng_key, num=2)
    return subkey


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(jaxsim.VelRepr.Inertial, id="inertial"),
        pytest.param(jaxsim.VelRepr.Body, id="body"),
        pytest.param(jaxsim.VelRepr.Mixed, id="mixed"),
    ],
)
def velocity_representation(request) -> jaxsim.VelRepr:
    """
    Parametrized fixture providing all supported velocity representations.

    Returns:
        A velocity representation.
    """

    return request.param


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(IntegratorType.SemiImplicitEuler, id="semi_implicit_euler"),
        pytest.param(IntegratorType.RungeKutta4, id="runge_kutta_4"),
        pytest.param(IntegratorType.RungeKutta4Fast, id="runge_kutta_4_fast"),
    ],
)
def integrator(request) -> str:
    """
    Fixture providing the integrator to use in the simulation.

    Returns:
        The integrator to use in the simulation.
    """

    return request.param


@pytest.fixture(scope="session")
def batch_size(request) -> int:
    """
    Fixture providing the batch size for vectorized benchmarks.

    Returns:
        The batch size for vectorized benchmarks.
    """

    return 1


# ================================
# Fixtures providing JaxSim models
# ================================

# All the fixtures in this section must have "session" scope.
# In this way, the models are generated only once and shared among all the tests.


# This is not a fixture.
def build_jaxsim_model(
    model_description: str | pathlib.Path | rod.Model,
) -> js.model.JaxSimModel:
    """
    Build a JaxSim model from a model description.

    Args:
        model_description: A model description provided by any fixture provider.

    Returns:
        A JaxSim model built from the provided description.
    """

    # Build the JaxSim model.
    model = js.model.JaxSimModel.build_from_model_description(
        model_description=model_description,
    )

    return model


@pytest.fixture(scope="session")
def jaxsim_model_box() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a box.

    Returns:
        The JaxSim model of a box.
    """

    import rod.builder.primitives
    import rod.urdf.exporter

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

    rod_model.add_frame(
        rod.Frame(
            name="box_frame",
            attached_to="box_link",
            pose=rod.Pose(relative_to="box_link", pose=[1, 1, 1, 0.5, 0.4, 0.3]),
        )
    )

    # Export the URDF string.
    urdf_string = rod.urdf.exporter.UrdfExporter(
        pretty=True, gazebo_preserve_fixed_joints=True
    ).to_urdf_string(sdf=rod_model)

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def jaxsim_model_sphere() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a sphere.

    Returns:
        The JaxSim model of a sphere.
    """

    import rod.builder.primitives
    import rod.urdf.exporter

    # Create on-the-fly a ROD model of a sphere.
    rod_model = (
        rod.builder.primitives.SphereBuilder(radius=0.1, mass=1.0, name="sphere")
        .build_model()
        .add_link()
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    # Export the URDF string.
    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model
    )

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def ergocub_model_description_path() -> pathlib.Path:
    """
    Fixture providing the path to the URDF model description of the ErgoCub robot.

    Returns:
        The path to the URDF model description of the ErgoCub robot.

    """

    try:
        os.environ["ROBOT_DESCRIPTION_COMMIT"] = "v0.7.7"

        import robot_descriptions.ergocub_description

    finally:
        _ = os.environ.pop("ROBOT_DESCRIPTION_COMMIT", None)

    model_urdf_path = pathlib.Path(
        robot_descriptions.ergocub_description.URDF_PATH.replace(
            "ergoCubSN002", "ergoCubSN001"
        )
    )

    return model_urdf_path


@pytest.fixture(scope="session")
def jaxsim_model_ergocub(
    ergocub_model_description_path: pathlib.Path,
) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the ErgoCub robot.

    Returns:
        The JaxSim model of the ErgoCub robot.

    """

    return build_jaxsim_model(model_description=ergocub_model_description_path)


@pytest.fixture(scope="session")
def jaxsim_model_ergocub_reduced(jaxsim_model_ergocub) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the ErgoCub robot with only locomotion joints.

    Returns:
        The JaxSim model of the ErgoCub robot with only locomotion joints.

    """

    model_full = jaxsim_model_ergocub

    # Get the names of the joints to keep.
    reduced_joints = tuple(
        j
        for j in model_full.joint_names()
        if "camera" not in j
        # Remove head and hands.
        and "neck" not in j
        and "wrist" not in j
        and "thumb" not in j
        and "index" not in j
        and "middle" not in j
        and "ring" not in j
        and "pinkie" not in j
        # Remove upper body.
        and "torso" not in j and "elbow" not in j and "shoulder" not in j
    )

    model = js.model.reduce(model=model_full, considered_joints=reduced_joints)

    return model


@pytest.fixture(scope="session")
def jaxsim_model_ur10() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of the UR10 robot.

    Returns:
        The JaxSim model of the UR10 robot.

    """

    import robot_descriptions.ur10_description

    model_urdf_path = pathlib.Path(robot_descriptions.ur10_description.URDF_PATH)

    return build_jaxsim_model(model_description=model_urdf_path)


@pytest.fixture(scope="session")
def jaxsim_model_single_pendulum() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a single pendulum.

    Returns:
        The JaxSim model of a single pendulum.
    """

    import rod.builder.primitives

    base_height = 2.15
    upper_height = 1.0

    # ===================
    # Create the builders
    # ===================

    base_builder = rod.builder.primitives.BoxBuilder(
        name="base",
        mass=1.0,
        x=0.15,
        y=0.15,
        z=base_height,
    )

    upper_builder = rod.builder.primitives.BoxBuilder(
        name="upper",
        mass=0.5,
        x=0.15,
        y=0.15,
        z=upper_height,
    )

    # =================
    # Create the joints
    # =================

    fixed = rod.Joint(
        name="fixed_joint",
        type="fixed",
        parent="world",
        child=base_builder.name,
    )

    pivot = rod.Joint(
        name="upper_joint",
        type="continuous",
        parent=base_builder.name,
        child=upper_builder.name,
        axis=rod.Axis(
            xyz=rod.Xyz([1, 0, 0]),
        ),
    )

    # ================
    # Create the links
    # ================

    base = (
        base_builder.build_link(
            name=base_builder.name,
            pose=rod.builder.primitives.PrimitiveBuilder.build_pose(
                pos=np.array([0, 0, base_height / 2])
            ),
        )
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    upper_pose = rod.builder.primitives.PrimitiveBuilder.build_pose(
        pos=np.array([0, 0, upper_height / 2])
    )

    upper = (
        upper_builder.build_link(
            name=upper_builder.name,
            pose=rod.builder.primitives.PrimitiveBuilder.build_pose(
                relative_to=base.name, pos=np.array([0, 0, upper_height])
            ),
        )
        .add_inertial(pose=upper_pose)
        .add_visual(pose=upper_pose)
        .add_collision(pose=upper_pose)
        .build()
    )

    rod_model = rod.Sdf(
        version="1.10",
        model=rod.Model(
            name="single_pendulum",
            link=[base, upper],
            joint=[fixed, pivot],
        ),
    )

    rod_model.model.resolve_frames()

    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model.models()[0]
    )

    model = build_jaxsim_model(model_description=urdf_string)

    return model


@pytest.fixture(scope="session")
def jaxsim_model_garpez() -> js.model.JaxSimModel:
    """Fixture to create the original (unscaled) Garpez model."""

    rod_model = create_scalable_garpez_model()

    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model
    )

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def jaxsim_model_garpez_scaled(request) -> js.model.JaxSimModel:
    """Fixture to create the scaled version of the Garpez model."""

    # Get the link scales from the request.
    link1_scale = request.param.get("link1_scale", 1.0)
    link2_scale = request.param.get("link2_scale", 1.0)
    link3_scale = request.param.get("link3_scale", 1.0)
    link4_scale = request.param.get("link4_scale", 1.0)

    rod_model = create_scalable_garpez_model(
        link1_scale=link1_scale,
        link2_scale=link2_scale,
        link3_scale=link3_scale,
        link4_scale=link4_scale,
    )

    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model
    )

    return build_jaxsim_model(model_description=urdf_string)


def create_scalable_garpez_model(
    link1_scale: float = 1.0,
    link2_scale: float = 1.0,
    link3_scale: float = 1.0,
    link4_scale: float = 1.0,
) -> rod.Model:
    """
    Build a scalable rod model to test parameterization and scaling.

    Args:
        link1_scale: Scale factor for link 1.
        link2_scale: Scale factor for link 2.
        link3_scale: Scale factor for link 3.
        link4_scale: Scale factor for link 4.

    Returns:
        A rod model with the specified link scales.

    Note:
        The model is built assuming a constant link density, hence scaling the link will also have an impact on the link mass.
    """

    import numpy as np
    from rod.builder import primitives

    # ========================
    # Create the link builders
    # ========================

    density = 1000.0  # Fixed density in kg/m^3

    l1_x, l1_y, l1_z = 0.3 * link1_scale, 0.2, 0.2
    l1_volume = l1_x * l1_y * l1_z
    l1_mass = density * l1_volume
    link1_builder = primitives.BoxBuilder(
        name="link1", mass=l1_mass, x=l1_x, y=l1_y, z=l1_z
    )

    l2_radius = 0.1 * link2_scale
    l2_volume = 4 / 3 * np.pi * l2_radius**3
    l2_mass = density * l2_volume
    link2_builder = primitives.SphereBuilder(
        name="link2", mass=l2_mass, radius=l2_radius
    )

    l3_radius = 0.05
    l3_length = 0.5 * link3_scale
    l3_volume = np.pi * l3_radius**2 * l3_length
    l3_mass = density * l3_volume
    link3_builder = primitives.CylinderBuilder(
        name="link3", mass=l3_mass, radius=l3_radius, length=l3_length
    )

    l4_x, l4_y, l4_z = 0.3 * link4_scale, 0.2, 0.1
    l4_volume = l4_x * l4_y * l4_z
    l4_mass = density * l4_volume
    link4_builder = primitives.BoxBuilder(
        name="link4", mass=l4_mass, x=l4_x, y=l4_y, z=l4_z
    )

    # =================
    # Create the joints
    # =================

    link1_to_link2 = rod.Joint(
        name="link1_to_link2",
        type="revolute",
        parent=link1_builder.name,
        child=link2_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=link1_builder.name,
            pos=np.array([link1_builder.x, link1_builder.y / 2, link1_builder.z / 2]),
        ),
        axis=rod.Axis(xyz=rod.Xyz(xyz=[0, 1, 0]), limit=rod.Limit()),
    )

    link2_to_link3 = rod.Joint(
        name="link2_to_link3",
        type="revolute",
        parent=link2_builder.name,
        child=link3_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=link2_builder.name,
            pos=np.array([link2_builder.radius, 0, -link2_builder.radius]),
        ),
        axis=rod.Axis(xyz=rod.Xyz(xyz=[0, 0, 1]), limit=rod.Limit()),
    )

    link3_to_link4 = rod.Joint(
        name="link3_to_link4",
        type="revolute",
        parent=link3_builder.name,
        child=link4_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=link3_builder.name,
            pos=np.array([-link3_builder.radius, 0, -link3_builder.length]),
        ),
        axis=rod.Axis(xyz=rod.Xyz(xyz=[1, 0, 0]), limit=rod.Limit()),
    )

    # ================
    # Create the links
    # ================

    link1_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([link1_builder.x, link1_builder.y, link1_builder.z]) / 2
    )

    link1 = (
        link1_builder.build_link(
            name=link1_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(relative_to="__model__"),
        )
        .add_inertial(pose=link1_elements_pose)
        .add_visual(pose=link1_elements_pose)
        .add_collision(pose=link1_elements_pose)
        .build()
    )

    link2_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([link2_builder.radius, 0, 0])
    )

    link2 = (
        link2_builder.build_link(
            name=link2_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=link1_to_link2.name
            ),
        )
        .add_inertial(pose=link2_elements_pose)
        .add_visual(pose=link2_elements_pose)
        .add_collision(pose=link2_elements_pose)
        .build()
    )

    link3_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([0, 0, -link3_builder.length / 2])
    )

    link3 = (
        link3_builder.build_link(
            name=link3_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=link2_to_link3.name
            ),
        )
        .add_inertial(pose=link3_elements_pose)
        .add_visual(pose=link3_elements_pose)
        .add_collision(pose=link3_elements_pose)
        .build()
    )

    link4_elements_pose = primitives.PrimitiveBuilder.build_pose(
        # pos=np.array([0, 0, -link4_builder.z / 2])
        pos=np.array([link4_builder.x / 2, 0, -link4_builder.z / 2])
    )

    link4 = (
        link4_builder.build_link(
            name=link4_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=link3_to_link4.name
            ),
        )
        .add_inertial(pose=link4_elements_pose)
        .add_visual(pose=link4_elements_pose)
        .add_collision(pose=link4_elements_pose)
        .build()
    )

    # ===========
    # Build model
    # ===========

    # Create model
    rod_model = rod.Model(
        name="model_demo",
        canonical_link=link1.name,
        link=[link1, link2, link3, link4],
        joint=[link1_to_link2, link2_to_link3, link3_to_link4],
    )

    rod_model.switch_frame_convention(
        frame_convention=rod.FrameConvention.Urdf,
        explicit_frames=True,
        attach_frames_to_links=True,
    )

    assert rod.Sdf(model=rod_model, version="1.10").serialize(validate=True)

    return rod_model


def create_model_with_missing_collision() -> rod.Model:
    """
    Build a rod model with a link that has a visual but no collision element.

    This model is used to test the export logic when collision elements are missing.

    Returns:
        A rod model with one link missing a collision element.
    """

    import numpy as np
    from rod.builder import primitives

    density = 1000.0  # Fixed density in kg/m^3

    # Create link1 with both visual and collision
    l1_x, l1_y, l1_z = 0.3, 0.2, 0.2
    l1_volume = l1_x * l1_y * l1_z
    l1_mass = density * l1_volume
    link1_builder = primitives.BoxBuilder(
        name="link1", mass=l1_mass, x=l1_x, y=l1_y, z=l1_z
    )

    # Create link2 with visual but WITHOUT collision
    l2_radius = 0.1
    l2_volume = 4 / 3 * np.pi * l2_radius**3
    l2_mass = density * l2_volume
    link2_builder = primitives.SphereBuilder(
        name="link2", mass=l2_mass, radius=l2_radius
    )

    # Create joint
    link1_to_link2 = rod.Joint(
        name="link1_to_link2",
        type="revolute",
        parent=link1_builder.name,
        child=link2_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=link1_builder.name,
            pos=np.array([link1_builder.x, link1_builder.y / 2, link1_builder.z / 2]),
        ),
        axis=rod.Axis(xyz=rod.Xyz(xyz=[0, 1, 0]), limit=rod.Limit()),
    )

    # Build link1 with visual and collision
    link1_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([link1_builder.x, link1_builder.y, link1_builder.z]) / 2
    )

    link1 = (
        link1_builder.build_link(
            name=link1_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(relative_to="__model__"),
        )
        .add_inertial(pose=link1_elements_pose)
        .add_visual(pose=link1_elements_pose)
        .add_collision(pose=link1_elements_pose)
        .build()
    )

    # Build link2 with visual but NO collision
    link2_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([link2_builder.radius, 0, 0])
    )

    link2 = (
        link2_builder.build_link(
            name=link2_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=link1_to_link2.name
            ),
        )
        .add_inertial(pose=link2_elements_pose)
        .add_visual(pose=link2_elements_pose)
        # Note: NO .add_collision() call here
        .build()
    )

    # Create model
    rod_model = rod.Model(
        name="model_missing_collision",
        canonical_link=link1.name,
        link=[link1, link2],
        joint=[link1_to_link2],
    )

    rod_model.switch_frame_convention(
        frame_convention=rod.FrameConvention.Urdf,
        explicit_frames=True,
        attach_frames_to_links=True,
    )

    assert rod.Sdf(model=rod_model, version="1.10").serialize(validate=True)

    return rod_model


@pytest.fixture(scope="session")
def jaxsim_model_missing_collision() -> js.model.JaxSimModel:
    """
    Fixture to create a model with a link that has a visual but no collision element.

    This is used to test the export logic when collision elements are missing.
    """

    rod_model = create_model_with_missing_collision()

    urdf_string = rod.urdf.exporter.UrdfExporter(pretty=True).to_urdf_string(
        sdf=rod_model
    )

    return build_jaxsim_model(model_description=urdf_string)


@pytest.fixture(scope="session")
def jaxsim_model_double_pendulum() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a double pendulum.
    Returns:
        The JaxSim model of a double pendulum.
    """

    model_path = pathlib.Path(__file__).parent / "assets" / "double_pendulum.sdf"
    rod_model = load_model_from_file(model_path)
    model = build_jaxsim_model(model_description=rod_model)

    return model


@pytest.fixture(scope="session")
def jaxsim_model_cartpole() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a cartpole.
    Returns:
        The JaxSim model of a cartpole.
    """

    model_path = (
        pathlib.Path(__file__).parent.parent / "examples" / "assets" / "cartpole.urdf"
    )
    rod_model = load_model_from_file(model_path, is_urdf=True)
    model = build_jaxsim_model(model_description=rod_model)

    return model


@pytest.fixture(scope="session")
def jaxsim_model_4_bar_linkage() -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a 4-bar linkage (opened configuration).

    Returns:
        The JaxSim model of the 4-bar linkage.
    """

    model_path = pathlib.Path(__file__).parent / "assets" / "4_bar_opened.urdf"
    rod_model = load_model_from_file(model_path, is_urdf=True)
    model = build_jaxsim_model(model_description=rod_model)

    return model


# ============================
# Collections of JaxSim models
# ============================


def get_jaxsim_model_fixture(
    model_name: str, request: pytest.FixtureRequest
) -> str | pathlib.Path:
    """
    Get the fixture providing the JaxSim model of a robot.

    Args:
        model_name: The name of the model.
        request: The request object.

    Returns:
        The JaxSim model of the robot.

    """

    match model_name:
        case "box":
            return request.getfixturevalue(jaxsim_model_box.__name__)
        case "sphere":
            return request.getfixturevalue(jaxsim_model_sphere.__name__)
        case "ergocub":
            return request.getfixturevalue(jaxsim_model_ergocub.__name__)
        case "ergocub_reduced":
            return request.getfixturevalue(jaxsim_model_ergocub_reduced.__name__)
        case "ur10":
            return request.getfixturevalue(jaxsim_model_ur10.__name__)
        case "single_pendulum":
            return request.getfixturevalue(jaxsim_model_single_pendulum.__name__)
        case "garpez":
            return request.getfixturevalue(jaxsim_model_garpez.__name__)
        case "garpez_scaled":
            return request.getfixturevalue(jaxsim_model_garpez_scaled.__name__)
        case _:
            raise ValueError(model_name)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "sphere",
        "ur10",
        "ergocub",
        "ergocub_reduced",
    ],
)
def jaxsim_models_all(request) -> pathlib.Path | str:
    """
    Fixture providing the JaxSim models of all supported robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "ur10",
        "ergocub_reduced",
    ],
)
def jaxsim_models_types(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of all types of supported robots.

    Note:
        At the moment, most of our tests use this fixture. It provides:
        - A robot with no joints.
        - A fixed-base robot.
        - A floating-base robot.

    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "box",
        "sphere",
    ],
)
def jaxsim_models_no_joints(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of robots with no joints.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "ergocub",
        "ergocub_reduced",
    ],
)
def jaxsim_models_floating_base(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of floating-base robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(
    scope="session",
    params=[
        "ur10",
    ],
)
def jaxsim_models_fixed_base(request) -> pathlib.Path | str:
    """
    Fixture providing JaxSim models of fixed-base robots.
    """

    model_name: str = request.param
    return get_jaxsim_model_fixture(model_name=model_name, request=request)


@pytest.fixture(scope="function")
def set_jax_32bit(monkeypatch):
    """
    Fixture that temporarily sets JAX precision to 32-bit for the duration of the test.
    """

    del globals()["jaxsim"]
    del globals()["js"]

    # Temporarily disable x64
    monkeypatch.setenv("JAX_ENABLE_X64", "0")


@pytest.fixture(scope="function")
def jaxsim_model_box_32bit(set_jax_32bit, request) -> js.model.JaxSimModel:
    """
    Fixture providing the JaxSim model of a box with 32-bit precision.

    Returns:
        The JaxSim model of a box with 32-bit precision.

    """

    return get_jaxsim_model_fixture(model_name="box", request=request)
