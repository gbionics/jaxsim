import copy
import pathlib
import tempfile
import warnings
from collections.abc import Sequence
from typing import Any

import jaxlie
import mujoco as mj
import numpy as np
import rod.urdf.exporter
from lxml import etree as ET

from jaxsim import logging

from .utils import MujocoCamera

MujocoCameraType = (
    MujocoCamera | Sequence[MujocoCamera] | dict[str, str] | Sequence[dict[str, str]]
)


def load_rod_model(
    model_description: str | pathlib.Path | rod.Model,
    is_urdf: bool | None = None,
    model_name: str | None = None,
) -> rod.Model:
    """
    Load a ROD model from a URDF/SDF file or a ROD model.

    Args:
        model_description: The URDF/SDF file or ROD model to load.
        is_urdf: Whether to force parsing the model description as a URDF file.
        model_name: The name of the model to load from the resource.

    Returns:
        rod.Model: The loaded ROD model.
    """

    # Parse the SDF resource.
    sdf_element = rod.Sdf.load(sdf=model_description, is_urdf=is_urdf)

    # Fail if the SDF resource has no model.
    if len(sdf_element.models()) == 0:
        raise RuntimeError("Failed to find any model in the model description")

    # Return the model if there is only one.
    if len(sdf_element.models()) == 1:
        if model_name is not None and sdf_element.models()[0].name != model_name:
            raise ValueError(f"Model '{model_name}' not found in the description")

        return sdf_element.models()[0]

    # Require users to specify the model name if there are multiple models.
    if model_name is None:
        msg = "The resource has multiple models. Please specify the model name."
        raise ValueError(msg)

    # Build a dictionary of models in the resource for easy access.
    models = {m.name: m for m in sdf_element.models()}

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in the resource")

    return models[model_name]


class ModelToMjcf:
    """
    Class to convert a URDF/SDF file or a ROD model to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        model: str | pathlib.Path | rod.Model,
        considered_joints: list[str] | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        heightmap_samples_xy: tuple[int, int] = (101, 101),
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a model to a Mujoco MJCF string.

        Args:
            model: The URDF/SDF file or ROD model to convert.
            considered_joints: The list of joint names to consider in the conversion.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            heightmap_samples_xy: The number of points in the heightmap grid.
            cameras: The custom cameras to add to the scene.

        Returns:
            A tuple containing the MJCF string and the dictionary of assets.
        """

        match model:
            case rod.Model():
                rod_model = model
            case str() | pathlib.Path():
                # Convert the JaxSim model to a ROD model.
                rod_model = load_rod_model(
                    model_description=model,
                    is_urdf=None,
                    model_name=None,
                )
            case _:
                raise TypeError(f"Unsupported type for 'model': {type(model)}")

        # Convert the ROD model to MJCF.
        return RodModelToMjcf.convert(
            rod_model=rod_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            heightmap_samples_xy=heightmap_samples_xy,
            cameras=cameras,
        )


class RodModelToMjcf:
    """
    Class to convert a ROD model to a Mujoco MJCF string.
    """

    @staticmethod
    def assets_from_rod_model(
        rod_model: rod.Model,
    ) -> dict[str, bytes]:
        """
        Generate a dictionary of assets from a ROD model.

        Args:
            rod_model: The ROD model to extract the assets from.

        Returns:
            dict: A dictionary of assets.
        """

        import resolve_robotics_uri_py

        assets_files = dict()

        for link in rod_model.links():
            for visual in link.visuals():
                if visual.geometry.mesh and visual.geometry.mesh.uri:
                    assets_files[visual.geometry.mesh.uri] = (
                        resolve_robotics_uri_py.resolve_robotics_uri(
                            visual.geometry.mesh.uri
                        )
                    )

            for collision in link.collisions():
                if collision.geometry.mesh and collision.geometry.mesh.uri:
                    assets_files[collision.geometry.mesh.uri] = (
                        resolve_robotics_uri_py.resolve_robotics_uri(
                            collision.geometry.mesh.uri
                        )
                    )

        assets = {
            asset_name: asset.read_bytes() for asset_name, asset in assets_files.items()
        }

        return assets

    @staticmethod
    def add_floating_joint(
        urdf_string: str,
        base_link_name: str,
        floating_joint_name: str = "world_to_base",
    ) -> str:
        """
        Add a floating joint to a URDF string.

        Args:
            urdf_string: The URDF string to modify.
            base_link_name: The name of the base link to attach the floating joint.
            floating_joint_name: The name of the floating joint to add.

        Returns:
            str: The modified URDF string.
        """

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".urdf") as urdf_file:

            # Write the URDF string to a temporary file and move current position
            # to the beginning.
            urdf_file.write(urdf_string)
            urdf_file.seek(0)

            # Parse the MJCF string as XML (etree).
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.parse(source=urdf_file, parser=parser)

        root: ET._Element = tree.getroot()

        if root.find(f".//joint[@name='{floating_joint_name}']") is not None:
            msg = f"The URDF already has a floating joint '{floating_joint_name}'"
            warnings.warn(msg, stacklevel=2)
            return ET.tostring(root, pretty_print=True).decode()

        # Create the "world" link if it doesn't exist.
        if root.find(".//link[@name='world']") is None:
            _ = ET.SubElement(root, "link", name="world")

        # Create the floating joint.
        world_to_base = ET.SubElement(
            root, "joint", name=floating_joint_name, type="floating"
        )

        # Check that the base link exists.
        if root.find(f".//link[@name='{base_link_name}']") is None:
            raise ValueError(f"Link '{base_link_name}' not found in the URDF")

        # Attach the floating joint to the base link.
        ET.SubElement(world_to_base, "parent", link="world")
        ET.SubElement(world_to_base, "child", link=base_link_name)

        urdf_string = ET.tostring(root, pretty_print=True).decode()
        return urdf_string

    @staticmethod
    def convert(
        rod_model: rod.Model,
        considered_joints: list[str] | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        heightmap_samples_xy: tuple[int, int] = (101, 101),
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a ROD model to a Mujoco MJCF string.

        Args:
            rod_model: The ROD model to convert.
            considered_joints: The list of joint names to consider in the conversion.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            heightmap_samples_xy: The number of points in the heightmap grid.
            cameras: The custom cameras to add to the scene.

        Returns:
            A tuple containing the MJCF string and the dictionary of assets.
        """

        # -------------------------------------
        # Convert the model description to URDF
        # -------------------------------------

        # Consider all joints if not specified otherwise.
        considered_joints = set(
            considered_joints
            if considered_joints is not None
            else [j.name for j in rod_model.joints() if j.type != "fixed"]
        )

        # If considered joints are passed, make sure that they are all part of the model.
        if considered_joints - {j.name for j in rod_model.joints()}:
            extra_joints = considered_joints - {j.name for j in rod_model.joints()}

            msg = f"Couldn't find the following joints in the model: '{extra_joints}'"
            raise ValueError(msg)

        # Create a dictionary of joints for quick access.
        joints_dict = {j.name: j for j in rod_model.joints()}

        # Convert all the joints not considered to fixed joints.
        for joint_name in {j.name for j in rod_model.joints()} - considered_joints:
            joints_dict[joint_name].type = "fixed"

        # Convert the ROD model to URDF.
        urdf_string = rod.urdf.exporter.UrdfExporter(
            gazebo_preserve_fixed_joints=False, pretty=True
        ).to_urdf_string(
            sdf=rod.Sdf(model=rod_model, version="1.7"),
        )

        # -------------------------------------
        # Add a floating joint if floating-base
        # -------------------------------------

        base_link_name = rod_model.get_canonical_link()

        if not rod_model.is_fixed_base():
            considered_joints |= {"world_to_base"}
            urdf_string = RodModelToMjcf.add_floating_joint(
                urdf_string=urdf_string,
                base_link_name=base_link_name,
                floating_joint_name="world_to_base",
            )

        # ---------------------------------------
        # Inject the <mujoco> element in the URDF
        # ---------------------------------------

        parser = ET.XMLParser(remove_blank_text=True)
        root = ET.fromstring(text=urdf_string.encode(), parser=parser)

        mujoco_element = (
            ET.SubElement(root, "mujoco")
            if len(root.findall("./mujoco")) == 0
            else root.find("./mujoco")
        )

        _ = ET.SubElement(
            mujoco_element,
            "compiler",
            balanceinertia="true",
            discardvisual="false",
        )

        urdf_string = ET.tostring(root, pretty_print=True).decode()

        # ------------------------------
        # Post-process all dummy visuals
        # ------------------------------

        parser = ET.XMLParser(remove_blank_text=True)
        root: ET._Element = ET.fromstring(text=urdf_string.encode(), parser=parser)

        # Give a tiny radius to all dummy spheres
        for geometry in root.findall(".//visual/geometry[sphere]"):
            radius = np.fromstring(
                geometry.find("./sphere").attrib["radius"], sep=" ", dtype=float
            )
            if np.allclose(radius, np.zeros(1)):
                geometry.find("./sphere").set("radius", "0.001")

        # Give a tiny volume to all dummy boxes
        for geometry in root.findall(".//visual/geometry[box]"):
            size = np.fromstring(
                geometry.find("./box").attrib["size"], sep=" ", dtype=float
            )
            if np.allclose(size, np.zeros(3)):
                geometry.find("./box").set("size", "0.001 0.001 0.001")

        urdf_string = ET.tostring(root, pretty_print=True).decode()

        # ------------------------
        # Convert the URDF to MJCF
        # ------------------------

        # Load the URDF model into Mujoco.
        assets = RodModelToMjcf.assets_from_rod_model(rod_model=rod_model)
        mj_model = mj.MjModel.from_xml_string(xml=urdf_string, assets=assets)

        # Get the joint names.
        mj_joint_names = {
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(mj_model.njnt)
        }

        # Check that the Mujoco model only has the considered joints.
        if mj_joint_names != considered_joints:
            extra1 = mj_joint_names - considered_joints
            extra2 = considered_joints - mj_joint_names
            extra_joints = extra1.union(extra2)
            msg = "The Mujoco model has the following extra/missing joints: '{}'"
            raise ValueError(msg.format(extra_joints))

        # Windows locks open files, so we use mkstemp() to create a temporary file without keeping it open.
        with tempfile.NamedTemporaryFile(
            suffix=".xml", prefix=f"{rod_model.name}_", delete=False
        ) as tmp:
            temp_filename = tmp.name

        try:
            # Convert the in-memory Mujoco model to MJCF.
            mj.mj_saveLastXML(temp_filename, mj_model)

            # Parse the MJCF file as XML.
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.parse(source=temp_filename, parser=parser)

        finally:
            pathlib.Path(temp_filename).unlink()

        # Get the root element.
        root: ET._Element = tree.getroot()

        # Find the <mujoco> element (might be the root itself).
        mujoco_element: ET._Element = next(iter(root.iter("mujoco")))

        # --------------
        # Add the frames
        # --------------

        for frame in rod_model.frames():
            frame: rod.Frame
            parent_name = frame.attached_to
            parent_element = mujoco_element.find(f".//body[@name='{parent_name}']")

            if parent_element is None and parent_name == base_link_name:
                parent_element = mujoco_element.find(".//worldbody")

            if parent_element is not None:
                quat = jaxlie.SO3.from_rpy_radians(*frame.pose.rpy).wxyz
                _ = ET.SubElement(
                    parent_element,
                    "site",
                    name=frame.name,
                    pos=" ".join(map(str, frame.pose.xyz)),
                    quat=" ".join(map(str, quat)),
                )
            else:
                logging.debug(f"Parent link '{parent_name}' not found")

        # --------------
        # Add the motors
        # --------------

        if len(mujoco_element.findall(".//actuator")) > 0:
            raise RuntimeError("The model already has <actuator> elements.")

        # Add the actuator element.
        actuator_element = ET.SubElement(mujoco_element, "actuator")

        # Add a motor for each joint.
        for joint_element in mujoco_element.findall(".//joint"):
            assert (
                joint_element.attrib["name"] in considered_joints
            ), joint_element.attrib["name"]
            if joint_element.attrib.get("type", "hinge") in {"free", "ball"}:
                continue
            ET.SubElement(
                actuator_element,
                "motor",
                name=f"{joint_element.attrib['name']}_motor",
                joint=joint_element.attrib["name"],
                gear="1",
            )

        # ---------------------------------------------
        # Set full transparency of collision geometries
        # ---------------------------------------------

        parser = ET.XMLParser(remove_blank_text=True)

        # Get all the (optional) names of the URDF collision elements
        collision_names = {
            c.attrib["name"]
            for c in ET.fromstring(text=urdf_string.encode(), parser=parser).findall(
                ".//collision[geometry]"
            )
            if "name" in c.attrib
        }

        # Set alpha=0 to the color of all collision elements
        for geometry_element in mujoco_element.findall(".//geom[@rgba]"):
            if geometry_element.attrib.get("name") in collision_names:
                r, g, b, _ = geometry_element.attrib["rgba"].split(" ")
                geometry_element.set("rgba", f"{r} {g} {b} 0")

        # -----------------------
        # Create the scene assets
        # -----------------------

        asset_element = (
            ET.SubElement(mujoco_element, "asset")
            if len(mujoco_element.findall(".//asset")) == 0
            else mujoco_element.find(".//asset")
        )

        _ = ET.SubElement(
            asset_element,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.3 0.5 0.7",
            rgb2="0 0 0",
            width="512",
            height="512",
        )

        _ = ET.SubElement(
            asset_element,
            "texture",
            name="plane_texture",
            type="2d",
            builtin="checker",
            rgb1="0.1 0.2 0.3",
            rgb2="0.2 0.3 0.4",
            width="512",
            height="512",
            mark="cross",
            markrgb=".8 .8 .8",
        )

        _ = ET.SubElement(
            asset_element,
            "material",
            name="plane_material",
            texture="plane_texture",
            reflectance="0.2",
            texrepeat="5 5",
            texuniform="true",
        )

        _ = (
            ET.SubElement(
                asset_element,
                "hfield",
                name="terrain",
                nrow=str(int(heightmap_samples_xy[0])),
                ncol=str(int(heightmap_samples_xy[1])),
                # The following 'size' is a placeholder, it is updated dynamically
                # when a hfield/heightmap is stored into MjData.
                size="1 1 1 1",
            )
            if heightmap
            else None
        )

        # ----------------------------------
        # Populate the scene with the assets
        # ----------------------------------

        worldbody_scene_element = ET.SubElement(mujoco_element, "worldbody")

        _ = ET.SubElement(
            worldbody_scene_element,
            "geom",
            name="floor",
            type="plane" if not heightmap else "hfield",
            size="0 0 0.05",
            material="plane_material",
            condim="3",
            contype="1",
            conaffinity="1",
            zaxis=" ".join(map(str, plane_normal)),
            **({"hfield": "terrain"} if heightmap else {}),
        )

        _ = ET.SubElement(
            worldbody_scene_element,
            "light",
            name="sun",
            mode="fixed",
            directional="true",
            castshadow="true",
            pos="0 0 10",
            dir="0 0 -1",
        )

        # -------------------------------------------------------
        # Add a camera following the CoM of the worldbody element
        # -------------------------------------------------------

        worldbody_element = None

        # Find the <worldbody> element of our model by searching the one that contains
        # all the considered joints. This is needed because there might be multiple
        # <worldbody> elements inside <mujoco>.
        for wb in mujoco_element.findall(".//worldbody"):
            if all(
                wb.find(f".//joint[@name='{j}']") is not None for j in considered_joints
            ):
                worldbody_element = wb
                break

        if worldbody_element is None:
            raise RuntimeError("Failed to find the <worldbody> element of the model")

        # Camera attached to the model
        # It can be manually copied from `python -m mujoco.viewer --mjcf=<URDF_PATH>`
        _ = ET.SubElement(
            worldbody_element,
            "camera",
            name="track",
            mode="trackcom",
            pos="1.930 -2.279 0.556",
            xyaxes="0.771 0.637 0.000 -0.116 0.140 0.983",
            fovy="60",
        )

        # Add user-defined camera.
        for camera in cameras if isinstance(cameras, Sequence) else [cameras]:

            mj_camera = (
                camera
                if isinstance(camera, MujocoCamera)
                else MujocoCamera.build(**camera)
            )

            _ = ET.SubElement(worldbody_element, "camera", mj_camera.asdict())

        # ------------------------------------------------
        # Add a light following the  CoM of the first link
        # ------------------------------------------------

        if not rod_model.is_fixed_base():

            # Light attached to the model
            _ = ET.SubElement(
                worldbody_element,
                "light",
                name="light_model",
                mode="targetbodycom",
                target=worldbody_element.find(".//body").attrib["name"],
                directional="false",
                castshadow="true",
                pos="1 0 5",
            )

        # --------------------------------
        # Return the resulting MJCF string
        # --------------------------------

        mjcf_string = ET.tostring(root, pretty_print=True).decode()
        return mjcf_string, assets


class UrdfToMjcf:
    """
    Class to convert a URDF file to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        urdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a URDF file to a Mujoco MJCF string.

        Args:
            urdf: The URDF file to convert.
            considered_joints: The list of joint names to consider in the conversion.
            model_name: The name of the model to convert.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            cameras: The list of cameras to add to the scene.

        Returns:
            tuple: A tuple containing the MJCF string and the assets dictionary.
        """

        logging.warning("This method is deprecated. Use 'ModelToMjcf.convert' instead.")

        # Get the ROD model.
        rod_model = load_rod_model(
            model_description=urdf,
            is_urdf=True,
            model_name=model_name,
        )

        # Convert the ROD model to MJCF.
        return RodModelToMjcf.convert(
            rod_model=rod_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )


class SdfToMjcf:
    """
    Class to convert a SDF file to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        sdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a SDF file to a Mujoco MJCF string.

        Args:
            sdf: The SDF file to convert.
            considered_joints: The list of joint names to consider in the conversion.
            model_name: The name of the model to convert.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            cameras: The list of cameras to add to the scene.

        Returns:
            tuple: A tuple containing the MJCF string and the assets dictionary.
        """

        logging.warning("This method is deprecated. Use 'ModelToMjcf.convert' instead.")

        # Get the ROD model.
        rod_model = load_rod_model(
            model_description=sdf,
            is_urdf=False,
            model_name=model_name,
        )

        # Convert the ROD model to MJCF.
        return RodModelToMjcf.convert(
            rod_model=rod_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )
