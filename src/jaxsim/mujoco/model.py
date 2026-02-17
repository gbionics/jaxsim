from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import mujoco as mj
import numpy as np
import numpy.typing as npt
import xmltodict
from scipy.spatial.transform import Rotation

import jaxsim.typing as jtp

HeightmapCallable = Callable[[jtp.FloatLike, jtp.FloatLike], jtp.FloatLike]


class MujocoModelHelper:
    """
    Helper class to create and interact with Mujoco models and data objects.
    """

    def __init__(self, model: mj.MjModel, data: mj.MjData | None = None) -> None:
        """
        Initialize the MujocoModelHelper object.

        Args:
            model: A Mujoco model object.
            data: A Mujoco data object. If None, a new one will be created.
        """

        self.model = model
        self.data = data if data is not None else mj.MjData(self.model)

        # Populate the data with kinematics.
        mj.mj_forward(self.model, self.data)

        # Keep the cache of this method local to improve GC.
        self.mask_qpos = functools.cache(self._mask_qpos)

    @staticmethod
    def build_from_xml(
        mjcf_description: str | pathlib.Path,
        assets: dict[str, Any] | None = None,
        heightmap: HeightmapCallable | None = None,
        heightmap_name: str = "terrain",
        heightmap_radius_xy: tuple[float, float] = (1.0, 1.0),
    ) -> MujocoModelHelper:
        """
        Build a Mujoco model from an MJCF description.

        Args:
            mjcf_description:
                A string containing the XML description of the Mujoco model
                or a path to a file containing the XML description.
            assets: An optional dictionary containing the assets of the model.
            heightmap:
                A function in two variables that returns the height of a terrain
                in the specified coordinate point.
            heightmap_name:
                The default name of the heightmap in the MJCF description
                to load the corresponding configuration.
            heightmap_radius_xy:
                The extension of the heightmap in the x-y surface corresponding to the
                plane over which the grid of the sampled heightmap is generated.

        Returns:
            A MujocoModelHelper object.
        """

        # Read the XML description if it is a path to file.
        mjcf_description = (
            mjcf_description.read_text()
            if isinstance(mjcf_description, pathlib.Path)
            else mjcf_description
        )

        if heightmap is None:
            hfield = None

        else:

            mjcf_description_dict = xmltodict.parse(xml_input=mjcf_description)

            # Create a dictionary of all hfield configurations from the MJCF.
            hfields = mjcf_description_dict["mujoco"]["asset"].get("hfield", [])
            hfields = hfields if isinstance(hfields, list) else [hfields]
            hfields_dict = {hfield["@name"]: hfield for hfield in hfields}

            if heightmap_name not in hfields_dict:
                raise ValueError(f"Heightmap '{heightmap_name}' not found in MJCF")

            hfield_element = hfields_dict[heightmap_name]

            # Generate the hfield by sampling the heightmap function.
            hfield = generate_hfield(
                heightmap=heightmap,
                samples_xy=(int(hfield_element["@nrow"]), int(hfield_element["@ncol"])),
                radius_xy=heightmap_radius_xy,
            )

            # Update dynamically the '/asset/hfield[@name=heightmap_name]@size' attribute
            # with the information of the sampled points.
            # This is necessary for correctly rendering the heightmap over the
            # specified xy area with the correct z elevation.
            size = [float(el) for el in hfield_element["@size"].split(" ")]
            size[0], size[1] = heightmap_radius_xy
            size[2] = 1.0
            # The following could be zero but Mujoco complains if it's exactly zero.
            size[3] = max(0.000_001, -min(hfield))

            # Replace the 'size' attribute.
            hfields_dict[heightmap_name]["@size"] = " ".join(str(el) for el in size)

            # Update the hfield elements of the original MJCF.
            # Only the hfield corresponding to 'heightmap_name' was actually edited.
            mjcf_description_dict["mujoco"]["asset"]["hfield"] = list(
                hfields_dict.values()
            )

            # Serialize the updated MJCF to XML.
            mjcf_description = xmltodict.unparse(
                input_dict=mjcf_description_dict, pretty=True
            )

        # Create the Mujoco model from the XML and, optionally, the dictionary of assets.
        model = mj.MjModel.from_xml_string(xml=mjcf_description, assets=assets)
        data = mj.MjData(model)

        # Store the sampled heightmap into the Mujoco model.
        if heightmap is not None:
            assert hfield is not None
            model.hfield_data = hfield

        return MujocoModelHelper(model=model, data=data)

    def time(self) -> float:
        """Return the simulation time."""

        return self.data.time

    def timestep(self) -> float:
        """Return the simulation timestep."""

        return self.model.opt.timestep

    def gravity(self) -> npt.NDArray:
        """Return the 3D gravity vector."""

        return np.array([0, 0, self.model.gravity])

    # =========================
    # Methods for the base link
    # =========================

    def is_floating_base(self) -> bool:
        """Return true if the model is floating-base."""

        # A body with no joints is considered a fixed-base model.
        # In fact, in mujoco, a floating-base model has a 6 DoFs first joint.
        if self.number_of_joints() == 0:
            return False

        # We just check that the first joint has 6 DoFs.
        joint0_type = self.model.jnt_type[0]
        return joint0_type == mj.mjtJoint.mjJNT_FREE

    def is_fixed_base(self) -> bool:
        """Return true if the model is fixed-base."""

        return not self.is_floating_base()

    def base_link(self) -> str:
        """Return the name of the base link."""

        return mj.mj_id2name(
            self.model, mj.mjtObj.mjOBJ_BODY, 0 if self.is_fixed_base() else 1
        )

    def base_position(self) -> npt.NDArray:
        """Return the 3D position of the base link."""

        return (
            self.data.qpos[:3]
            if self.is_floating_base()
            else self.body_position(body_name=self.base_link())
        )

    def base_orientation(self, dcm: bool = False) -> npt.NDArray:
        """Return the orientation of the base link."""

        return (
            (
                np.reshape(self.data.xmat[0], newshape=(3, 3))
                if dcm
                else self.data.xquat[0]
            )
            if self.is_floating_base()
            else self.body_orientation(body_name=self.base_link(), dcm=dcm)
        )

    def set_base_position(self, position: npt.NDArray) -> None:
        """Set the 3D position of the base link."""

        if self.is_fixed_base():
            raise ValueError("The position of a fixed-base model cannot be set.")

        position = np.atleast_1d(np.array(position).squeeze())

        if position.size != 3:
            raise ValueError(f"Wrong position size ({position.size})")

        self.data.qpos[:3] = position

    def set_base_orientation(self, orientation: npt.NDArray, dcm: bool = False) -> None:
        """Set the 3D position of the base link."""

        if self.is_fixed_base():
            raise ValueError("The orientation of a fixed-base model cannot be set.")

        orientation = (
            np.atleast_2d(np.array(orientation).squeeze())
            if dcm
            else np.atleast_1d(np.array(orientation).squeeze())
        )

        if orientation.shape != ((4,) if not dcm else (3, 3)):
            raise ValueError(f"Wrong orientation shape {orientation.shape}")

        def is_quaternion(Q):
            return np.allclose(np.linalg.norm(Q), 1.0)

        def is_dcm(R):
            return np.allclose(np.linalg.det(R), 1.0) and np.allclose(
                R.T @ R, np.eye(3)
            )

        if not (is_quaternion(orientation) if not dcm else is_dcm(orientation)):
            raise ValueError("The orientation is not a valid element of SO(3)")

        W_Q_B = (
            Rotation.from_matrix(orientation).as_quat(
                canonical=True, scalar_first=False
            )
            if dcm
            else orientation
        )

        self.data.qpos[3:7] = W_Q_B

    # ==================
    # Methods for joints
    # ==================

    def number_of_joints(self) -> int:
        """Return the number of joints in the model."""

        return self.model.njnt

    def number_of_dofs(self) -> int:
        """Return the number of DoFs in the model."""

        return self.model.nq

    def joint_names(self) -> list[str]:
        """Return the names of the joints in the model."""

        return [
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(0 if self.is_fixed_base() else 1, self.number_of_joints())
        ]

    def joint_dofs(self, joint_name: str) -> int:
        """Return the number of DoFs of a joint."""

        if joint_name not in self.joint_names():
            raise ValueError(f"Joint '{joint_name}' not found")

        return self.data.joint(joint_name).qpos.size

    def joint_position(self, joint_name: str) -> npt.NDArray:
        """Return the position of a joint."""

        if joint_name not in self.joint_names():
            raise ValueError(f"Joint '{joint_name}' not found")

        return self.data.joint(joint_name).qpos

    def joint_positions(self, joint_names: list[str] | None = None) -> npt.NDArray:
        """Return the positions of the joints."""

        joint_names = joint_names if joint_names is not None else self.joint_names()

        return np.hstack(
            [self.joint_position(joint_name) for joint_name in joint_names]
        )

    def set_joint_position(
        self, joint_name: str, position: npt.NDArray | float
    ) -> None:
        """Set the position of a joint."""

        position = np.atleast_1d(np.array(position).squeeze())

        if position.size != self.joint_dofs(joint_name=joint_name):
            raise ValueError(
                f"Wrong position size ({position.size}) of "
                f"{self.joint_dofs(joint_name=joint_name)}-DoFs joint '{joint_name}'."
            )

        idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        offset = self.model.jnt_qposadr[idx]

        sl = np.s_[offset : offset + self.joint_dofs(joint_name=joint_name)]
        self.data.qpos[sl] = position

    def set_joint_positions(
        self, joint_names: Sequence[str], positions: npt.NDArray | list[npt.NDArray]
    ) -> None:
        """Set the positions of multiple joints."""

        mask = self.mask_qpos(joint_names=tuple(joint_names))
        self.data.qpos[mask] = positions

    # ==================
    # Methods for bodies
    # ==================

    def number_of_bodies(self) -> int:
        """Return the number of bodies in the model."""

        return self.model.nbody

    def body_names(self) -> list[str]:
        """Return the names of the bodies in the model."""

        return [
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, idx)
            for idx in range(self.number_of_bodies())
        ]

    def body_position(self, body_name: str) -> npt.NDArray:
        """Return the position of a body."""

        if body_name not in self.body_names():
            raise ValueError(f"Body '{body_name}' not found")

        return self.data.body(body_name).xpos

    def body_orientation(self, body_name: str, dcm: bool = False) -> npt.NDArray:
        """Return the orientation of a body."""

        if body_name not in self.body_names():
            raise ValueError(f"Body '{body_name}' not found")

        return (
            self.data.body(body_name).xmat if dcm else self.data.body(body_name).xquat
        )

    # ======================
    # Methods for geometries
    # ======================

    def number_of_geometries(self) -> int:
        """Return the number of geometries in the model."""

        return self.model.ngeom

    def geometry_names(self) -> list[str]:
        """Return the names of the geometries in the model."""

        return [
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, idx)
            for idx in range(self.number_of_geometries())
        ]

    def geometry_position(self, geometry_name: str) -> npt.NDArray:
        """Return the position of a geometry."""

        if geometry_name not in self.geometry_names():
            raise ValueError(f"Geometry '{geometry_name}' not found")

        return self.data.geom(geometry_name).xpos

    def geometry_orientation(
        self, geometry_name: str, dcm: bool = False
    ) -> npt.NDArray:
        """Return the orientation of a geometry."""

        if geometry_name not in self.geometry_names():
            raise ValueError(f"Geometry '{geometry_name}' not found")

        R = np.reshape(self.data.geom(geometry_name).xmat, newshape=(3, 3))

        if dcm:
            return R

        q_xyzw = Rotation.from_matrix(R).as_quat(canonical=True, scalar_first=False)
        return q_xyzw

    # ===============
    # Private methods
    # ===============

    def _mask_qpos(self, joint_names: tuple[str, ...]) -> npt.NDArray:
        """
        Create a mask to access the DoFs of the desired `joint_names` in the `qpos` array.

        Args:
            joint_names: A tuple containing the names of the joints.

        Returns:
            A 1D array containing the indices of the `qpos` array to access the DoFs of
            the desired `joint_names`.

        Note:
            This method takes a tuple of strings because we cache the output mask for
            each combination of joint names. We need a hashable object for the cache.
        """

        # Get the indices of the joints in `joint_names`.
        idxs = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in joint_names
        ]

        # We first get the index of each joint in the qpos array, and for those that
        # have multiple DoFs, we expand their mask by appending new elements.
        # Finally, we flatten the list of arrays to a single array, that is the
        # final qpos mask accessing all the DoFs of the desired `joint_names`.
        return np.atleast_1d(
            np.hstack(
                [
                    np.array(
                        [
                            self.model.jnt_qposadr[idx] + i
                            for i in range(self.joint_dofs(joint_name=joint_name))
                        ]
                    )
                    for idx, joint_name in zip(idxs, joint_names, strict=True)
                ]
            ).squeeze()
        )


def generate_hfield(
    heightmap: HeightmapCallable,
    samples_xy: tuple[int, int] = (11, 11),
    radius_xy: tuple[float, float] = (1.0, 1.0),
) -> npt.NDArray:
    """
    Generate an array with elevation points sampled from a heightmap function.

    The map will have the following format:
    ```
    heightmap[0, 0] heightmap[0, 1] ... heightmap[0, size[1]-1]
    heightmap[1, 0] heightmap[1, 1] ... heightmap[1, size[1]-1]
    ...
    heightmap[size[0]-1, 0] heightmap[size[0]-1, 1] ... heightmap[size[0]-1, size[1]-1]
    ```

    Args:
        heightmap:
            A function that takes two arguments (x, y) and returns the height
            at that point.
        samples_xy: A tuple of two integers representing the size of the grid.
        radius_xy:
            A tuple of two floats representing extension of the heightmap in the
            x-y surface corresponding to the area over which the grid of the sampled
            heightmap is generated.

    Returns:
        A flat array of the sampled terrain heightmap.
    """

    # Generate the grid.
    x = np.linspace(-radius_xy[0], radius_xy[0], samples_xy[0])
    y = np.linspace(-radius_xy[1], radius_xy[1], samples_xy[1])

    # Generate the heightmap.
    return np.array([[heightmap(xi, yi) for xi in x] for yi in y]).flatten()
