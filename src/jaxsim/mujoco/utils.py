from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco as mj
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from .model import MujocoModelHelper


def mujoco_data_from_jaxsim(
    mujoco_model: mj.MjModel,
    jaxsim_model,
    jaxsim_data,
    mujoco_data: mj.MjData | None = None,
    update_removed_joints: bool = True,
) -> mj.MjData:
    """
    Create a Mujoco data object from a JaxSim model and data objects.

    Args:
        mujoco_model: The Mujoco model object corresponding to the JaxSim model.
        jaxsim_model: The JaxSim model object from which the Mujoco model was created.
        jaxsim_data: The JaxSim data object containing the state of the model.
        mujoco_data: An optional Mujoco data object. If None, a new one will be created.
        update_removed_joints:
            If True, the positions of the joints that have been removed during the
            model reduction process will be set to their initial values.

    Returns:
        The Mujoco data object containing the state of the JaxSim model.

    Note:
        This method is useful to initialize a Mujoco data object used for visualization
        with the state of a JaxSim model. In particular, this function takes care of
        initializing the positions of the joints that have been removed during the
        model reduction process. After the initial creation of the Mujoco data object,
        it's faster to update the state using an external MujocoModelHelper object.
    """

    # The package `jaxsim.mujoco` is supposed to be jax-independent.
    # We import all the JaxSim resources privately.
    import jaxsim.api as js

    if not isinstance(jaxsim_model, js.model.JaxSimModel):
        raise ValueError("The `jaxsim_model` argument must be a JaxSimModel object.")

    if not isinstance(jaxsim_data, js.data.JaxSimModelData):
        raise ValueError("The `jaxsim_data` argument must be a JaxSimModelData object.")

    # Create the helper to operate on the Mujoco model and data.
    model_helper = MujocoModelHelper(model=mujoco_model, data=mujoco_data)

    # If the model is fixed-base, the Mujoco model won't have the joint corresponding
    # to the floating base, and the helper would raise an exception.
    if jaxsim_model.floating_base():

        # Set the model position.
        model_helper.set_base_position(position=np.array(jaxsim_data.base_position))

        # Set the model orientation.
        model_helper.set_base_orientation(
            orientation=np.array(jaxsim_data.base_orientation)
        )

    # Set the joint positions.
    if jaxsim_model.dofs() > 0:

        model_helper.set_joint_positions(
            joint_names=list(jaxsim_model.joint_names()),
            positions=np.array(jaxsim_data.joint_positions),
        )

    # Updating these joints is not necessary after the first time.
    # Users can disable this update after initialization.
    if update_removed_joints:

        # Create a dictionary with the joints that have been removed for various reasons
        # (like link lumping due to model reduction).
        joints_removed_dict = {
            j.name: j
            for j in jaxsim_model.description._joints_removed
            if j.name not in set(jaxsim_model.joint_names())
        }

        # Set the positions of the removed joints.
        _ = [
            model_helper.set_joint_position(
                position=joints_removed_dict[joint_name].initial_position,
                joint_name=joint_name,
            )
            # Select all original joint that have been removed from the JaxSim model
            # that are still present in the Mujoco model.
            for joint_name in joints_removed_dict
            if joint_name in model_helper.joint_names()
        ]

    # Return the mujoco data with updated kinematics.
    mj.mj_forward(mujoco_model, model_helper.data)

    return model_helper.data


@dataclasses.dataclass
class MujocoCamera:
    """
    Helper class storing parameters of a Mujoco camera.

    Refer to the official documentation for more details:
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
    """

    mode: str = "fixed"

    target: str | None = None
    fovy: str = "45"
    pos: str = "0 0 0"

    quat: str | None = None
    axisangle: str | None = None
    xyaxes: str | None = None
    zaxis: str | None = None
    euler: str | None = None

    name: str | None = None

    @classmethod
    def build(cls, **kwargs) -> MujocoCamera:
        """
        Build a Mujoco camera from a dictionary.
        """

        if not all(isinstance(value, str) for value in kwargs.values()):
            raise ValueError(f"Values must be strings: {kwargs}")

        return cls(**kwargs)

    @staticmethod
    def build_from_target_view(
        camera_name: str,
        mode: str = "fixed",
        lookat: Sequence[float | int] | npt.NDArray = (0, 0, 0),
        distance: float | int | npt.NDArray = 3,
        azimuth: float | int | npt.NDArray = 90,
        elevation: float | int | npt.NDArray = -45,
        fovy: float | int | npt.NDArray = 45,
        degrees: bool = True,
        **kwargs,
    ) -> MujocoCamera:
        """
        Create a custom camera that looks at a target point.

        Note:
            The choice of the parameters is easier if we imagine to consider a target
            frame `T` whose origin is located over the lookat point and having the same
            orientation of the world frame `W`. We also introduce a camera frame `C`
            whose origin is located over the lower-left corner of the image, and having
            the x-axis pointing right and the y-axis pointing up in image coordinates.
            The camera renders what it sees in the -z direction of frame `C`.

        Args:
            camera_name: The name of the camera.
            mode: Camera positioning mode:
                - **"fixed"**: Fixed position and orientation relative to the body.
                - **"track"**: Fixed offset from the body in world coordinates, constant orientation.
                - **"trackcom"**: Like `"track"`, but relative to the center of mass of the subtree.
                - **"targetbody"**: Fixed position in body frame, oriented toward a target body.
                - **"targetbodycom"**: Like `"targetbody"`, but targets the subtree's center of mass.
            lookat: The target point to look at (origin of `T`).
            distance:
                The distance from the target point (displacement between the origins
                of `T` and `C`).
            azimuth:
                The rotation around z of the camera. With an angle of 0, the camera
                would loot at the target point towards the positive x-axis of `T`.
            elevation:
                The rotation around the x-axis of the camera frame `C`. Note that if
                you want to lift the view angle, the elevation is negative.
            fovy: The field of view of the camera.
            degrees: Whether the angles are in degrees or radians.
            **kwargs: Additional camera parameters.

        Returns:
            The custom camera.
        """

        # Start from a frame whose origin is located over the lookat point.
        # We initialize a -90 degrees rotation around the z-axis because due to
        # the default camera coordinate system (x pointing right, y pointing up).
        W_H_C = np.eye(4)
        W_H_C[0:3, 3] = np.array(lookat)
        W_H_C[0:3, 0:3] = Rotation.from_euler(
            seq="ZX", angles=[-90, 90], degrees=True
        ).as_matrix()

        # Process the azimuth.
        R_az = Rotation.from_euler(seq="Y", angles=azimuth, degrees=degrees).as_matrix()
        W_H_C[0:3, 0:3] = W_H_C[0:3, 0:3] @ R_az

        # Process elevation.
        R_el = Rotation.from_euler(
            seq="X", angles=elevation, degrees=degrees
        ).as_matrix()
        W_H_C[0:3, 0:3] = W_H_C[0:3, 0:3] @ R_el

        # Process distance.
        tf_distance = np.eye(4)
        tf_distance[2, 3] = distance
        W_H_C = W_H_C @ tf_distance

        # Extract the position and the quaternion.
        p = W_H_C[0:3, 3]
        Q = Rotation.from_matrix(W_H_C[0:3, 0:3]).as_quat(scalar_first=True)

        return MujocoCamera.build(
            name=camera_name,
            mode=mode,
            fovy=str(fovy if degrees else np.rad2deg(fovy)),
            pos=" ".join(p.astype(str).tolist()),
            quat=" ".join(Q.astype(str).tolist()),
            **kwargs,
        )

    def asdict(self) -> dict[str, str]:
        """
        Convert the camera to a dictionary.
        """
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
