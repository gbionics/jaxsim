from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxlie
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.math import Rotation
from jaxsim.parsers.descriptions import JointGenericAxis, JointType, ModelDescription
from jaxsim.parsers.kinematic_graph import KinematicGraphTransforms
from jaxsim.utils.jaxsim_dataclass import JaxsimDataclass


@jax_dataclasses.pytree_dataclass
class JointModel(JaxsimDataclass):
    """
    Class describing the joint kinematics of a robot model.

    Attributes:
        λ_H_pre:
            The homogeneous transformation between the parent link and
            the predecessor frame of each joint.
        suc_H_i:
            The homogeneous transformation between the successor frame and
            the child link of each joint.
        joint_dofs: The number of DoFs of each joint.
        joint_names: The names of each joint.
        joint_types: The types of each joint.

    Note:
        Due to the presence of the static attributes, this class needs to be created
        already in a vectorized form. In other words, it cannot be created using vmap.
    """

    λ_H_pre: jtp.Array
    suc_H_i: jtp.Array

    joint_dofs: Static[tuple[int, ...]]
    joint_names: Static[tuple[str, ...]]
    joint_types: Static[tuple[int, ...]]
    joint_axis: Static[tuple[JointGenericAxis, ...]]

    @staticmethod
    def build(description: ModelDescription) -> JointModel:
        """
        Build the joint model of a model description.

        Args:
            description: The model description to consider.

        Returns:
            The joint model of the considered model description.
        """

        # Note: the joint index is equal to its child link index, therefore it
        # starts from 1.
        ordered_joints = sorted(
            list(description.joints_dict.values()),
            key=lambda j: j.index,
        )

        # Allocate the parent-to-predecessor and successor-to-child transforms.
        λ_H_pre = jnp.zeros(shape=(1 + len(ordered_joints), 4, 4), dtype=float)
        suc_H_i = jnp.zeros(shape=(1 + len(ordered_joints), 4, 4), dtype=float)

        # Initialize an identical parent-to-predecessor transform for the joint
        # between the world frame W and the base link B.
        λ_H_pre = λ_H_pre.at[0].set(jnp.eye(4))

        # Initialize an identical successor-to-child transform for the joint
        # between the world frame W and the base link B.
        suc_H_i = suc_H_i.at[0].set(jnp.eye(4))

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=description)

        # Compute the parent-to-predecessor and successor-to-child transforms for
        # each joint belonging to the model.
        # Note that the joint indices starts from i=1 given our joint model,
        # therefore the entries at index 0 are not updated.
        for joint in ordered_joints:
            λ_H_pre = λ_H_pre.at[joint.index].set(
                fk.relative_transform(relative_to=joint.parent.name, name=joint.name)
            )
            suc_H_i = suc_H_i.at[joint.index].set(
                fk.relative_transform(relative_to=joint.name, name=joint.child.name)
            )

        # Define the DoFs of the base link.
        base_dofs = 0 if description.fixed_base else 6

        # We always add a dummy fixed joint between world and base.
        # TODO: Port floating-base support also at this level, not only in RBDAs.
        return JointModel(
            λ_H_pre=λ_H_pre,
            suc_H_i=suc_H_i,
            # Static attributes
            joint_dofs=tuple([base_dofs] + [1 for _ in ordered_joints]),
            joint_names=tuple(["world_to_base"] + [j.name for j in ordered_joints]),
            joint_types=tuple([JointType.Fixed] + [j.jtype for j in ordered_joints]),
            joint_axis=tuple(JointGenericAxis(axis=j.axis) for j in ordered_joints),
        )

    def parent_H_predecessor(self, joint_index: jtp.IntLike) -> jtp.Matrix:
        r"""
        Return the homogeneous transformation between the parent link and
        the predecessor frame of a joint.

        Args:
            joint_index: The index of the joint.

        Returns:
            The homogeneous transformation
            :math:`{}^{\lambda(i)} \mathbf{H}_{\text{pre}(i)}`.
        """

        return self.λ_H_pre[joint_index]

    def successor_H_child(self, joint_index: jtp.IntLike) -> jtp.Matrix:
        r"""
        Return the homogeneous transformation between the successor frame and
        the child link of a joint.

        Args:
            joint_index: The index of the joint.

        Returns:
            The homogeneous transformation
            :math:`{}^{\text{suc}(i)} \mathbf{H}_i`.
        """

        return self.suc_H_i[joint_index]


@jax.jit
def supported_joint_motion(
    joint_types: jtp.Array, joint_positions: jtp.Matrix, joint_axes: jtp.Matrix
) -> jtp.Matrix:
    """
    Compute the transforms of the joints.

    Args:
        joint_types: The types of the joints.
        joint_positions: The positions of the joints.
        joint_axes: The axes of the joints.

    Returns:
        The transforms of the joints.
    """

    # Prepare the joint position
    s = jnp.array(joint_positions).astype(float)

    def compute_F() -> tuple[jtp.Matrix, jtp.Array]:
        return jaxlie.SE3.identity()

    def compute_R() -> tuple[jtp.Matrix, jtp.Array]:

        # Get the additional argument specifying the joint axis.
        # This is a metadata required by only some joint types.
        axis = jnp.array(joint_axes).astype(float).squeeze()

        pre_H_suc = jaxlie.SE3.from_matrix(
            matrix=jnp.eye(4).at[:3, :3].set(Rotation.from_axis_angle(vector=s * axis))
        )

        return pre_H_suc

    def compute_P() -> tuple[jtp.Matrix, jtp.Array]:

        # Get the additional argument specifying the joint axis.
        # This is a metadata required by only some joint types.
        axis = jnp.array(joint_axes).astype(float).squeeze()

        pre_H_suc = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.identity(),
            translation=jnp.array(s * axis),
        )

        return pre_H_suc

    return jax.lax.switch(
        index=joint_types,
        branches=(
            compute_F,  # JointType.Fixed
            compute_R,  # JointType.Revolute
            compute_P,  # JointType.Prismatic
        ),
    ).as_matrix()
