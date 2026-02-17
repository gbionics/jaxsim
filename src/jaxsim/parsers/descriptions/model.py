from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence

from jaxsim import logging
from jaxsim.logging import jaxsim_warn

from ..kinematic_graph import KinematicGraph, KinematicGraphTransforms, RootPose
from .collision import CollidablePoint, CollisionShape
from .joint import JointDescription
from .link import LinkDescription


@dataclasses.dataclass(frozen=True, eq=False, unsafe_hash=False)
class ModelDescription(KinematicGraph):
    """
    Intermediate representation representing the kinematic graph of a robot model.

    Attributes:
        name: The name of the model.
        fixed_base: Whether the model is either fixed-base or floating-base.
        collision_shapes: List of collision shapes associated with the model.
    """

    name: str = None

    fixed_base: bool = True

    collision_shapes: tuple[CollisionShape, ...] = dataclasses.field(
        default_factory=list, repr=False
    )

    @staticmethod
    def build_model_from(
        name: str,
        links: list[LinkDescription],
        joints: list[JointDescription],
        frames: list[LinkDescription] | None = None,
        collisions: tuple[CollisionShape, ...] = (),
        fixed_base: bool = False,
        base_link_name: str | None = None,
        considered_joints: Sequence[str] | None = None,
        model_pose: RootPose = RootPose(),
    ) -> ModelDescription:
        """
        Build a model description from provided components.

        Args:
            name: The name of the model.
            links: List of link descriptions.
            joints: List of joint descriptions.
            frames: List of frame descriptions.
            collisions: List of collision shapes associated with the model.
            fixed_base: Indicates whether the model has a fixed base.
            base_link_name: Name of the base link (i.e. the root of the kinematic tree).
            considered_joints: List of joint names to consider (by default all joints).
            model_pose: Pose of the model's root (by default an identity transform).

        Returns:
            A ModelDescription instance representing the model.
        """

        # Create the full kinematic graph.
        kinematic_graph = KinematicGraph.build_from(
            links=links,
            joints=joints,
            frames=frames,
            root_link_name=base_link_name,
            root_pose=model_pose,
        )

        # Reduce the graph if needed.
        if considered_joints is not None:
            kinematic_graph = kinematic_graph.reduce(
                considered_joints=considered_joints
            )

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=kinematic_graph)

        # Container of the final model's collision shapes.
        final_collisions: list[CollisionShape] = []

        # Move and express the collision shapes of removed links to the resulting
        # lumped link that replace the combination of the removed link and its parent.
        for collision_shape in collisions:

            # Assume they have an unique parent link
            if (
                len({cp.parent_link.name for cp in collision_shape.collidable_points})
                != 1
            ):
                msg = "Collision shape not currently supported (multiple parent links)"
                raise RuntimeError(msg)

            # Get the parent link of the collision shape.
            # Note that this link could have been lumped and we need to find the
            # link in which it was lumped into.
            parent_link_of_shape = collision_shape.collidable_points[0].parent_link

            # If it is part of the (reduced) graph, add it as it is...
            if parent_link_of_shape.name in kinematic_graph.link_names():
                final_collisions.append(collision_shape)
                continue

            # ... otherwise look for the frame
            if parent_link_of_shape.name not in kinematic_graph.frame_names():
                msg = "Parent frame '{}' of collision shape not found, ignoring shape"
                logging.info(msg.format(parent_link_of_shape.name))
                continue

            # Create a new collision shape
            new_collision_shape = CollisionShape(collidable_points=())
            final_collisions.append(new_collision_shape)

            # If the frame was found, update the collidable points' pose and add them
            # to the new collision shape.
            for cp in collision_shape.collidable_points:
                # Find the link that is part of the (reduced) model in which the
                # collision shape's parent was lumped into
                real_parent_link_name = kinematic_graph.frames_dict[
                    parent_link_of_shape.name
                ].parent_name

                # Change the link associated to the collidable point, updating their
                # relative pose
                moved_cp = cp.change_link(
                    new_link=kinematic_graph.links_dict[real_parent_link_name],
                    new_H_old=fk.relative_transform(
                        relative_to=real_parent_link_name,
                        name=cp.parent_link.name,
                    ),
                )

                # Store the updated collision.
                new_collision_shape.collidable_points += (moved_cp,)

        # Build the model
        model = ModelDescription(
            name=name,
            root_pose=kinematic_graph.root_pose,
            fixed_base=fixed_base,
            collision_shapes=tuple(final_collisions),
            root=kinematic_graph.root,
            joints=kinematic_graph.joints,
            frames=kinematic_graph.frames,
            _joints_removed=kinematic_graph.joints_removed,
        )

        # Check that the root link of kinematic graph is the desired base link.
        assert kinematic_graph.root.name == base_link_name, kinematic_graph.root.name

        return model

    def reduce(self, considered_joints: Sequence[str]) -> ModelDescription:
        """
        Reduce the model by removing specified joints.

        Args:
            considered_joints: Sequence of joint names to consider.

        Returns:
            A `ModelDescription` instance that only includes the considered joints.
        """

        jaxsim_warn(
            "The joint order in the model description is not preserved when reducing "
            "the model. Consider using the `names_to_indices` method to get the correct "
            "order of the joints, or use the `joint_names()` method to inspect the internal joint ordering."
        )

        if set(considered_joints) - set(self.joint_names()):
            extra_joints = set(considered_joints) - set(self.joint_names())
            msg = f"Found joints not part of the model: {extra_joints}"
            raise ValueError(msg)

        reduced_model_description = ModelDescription.build_model_from(
            name=self.name,
            links=list(self.links_dict.values()),
            joints=self.joints,
            frames=self.frames,
            collisions=self.collision_shapes,
            fixed_base=self.fixed_base,
            base_link_name=next(iter(self)).name,
            model_pose=self.root_pose,
            considered_joints=considered_joints,
        )

        # Include the unconnected/removed joints from the original model.
        for joint in self.joints_removed:
            reduced_model_description.joints_removed.append(joint)

        return reduced_model_description

    def update_collision_shape_of_link(self, link_name: str, enabled: bool) -> None:
        """
        Enable or disable collision shapes associated with a link.

        Args:
            link_name: The name of the link.
            enabled: Enable or disable collision shapes associated with the link.
        """

        if link_name not in self.link_names():
            raise ValueError(link_name)

        for point in self.collision_shape_of_link(
            link_name=link_name
        ).collidable_points:
            point.enabled = enabled

    def collision_shape_of_link(self, link_name: str) -> CollisionShape:
        """
        Get the collision shape associated with a specific link.

        Args:
            link_name: The name of the link.

        Returns:
            The collision shape associated with the link.
        """

        if link_name not in self.link_names():
            raise ValueError(link_name)

        return CollisionShape(
            collidable_points=[
                point
                for shape in self.collision_shapes
                for point in shape.collidable_points
                if point.parent_link.name == link_name
            ]
        )

    def all_enabled_collidable_points(self) -> list[CollidablePoint]:
        """
        Get all enabled collidable points in the model.

        Returns:
            The list of all enabled collidable points.

        """

        # Get iterator of all collidable points
        all_collidable_points = itertools.chain.from_iterable(
            [shape.collidable_points for shape in self.collision_shapes]
        )

        # Return enabled collidable points
        return [cp for cp in all_collidable_points if cp.enabled]

    def __eq__(self, other: ModelDescription) -> bool:

        if not isinstance(other, ModelDescription):
            return False

        if not (
            self.name == other.name
            and self.fixed_base == other.fixed_base
            and self.root == other.root
            and self.joints == other.joints
            and self.frames == other.frames
            and self.root_pose == other.root_pose
        ):
            return False

        return True

    def __hash__(self) -> int:

        return hash(
            (
                hash(self.name),
                hash(self.fixed_base),
                hash(self.root),
                hash(tuple(self.joints)),
                hash(tuple(self.frames)),
                hash(self.root_pose),
            )
        )
