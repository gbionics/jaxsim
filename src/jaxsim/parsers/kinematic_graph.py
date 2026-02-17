from __future__ import annotations

import copy
import dataclasses
import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import jaxsim.utils
from jaxsim import logging
from jaxsim.utils import Mutability

from .descriptions.joint import JointDescription, JointType
from .descriptions.link import LinkDescription


@dataclasses.dataclass
class RootPose:
    """
    Represents the root pose in a kinematic graph.

    Attributes:
        root_position: The 3D position of the root link of the graph.
        root_quaternion:
            The quaternion representing the rotation of the root link of the graph.

    Note:
        The root link of the kinematic graph is the base link.
    """

    root_position: npt.NDArray = dataclasses.field(default_factory=lambda: np.zeros(3))

    root_quaternion: npt.NDArray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 0, 0, 0])
    )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                HashedNumpyArray.hash_of_array(self.root_position),
                HashedNumpyArray.hash_of_array(self.root_quaternion),
            )
        )

    def __eq__(self, other: RootPose) -> bool:

        if not isinstance(other, RootPose):
            return False

        if not np.allclose(self.root_position, other.root_position):
            return False

        if not np.allclose(self.root_quaternion, other.root_quaternion):
            return False

        return True


@dataclasses.dataclass(frozen=True)
class KinematicGraph(Sequence[LinkDescription]):
    """
    Class storing a kinematic graph having links as nodes and joints as edges.

    Attributes:
        root: The root node of the kinematic graph.
        frames: List of frames rigidly attached to the graph nodes.
        joints: List of joints connecting the graph nodes.
        root_pose: The pose of the kinematic graph's root.
    """

    root: LinkDescription
    frames: list[LinkDescription] = dataclasses.field(
        default_factory=list, hash=False, compare=False
    )
    joints: list[JointDescription] = dataclasses.field(
        default_factory=list, hash=False, compare=False
    )

    root_pose: RootPose = dataclasses.field(default_factory=RootPose)

    # Private attribute storing optional additional info.
    _extra_info: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, hash=False, compare=False
    )

    # Private attribute storing the unconnected joints from the parsed model and
    # the joints removed after model reduction.
    _joints_removed: list[JointDescription] = dataclasses.field(
        default_factory=list, repr=False, hash=False, compare=False
    )

    @functools.cached_property
    def links_dict(self) -> dict[str, LinkDescription]:
        """
        Get a dictionary of links indexed by their name.
        """
        return {l.name: l for l in iter(self)}

    @functools.cached_property
    def frames_dict(self) -> dict[str, LinkDescription]:
        """
        Get a dictionary of frames indexed by their name.
        """
        return {f.name: f for f in self.frames}

    @functools.cached_property
    def joints_dict(self) -> dict[str, JointDescription]:
        """
        Get a dictionary of joints indexed by their name.
        """
        return {j.name: j for j in self.joints}

    @functools.cached_property
    def joints_connection_dict(
        self,
    ) -> dict[tuple[str, str], JointDescription]:
        """
        Get a dictionary of joints indexed by the tuple (parent, child) link names.
        """
        return {(j.parent.name, j.child.name): j for j in self.joints}

    def __post_init__(self) -> None:

        # Assign the link index by traversing the graph with BFS.
        # Here we assume the model being fixed-base, therefore the base link will
        # have index 0. We will deal with the floating base in a later stage.
        for index, link in enumerate(self):
            link.mutable(validate=False).index = index

        # Get the names of the links, frames, and joints.
        link_names = [l.name for l in self]
        frame_names = [f.name for f in self.frames]
        joint_names = [j.name for j in self.joints]

        # Make sure that they are unique.
        assert len(link_names) == len(set(link_names))
        assert len(frame_names) == len(set(frame_names))
        assert len(joint_names) == len(set(joint_names))
        assert set(link_names).isdisjoint(set(frame_names))
        assert set(link_names).isdisjoint(set(joint_names))

        # Order frames with their name.
        super().__setattr__("frames", sorted(self.frames, key=lambda f: f.name))

        # Assign the frame index following the name-based indexing.
        # We assume the model being fixed-base, therefore the first frame will
        # have last_link_idx + 1.
        for index, frame in enumerate(self.frames):
            with frame.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                frame.index = index + len(self.link_names())

        # Number joints so that their index matches their child link index.
        # Therefore, the first joint has index 1.
        links_dict = {l.name: l for l in iter(self)}
        for joint in self.joints:
            with joint.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                joint.index = links_dict[joint.child.name].index

        # Check that joint indices are unique.
        assert len([j.index for j in self.joints]) == len(
            {j.index for j in self.joints}
        )

        # Order joints with their indices.
        super().__setattr__("joints", sorted(self.joints, key=lambda j: j.index))

    @staticmethod
    def build_from(
        links: list[LinkDescription],
        joints: list[JointDescription],
        frames: list[LinkDescription] | None = None,
        root_link_name: str | None = None,
        root_pose: RootPose = RootPose(),
    ) -> KinematicGraph:
        """
        Build a KinematicGraph from links, joints, and frames.

        Args:
            links: A list of link descriptions.
            joints: A list of joint descriptions.
            frames: A list of frame descriptions.
            root_link_name:
                The name of the root link. If not provided, it's assumed to be the
                first link's name.
            root_pose: The root pose of the kinematic graph.

        Returns:
            The resulting kinematic graph.
        """

        # Consider the first link as the root link if not provided.
        if root_link_name is None:
            root_link_name = links[0].name
            logging.debug(msg=f"Assuming '{root_link_name}' as the root link")

        # Couple links and joints and create the graph of links.
        # Note that the pose of the frames is not updated; it is the caller's
        # responsibility to update their pose if they want to use them.
        (
            graph_root_node,
            graph_joints,
            graph_frames,
            unconnected_links,
            unconnected_joints,
            unconnected_frames,
        ) = KinematicGraph._create_graph(
            links=links, joints=joints, root_link_name=root_link_name, frames=frames
        )

        for link in unconnected_links:
            logging.warning(msg=f"Ignoring unconnected link: '{link.name}'")

        for joint in unconnected_joints:
            logging.warning(msg=f"Ignoring unconnected joint: '{joint.name}'")

        for frame in unconnected_frames:
            logging.warning(msg=f"Ignoring unconnected frame: '{frame.name}'")

        return KinematicGraph(
            root=graph_root_node,
            joints=graph_joints,
            frames=graph_frames,
            root_pose=root_pose,
            _joints_removed=unconnected_joints,
        )

    @staticmethod
    def _create_graph(
        links: list[LinkDescription],
        joints: list[JointDescription],
        root_link_name: str,
        frames: list[LinkDescription] | None = None,
    ) -> tuple[
        LinkDescription,
        list[JointDescription],
        list[LinkDescription],
        list[LinkDescription],
        list[JointDescription],
        list[LinkDescription],
    ]:
        """
        Low-level creator of kinematic graph components.

        Args:
            links: A list of parsed link descriptions.
            joints: A list of parsed joint descriptions.
            root_link_name: The name of the root link used as root node of the graph.
            frames: A list of parsed frame descriptions.

        Returns:
            A tuple containing the root node of the graph (defining the entire kinematic
            tree by iterating on its child nodes), the list of joints representing the
            actual graph edges, the list of frames rigidly attached to the graph nodes,
            the list of unconnected links, the list of unconnected joints, and the list
            of unconnected frames.
        """

        # Create a dictionary that maps the link name to the link, for easy retrieval.
        links_dict: dict[str, LinkDescription] = {
            l.name: l.mutable(validate=False) for l in links
        }

        # Create an empty list of frames if not provided.
        frames = frames if frames is not None else []

        # Create a dictionary that maps the frame name to the frame, for easy retrieval.
        frames_dict = {frame.name: frame for frame in frames}

        # Check that our parser correctly resolved the frame's parent to be a link.
        for frame in frames:
            assert frame.parent_name != "", frame
            assert frame.parent_name is not None, frame
            assert frame.parent_name != "__model__", frame
            assert frame.parent_name not in frames_dict, frame

        # ===========================================================
        # Populate the kinematic graph with links, joints, and frames
        # ===========================================================

        # Check the existence of the root link.
        if root_link_name not in links_dict:
            raise ValueError(root_link_name)

        # Reset the connections of the root link.
        for link in links_dict.values():
            link.children = tuple()

        # Couple links and joints creating the kinematic graph.
        for joint in joints:

            # Get the parent and child links of the joint.
            parent_link = links_dict[joint.parent.name]
            child_link = links_dict[joint.child.name]

            assert child_link.name == joint.child.name
            assert parent_link.name == joint.parent.name

            # Assign link's parent.
            child_link.parent_name = parent_link.name

            # Assign link's children and make sure they are unique.
            if child_link.name not in {l.name for l in parent_link.children}:
                with parent_link.mutable_context(Mutability.MUTABLE_NO_VALIDATION):
                    parent_link.children = (*parent_link.children, child_link)

        # Collect all the links of the kinematic graph.
        all_links_in_graph = list(
            KinematicGraph.breadth_first_search(root=links_dict[root_link_name])
        )

        # Get the names of all links in the kinematic graph.
        all_link_names_in_graph = [l.name for l in all_links_in_graph]

        # Collect all the joints of the kinematic graph.
        all_joints_in_graph = [
            joint
            for joint in joints
            if joint.parent.name in all_link_names_in_graph
            and joint.child.name in all_link_names_in_graph
        ]

        # Get the names of all joints in the kinematic graph.
        all_joint_names_in_graph = [j.name for j in all_joints_in_graph]

        # Collect all the frames of the kinematic graph.
        # Note: our parser ensures that the parent of a frame is not another frame.
        all_frames_in_graph = [
            frame for frame in frames if frame.parent_name in all_link_names_in_graph
        ]

        # Get the names of all frames in the kinematic graph.
        all_frames_names_in_graph = [f.name for f in all_frames_in_graph]

        # ============================
        # Collect unconnected elements
        # ============================

        # Collect all the joints that are not part of the kinematic graph.
        removed_joints = [j for j in joints if j.name not in all_joint_names_in_graph]

        for joint in removed_joints:
            msg = "Joint '{}' is unconnected and it will be removed"
            logging.debug(msg=msg.format(joint.name))

        # Collect all the links that are not part of the kinematic graph.
        unconnected_links = [l for l in links if l.name not in all_link_names_in_graph]

        # Update the unconnected links by removing their children. The other properties
        # are left untouched, it's caller responsibility to post-process them if needed.
        for link in unconnected_links:
            link.children = tuple()
            msg = "Link '{}' won't be part of the kinematic graph because unconnected"
            logging.debug(msg=msg.format(link.name))

        # Collect all the frames that are not part of the kinematic graph.
        unconnected_frames = [
            f for f in frames if f.name not in all_frames_names_in_graph
        ]

        for frame in unconnected_frames:
            msg = "Frame '{}' won't be part of the kinematic graph because unconnected"
            logging.debug(msg=msg.format(frame.name))

        return (
            links_dict[root_link_name].mutable(mutable=False),
            list(set(joints) - set(removed_joints)),
            all_frames_in_graph,
            unconnected_links,
            list(set(removed_joints)),
            unconnected_frames,
        )

    def reduce(self, considered_joints: Sequence[str]) -> KinematicGraph:
        """
        Reduce the kinematic graph by removing unspecified joints.

        When a joint is removed, the mass and inertia of its child link are lumped
        with those of its parent link, obtaining a new link that combines the two.
        The description of the removed joint specifies the default angle (usually 0)
        that is considered when the joint is removed.

        Args:
            considered_joints: A list of joint names to consider.

        Returns:
            The reduced kinematic graph.
        """

        # The current object represents the complete kinematic graph
        full_graph = self

        # Get the names of the joints to remove
        joint_names_to_remove = list(
            set(full_graph.joint_names()) - set(considered_joints)
        )

        # Return early if there is no action to take
        if len(joint_names_to_remove) == 0:
            logging.info("The kinematic graph doesn't need to be reduced")
            return copy.deepcopy(self)

        # Check if all considered joints are part of the full kinematic graph
        if set(considered_joints) - {j.name for j in full_graph.joints}:
            extra_j = set(considered_joints) - {j.name for j in full_graph.joints}
            msg = f"Not all joints to consider are part of the graph ({{{extra_j}}})"
            raise ValueError(msg)

        # Extract data we need to modify from the full graph
        links_dict = copy.deepcopy(full_graph.links_dict)
        joints_dict = copy.deepcopy(full_graph.joints_dict)

        # Create the object to compute forward kinematics.
        fk = KinematicGraphTransforms(graph=full_graph)

        # The following steps are implemented below in order to create the reduced graph:
        #
        # 1. Lump the mass of the removed links into their parent
        # 2. Update the pose and parent link of joints having the removed link as parent
        # 3. Create the reduced graph considering the removed links as frames
        # 4. Resolve the pose of the frames wrt their reduced graph parent
        #
        # We name "removed link" the link to remove, and "lumped link" the new link that
        # combines the removed link and its parent. The lumped link will share the frame
        # of the removed link's parent and the inertial properties of the two links that
        # have been combined.

        # =======================================================
        # 1. Lump the mass of the removed links into their parent
        # =======================================================

        # Get all the links to remove. They will be lumped with their parent.
        links_to_remove = [
            joint.child.name
            for joint_name, joint in joints_dict.items()
            if joint_name in joint_names_to_remove
        ]

        # Lump the mass and the inertia traversing the tree from the leaf to the root,
        # this way we propagate these properties back even in the case when also the
        # parent link of a removed joint has to be lumped with its parent.
        for link in reversed(full_graph):
            if link.name not in links_to_remove:
                continue

            # Get the link to remove and its parent, i.e. the lumped link
            link_to_remove = links_dict[link.name]
            parent_of_link_to_remove = links_dict[link.parent_name]

            msg = "Lumping chain: {}->({})->{}"
            logging.debug(
                msg.format(
                    link_to_remove.name,
                    self.joints_connection_dict[
                        parent_of_link_to_remove.name, link_to_remove.name
                    ].name,
                    parent_of_link_to_remove.name,
                )
            )

            # Lump the link
            lumped_link = parent_of_link_to_remove.lump_with(
                link=link_to_remove,
                lumped_H_removed=fk.relative_transform(
                    relative_to=parent_of_link_to_remove.name, name=link_to_remove.name
                ),
            )

            # Pop the original two links from the dictionary...
            _ = links_dict.pop(link_to_remove.name)
            _ = links_dict.pop(parent_of_link_to_remove.name)

            # ... and insert the lumped link (having the same name of the parent)
            links_dict[lumped_link.name] = lumped_link

            # Insert back in the dict an entry from the removed link name to the new
            # lumped link. We need this info later, when we process the remaining joints.
            links_dict[link_to_remove.name] = lumped_link

            # As a consequence of the back-insertion, we need to adjust the resulting
            # lumped link of links that have been removed previously.
            # Note: in the dictionary, only items whose key is not matching value.name
            #       are links that have been removed.
            for previously_removed_link_name in {
                link_name
                for link_name, link in links_dict.items()
                if link_name != link.name and link.name == link_to_remove.name
            }:
                links_dict[previously_removed_link_name] = lumped_link

        # ==============================================================================
        # 2. Update the pose and parent link of joints having the removed link as parent
        # ==============================================================================

        # Find the joints having the removed links as parent
        joints_with_removed_parent_link = [
            joints_dict[joint_name]
            for joint_name in considered_joints
            if joints_dict[joint_name].parent.name in links_to_remove
        ]

        # Update the pose of all joints having as parent link a removed link
        for joint in joints_with_removed_parent_link:
            # Update the pose. Note that after the lumping process, the dict entry
            # links_dict[joint.parent.name] contains the final lumped link
            with joint.mutable_context(mutability=Mutability.MUTABLE):
                joint.pose = fk.relative_transform(
                    relative_to=links_dict[joint.parent.name].name, name=joint.name
                )
            with joint.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                # Update the parent link
                joint.parent = links_dict[joint.parent.name]

        # ===================================================================
        # 3. Create the reduced graph considering the removed links as frames
        # ===================================================================

        # Get all the original links from the full graph
        full_graph_links_dict = copy.deepcopy(full_graph.links_dict)

        # Get all the final links from the reduced graph
        links_to_keep = [
            l for link_name, l in links_dict.items() if link_name not in links_to_remove
        ]

        # Override the entries of the full graph with those of the reduced graph.
        # Those that are not overridden will become frames.
        for link in links_to_keep:
            full_graph_links_dict[link.name] = link

        # Create the reduced graph data. We pass the full list of links so that those
        # that are not part of the graph will be returned as frames.
        (
            reduced_root_node,
            reduced_joints,
            reduced_frames,
            unconnected_links,
            unconnected_joints,
            unconnected_frames,
        ) = KinematicGraph._create_graph(
            links=list(full_graph_links_dict.values()),
            joints=[joints_dict[joint_name] for joint_name in considered_joints],
            root_link_name=full_graph.root.name,
        )

        assert {f.name for f in self.frames}.isdisjoint(
            {f.name for f in unconnected_frames + reduced_frames}
        )

        for link in unconnected_links:
            logging.debug(msg=f"Link '{link.name}' is unconnected and became a frame")

        # Create the reduced graph.
        reduced_graph = KinematicGraph(
            root=reduced_root_node,
            joints=reduced_joints,
            frames=self.frames + unconnected_links + reduced_frames,
            root_pose=full_graph.root_pose,
            _joints_removed=(
                self._joints_removed
                + unconnected_joints
                + [joints_dict[name] for name in joint_names_to_remove]
            ),
        )

        # ================================================================
        # 4. Resolve the pose of the frames wrt their reduced graph parent
        # ================================================================

        # Build a new object to compute FK on the reduced graph.
        fk_reduced = KinematicGraphTransforms(graph=reduced_graph)

        # We need to adjust the pose of the frames since their parent link
        # could have been removed by the reduction process.
        for frame in reduced_graph.frames:

            # Always find the real parent link of the frame
            name_of_new_parent_link = fk_reduced.find_parent_link_of_frame(
                name=frame.name
            )
            assert name_of_new_parent_link in reduced_graph, name_of_new_parent_link

            # Notify the user if the parent link has changed.
            if name_of_new_parent_link != frame.parent_name:
                msg = "New parent of frame '{}' is '{}'"
                logging.debug(msg=msg.format(frame.name, name_of_new_parent_link))

            # Always recompute the pose of the frame, and set zero inertial params.
            with frame.mutable_context(jaxsim.utils.Mutability.MUTABLE_NO_VALIDATION):

                # Update kinematic parameters of the frame.
                # Note that here we compute the transform using the FK object of the
                # full model, so that we are sure that the kinematic is not altered.
                frame.pose = fk.relative_transform(
                    relative_to=name_of_new_parent_link, name=frame.name
                )

                # Update the parent link such that the pose is expressed in its frame.
                frame.parent_name = name_of_new_parent_link

                # Update dynamic parameters of the frame.
                frame.mass = 0.0
                frame.inertia = np.zeros_like(frame.inertia)

        # Return the reduced graph.
        return reduced_graph

    def link_names(self) -> list[str]:
        """
        Get the names of all links in the kinematic graph (i.e. the nodes).

        Returns:
            The list of link names.
        """
        return list(self.links_dict.keys())

    def joint_names(self) -> list[str]:
        """
        Get the names of all joints in the kinematic graph (i.e. the edges).

        Returns:
            The list of joint names.
        """
        return list(self.joints_dict.keys())

    def frame_names(self) -> list[str]:
        """
        Get the names of all frames in the kinematic graph.

        Returns:
            The list of frame names.
        """

        return list(self.frames_dict.keys())

    def print_tree(self) -> None:
        """
        Print the tree structure of the kinematic graph.
        """

        import pptree

        root_node = self.root

        pptree.print_tree(
            root_node,
            childattr="children",
            nameattr="name_and_index",
            horizontal=True,
        )

    @property
    def joints_removed(self) -> list[JointDescription]:
        """
        Get the list of joints removed during the graph reduction.

        Returns:
            The list of removed joints.
        """

        return self._joints_removed

    @staticmethod
    def breadth_first_search(
        root: LinkDescription,
        sort_children: Callable[[Any], Any] | None = lambda link: link.name,
    ) -> Iterable[LinkDescription]:
        """
        Perform a breadth-first search (BFS) traversal of the kinematic graph.

        Args:
            root: The root link for BFS.
            sort_children: A function to sort children of a node.

        Yields:
            The links in the kinematic graph in BFS order.
        """

        # Initialize the queue with the root node.
        queue = [root]

        # We assume that nodes have unique names and mark a link as visited using
        # its name. This speeds up considerably object comparison.
        visited = []
        visited.append(root.name)

        yield root

        while queue:

            # Extract the first element of the queue.
            l = queue.pop(0)

            # Note: sorting the links with their name so that the order of children
            # insertion does not matter when assigning the link index.
            for child in sorted(l.children, key=sort_children):

                if child.name in visited:
                    continue

                visited.append(child.name)
                queue.append(child)

                yield child

    # =================
    # Sequence protocol
    # =================

    def __iter__(self) -> Iterator[LinkDescription]:
        yield from KinematicGraph.breadth_first_search(root=self.root)

    def __reversed__(self) -> Iterable[LinkDescription]:
        yield from reversed(list(iter(self)))

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __contains__(self, item: str | LinkDescription) -> bool:
        if isinstance(item, str):
            return item in self.link_names()

        if isinstance(item, LinkDescription):
            return item in set(iter(self))

        raise TypeError(type(item).__name__)

    def __getitem__(self, key: int | str) -> LinkDescription:
        if isinstance(key, str):
            if key not in self.link_names():
                raise KeyError(key)

            return self.links_dict[key]

        if isinstance(key, int):
            if key > len(self):
                raise KeyError(key)

            return list(iter(self))[key]

        raise TypeError(type(key).__name__)

    def count(self, value: LinkDescription) -> int:
        """
        Count the occurrences of a link in the kinematic graph.
        """
        return list(iter(self)).count(value)

    def index(self, value: LinkDescription, start: int = 0, stop: int = -1) -> int:
        """
        Find the index of a link in the kinematic graph.
        """
        return list(iter(self)).index(value, start, stop)


# ====================
# Other useful classes
# ====================


@dataclasses.dataclass(frozen=True)
class KinematicGraphTransforms:
    """
    Class to compute forward kinematics on a kinematic graph.

    Attributes:
        graph: The kinematic graph on which to compute forward kinematics.
    """

    graph: KinematicGraph

    _transform_cache: dict[str, npt.NDArray] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    _initial_joint_positions: dict[str, float] = dataclasses.field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:

        super().__setattr__(
            "_initial_joint_positions",
            {joint.name: joint.initial_position for joint in self.graph.joints},
        )

    @property
    def initial_joint_positions(self) -> npt.NDArray:
        """
        Get the initial joint positions of the kinematic graph.
        """

        return np.atleast_1d(
            np.array(list(self._initial_joint_positions.values()))
        ).astype(float)

    @initial_joint_positions.setter
    def initial_joint_positions(
        self,
        positions: npt.NDArray | Sequence,
        joint_names: Sequence[str] | None = None,
    ) -> None:

        joint_names = (
            joint_names
            if joint_names is not None
            else list(self._initial_joint_positions.keys())
        )

        s = np.atleast_1d(np.array(positions).squeeze())

        if s.size != len(joint_names):
            raise ValueError(s.size, len(joint_names))

        for joint_name in joint_names:
            if joint_name not in self._initial_joint_positions:
                raise ValueError(joint_name)

        # Clear transform cache.
        self._transform_cache.clear()

        # Update initial joint positions.
        for joint_name, position in zip(joint_names, s, strict=True):
            self._initial_joint_positions[joint_name] = position

    def transform(self, name: str) -> npt.NDArray:
        """
        Compute the SE(3) transform of elements belonging to the kinematic graph.

        Args:
            name: The name of a link, a joint, or a frame.

        Returns:
            The 4x4 transform matrix of the element w.r.t. the model frame.
        """

        # If the transform was already computed, return it.
        if name in self._transform_cache:
            return self._transform_cache[name]

        # If the name is a joint, compute M_H_J transform.
        if name in self.graph.joint_names():

            # Get the joint.
            joint = self.graph.joints_dict[name]
            assert joint.name == name

            # Get the transform of the parent link.
            M_H_L = self.transform(name=joint.parent.name)

            # Rename the pose of the predecessor joint frame w.r.t. its parent link.
            L_H_pre = joint.pose

            # Compute the joint transform from the predecessor to the successor frame.
            pre_H_J = self.pre_H_suc(
                joint_type=joint.jtype,
                joint_axis=joint.axis,
                joint_position=self._initial_joint_positions[joint.name],
            )

            # Compute the M_H_J transform.
            self._transform_cache[name] = M_H_L @ L_H_pre @ pre_H_J
            return self._transform_cache[name]

        # If the name is a link, compute M_H_L transform.
        if name in self.graph.link_names():

            # Get the link.
            link = self.graph.links_dict[name]

            # Handle the pose between the __model__ frame and the root link.
            if link.name == self.graph.root.name:
                M_H_B = link.pose
                return M_H_B

            # Get the joint between the link and its parent.
            parent_joint = self.graph.joints_connection_dict[
                link.parent_name, link.name
            ]

            # Get the transform of the parent joint.
            M_H_J = self.transform(name=parent_joint.name)

            # Rename the pose of the link w.r.t. its parent joint.
            J_H_L = link.pose

            # Compute the M_H_L transform.
            self._transform_cache[name] = M_H_J @ J_H_L
            return self._transform_cache[name]

        # It can only be a plain frame.
        if name not in self.graph.frame_names():
            raise ValueError(name)

        # Get the frame.
        frame = self.graph.frames_dict[name]

        # Get the transform of the parent link.
        M_H_L = self.transform(name=frame.parent_name)

        # Rename the pose of the frame w.r.t. its parent link.
        L_H_F = frame.pose

        # Compute the M_H_F transform.
        self._transform_cache[name] = M_H_L @ L_H_F
        return self._transform_cache[name]

    def relative_transform(self, relative_to: str, name: str) -> npt.NDArray:
        """
        Compute the SE(3) relative transform of elements belonging to the kinematic graph.

        Args:
            relative_to: The name of the reference element.
            name: The name of a link, a joint, or a frame.

        Returns:
            The 4x4 transform matrix of the element w.r.t. the desired frame.
        """

        import jaxsim.math

        M_H_target = self.transform(name=name)
        M_H_R = self.transform(name=relative_to)

        # Compute the relative transform R_H_target, where R is the reference frame,
        # and i the frame of the desired link|joint|frame.
        return np.array(jaxsim.math.Transform.inverse(M_H_R)) @ M_H_target

    @staticmethod
    def pre_H_suc(
        joint_type: JointType,
        joint_axis: npt.NDArray,
        joint_position: float | None = None,
    ) -> npt.NDArray:
        """
        Compute the SE(3) transform from the predecessor to the successor frame.

        Args:
            joint_type: The type of the joint.
            joint_axis: The axis of the joint.
            joint_position: The position of the joint.

        Returns:
            The 4x4 transform matrix from the predecessor to the successor frame.
        """

        import jaxsim.math

        return np.array(
            jaxsim.math.supported_joint_motion(joint_type, joint_position, joint_axis)
        )

    def find_parent_link_of_frame(self, name: str) -> str:
        """
        Find the parent link of a frame.

        Args:
            name: The name of the frame.

        Returns:
            The name of the parent link of the frame.
        """

        try:
            frame = self.graph.frames_dict[name]
        except KeyError as e:
            raise ValueError(f"Frame '{name}' not found in the kinematic graph") from e

        if frame.parent_name in self.graph.links_dict:
            return frame.parent_name
        if frame.parent_name in self.graph.frames_dict:
            return self.find_parent_link_of_frame(name=frame.parent_name)

        msg = f"Failed to find parent element of frame '{name}' with name '{frame.parent_name}'"
        raise RuntimeError(msg)
