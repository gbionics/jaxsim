import dataclasses
import os
import pathlib
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import rod

from jaxsim import logging
from jaxsim.math import Quaternion
from jaxsim.parsers import descriptions, kinematic_graph

from . import utils


class SDFData(NamedTuple):
    """
    Data extracted from an SDF resource useful to build a JaxSim model.
    """

    model_name: str

    fixed_base: bool
    base_link_name: str

    link_descriptions: list[descriptions.LinkDescription]
    joint_descriptions: list[descriptions.JointDescription]
    frame_descriptions: list[descriptions.LinkDescription]
    collision_shapes: list[descriptions.CollisionShape]

    sdf_model: rod.Model | None = None
    model_pose: kinematic_graph.RootPose = kinematic_graph.RootPose()


def extract_model_data(
    model_description: pathlib.Path | str | rod.Model | rod.Sdf,
    model_name: str | None = None,
    is_urdf: bool | None = None,
) -> SDFData:
    """
    Extract data from an SDF/URDF resource useful to build a JaxSim model.

    Args:
        model_description:
            A path to an SDF/URDF file, a string containing its content, or
            a pre-parsed/pre-built rod model.
        model_name: The name of the model to extract from the SDF resource.
        is_urdf:
            Whether to force parsing the resource as a URDF file. Automatically
            detected if not provided.

    Returns:
        The extracted model data.
    """

    match model_description:
        case rod.Model():
            sdf_model = model_description
        case rod.Sdf() | str() | pathlib.Path():
            sdf_element = (
                model_description
                if isinstance(model_description, rod.Sdf)
                else rod.Sdf.load(sdf=model_description, is_urdf=is_urdf)
            )
            if not sdf_element.models():
                raise RuntimeError("Failed to find any model in SDF resource")

            # Assume the SDF resource has only one model, or the desired model name is given.
            sdf_models = {m.name: m for m in sdf_element.models()}
            sdf_model = (
                sdf_element.models()[0]
                if len(sdf_models) == 1
                else sdf_models[model_name]
            )

    # Log model name.
    logging.info(msg=f"Found model '{sdf_model.name}' in SDF resource")

    # Jaxsim supports only models compatible with URDF, i.e. those having all links
    # directly attached to their parent joint without additional roto-translations.
    # Furthermore, the following switch also post-processes frames such that their
    # pose is expressed wrt the parent link they are rigidly attached to.
    sdf_model.switch_frame_convention(frame_convention=rod.FrameConvention.Urdf)

    # Log type of base link.
    logging.debug(
        msg=f"Model '{sdf_model.name}' is {'fixed-base' if sdf_model.is_fixed_base() else 'floating-base'}"
    )

    # Log detected base link.
    logging.debug(msg=f"Considering '{sdf_model.get_canonical_link()}' as base link")

    # ===========
    # Parse links
    # ===========

    # Parse the links (unconnected).
    links = [
        descriptions.LinkDescription(
            name=l.name,
            mass=float(l.inertial.mass),
            inertia=utils.from_sdf_inertial(inertial=l.inertial),
            pose=l.pose.transform() if l.pose is not None else np.eye(4),
        )
        for l in sdf_model.links()
        if l.inertial.mass > 0
    ]

    # Create a dictionary to find easily links.
    links_dict: dict[str, descriptions.LinkDescription] = {l.name: l for l in links}

    # ============
    # Parse frames
    # ============

    # Parse the frames (unconnected).
    frames = [
        descriptions.LinkDescription(
            name=f.name,
            mass=jnp.array(0.0, dtype=float),
            inertia=jnp.zeros(shape=(3, 3)),
            parent_name=f.attached_to,
            pose=f.pose.transform() if f.pose is not None else jnp.eye(4),
        )
        for f in sdf_model.frames()
        if f.attached_to in links_dict
    ]

    # =========================
    # Process fixed-base models
    # =========================

    # In this case, we need to get the pose of the joint that connects the base link
    # to the world and combine their pose.
    if sdf_model.is_fixed_base():
        # Create a massless word link
        world_link = descriptions.LinkDescription(
            name="world", mass=0, inertia=np.zeros(shape=(6, 6))
        )

        # Gather joints connecting fixed-base models to the world.
        # TODO: the pose of this joint could be expressed wrt any arbitrary frame,
        #       here we assume is expressed wrt the model. This also means that the
        #       default model pose matches the pose of the fake "world" link.
        joints_with_world_parent = [
            descriptions.JointDescription(
                name=j.name,
                parent=world_link,
                child=links_dict[j.child],
                jtype=utils.joint_to_joint_type(joint=j),
                axis=(
                    np.array(j.axis.xyz.xyz)
                    if j.axis is not None
                    and j.axis.xyz is not None
                    and j.axis.xyz.xyz is not None
                    else None
                ),
                pose=j.pose.transform() if j.pose is not None else np.eye(4),
            )
            for j in sdf_model.joints()
            if j.type == "fixed"
            and j.parent == "world"
            and j.child in links_dict
            and j.pose.relative_to in {"__model__", "world", None}
        ]

        logging.debug(
            f"Found joints connecting to world: {[j.name for j in joints_with_world_parent]}"
        )

        if len(joints_with_world_parent) != 1:
            msg = "Found more/less than one joint connecting a fixed-base model to the world"
            raise ValueError(msg + f": {[j.name for j in joints_with_world_parent]}")

        base_link_name = joints_with_world_parent[0].child.name

        if base_link_name != sdf_model.get_canonical_link():
            msg = "Combining the pose of base link '{}' with the pose of joint '{}'"
            logging.debug(msg.format(base_link_name, joints_with_world_parent[0].name))

            # Combine the pose of the base link (child of the found fixed joint)
            # with the pose of the fixed joint connecting with the world.
            # Note: when base_link_name == sdf_model.get_canonical_link(),
            # switch_frame_convention(Urdf) has already resolved the base link pose
            # relative to "world".
            links_dict[base_link_name].mutable(validate=False).pose = (
                joints_with_world_parent[0].pose @ links_dict[base_link_name].pose
            )

        W_H_B = links_dict[base_link_name].pose

    else:
        base_link_name = sdf_model.get_canonical_link()
        W_H_M = sdf_model.pose.transform() if sdf_model.pose is not None else np.eye(4)
        W_H_B = W_H_M @ links_dict[base_link_name].pose

    model_pose = kinematic_graph.RootPose(
        root_position=W_H_B[0:3, 3],
        root_quaternion=Quaternion.from_dcm(dcm=W_H_B[0:3, 0:3]),
    )

    # ============
    # Parse joints
    # ============

    # Check that all joint poses are expressed w.r.t. their parent link.
    for j in sdf_model.joints():
        if j.pose is None:
            continue

        if j.parent == "world":
            if j.pose.relative_to in {"__model__", "world", None}:
                continue

            raise ValueError("Pose of fixed joint connecting to 'world' link not valid")

        if j.pose.relative_to != j.parent:
            msg = "Pose of joint '{}' is not expressed wrt its parent link '{}'"
            raise ValueError(msg.format(j.name, j.parent))

    # Parse the joints.
    joints = [
        descriptions.JointDescription(
            name=j.name,
            parent=links_dict[j.parent],
            child=links_dict[j.child],
            jtype=utils.joint_to_joint_type(joint=j),
            axis=(
                np.array(j.axis.xyz.xyz, dtype=float)
                if j.axis is not None
                and j.axis.xyz is not None
                and j.axis.xyz.xyz is not None
                else None
            ),
            pose=j.pose.transform() if j.pose is not None else np.eye(4),
            initial_position=0.0,
            position_limit=(
                float(
                    j.axis.limit.lower
                    if j.axis is not None
                    and j.axis.limit is not None
                    and j.axis.limit.lower is not None
                    else jnp.finfo(float).min
                ),
                float(
                    j.axis.limit.upper
                    if j.axis is not None
                    and j.axis.limit is not None
                    and j.axis.limit.upper is not None
                    else jnp.finfo(float).max
                ),
            ),
            friction_static=float(
                j.axis.dynamics.friction
                if j.axis is not None
                and j.axis.dynamics is not None
                and j.axis.dynamics.friction is not None
                else 0.0
            ),
            friction_viscous=float(
                j.axis.dynamics.damping
                if j.axis is not None
                and j.axis.dynamics is not None
                and j.axis.dynamics.damping is not None
                else 0.0
            ),
            position_limit_damper=float(
                j.axis.limit.dissipation
                if j.axis is not None
                and j.axis.limit is not None
                and j.axis.limit.dissipation is not None
                else os.environ.get("JAXSIM_JOINT_POSITION_LIMIT_DAMPER", 0.0)
            ),
            position_limit_spring=float(
                j.axis.limit.stiffness
                if j.axis is not None
                and j.axis.limit is not None
                and j.axis.limit.stiffness is not None
                else os.environ.get("JAXSIM_JOINT_POSITION_LIMIT_SPRING", 0.0)
            ),
        )
        for j in sdf_model.joints()
        if j.type in {"revolute", "continuous", "prismatic", "fixed"}
        and j.parent != "world"
        and j.child in links_dict
    ]

    # Create a dictionary to find the parent joint of the links.
    joint_dict = {j.child.name: j.name for j in joints}

    # Check that all the link poses are expressed wrt their parent joint.
    for l in sdf_model.links():
        if l.name not in links_dict:
            continue

        if l.pose is None:
            continue

        if l.name == sdf_model.get_canonical_link():
            continue

        if l.name not in joint_dict:
            raise ValueError(f"Failed to find parent joint of link '{l.name}'")

        if l.pose.relative_to != joint_dict[l.name]:
            msg = "Pose of link '{}' is not expressed wrt its parent joint '{}'"
            raise ValueError(msg.format(l.name, joint_dict[l.name]))

    # ================
    # Parse collisions
    # ================

    # Initialize the collision shapes
    collisions: list[descriptions.CollisionShape] = []

    # Parse the collisions
    for link in sdf_model.links():
        for collision in link.collisions():
            if collision.geometry.box is not None:
                box_collision = utils.create_box_collision(
                    collision=collision,
                    link_description=links_dict[link.name],
                )

                collisions.append(box_collision)
                continue

            if collision.geometry.sphere is not None:
                sphere_collision = utils.create_sphere_collision(
                    collision=collision,
                    link_description=links_dict[link.name],
                )

                collisions.append(sphere_collision)
                continue

            if collision.geometry.mesh is not None:
                if int(os.environ.get("JAXSIM_COLLISION_MESH_ENABLED", "0")):
                    logging.warning("Mesh collision support is still experimental.")
                    mesh_collision = utils.create_mesh_collision(
                        collision=collision,
                        link_description=links_dict[link.name],
                        method=utils.meshes.extract_points_vertices,
                    )

                    collisions.append(mesh_collision)

                else:
                    logging.warning(
                        f"Skipping collision shape 'mesh' in link '{link.name}' because mesh collisions are disabled."
                    )

                continue

            # Check any remaining non-None geometry types.
            for attr_name in collision.geometry.__dict__:
                if getattr(collision.geometry, attr_name) is not None:
                    logging.warning(
                        f"Skipping collision shape '{attr_name}' in link '{link.name}' as not supported."
                    )

    return SDFData(
        model_name=sdf_model.name,
        link_descriptions=links,
        joint_descriptions=joints,
        frame_descriptions=frames,
        collision_shapes=collisions,
        fixed_base=sdf_model.is_fixed_base(),
        base_link_name=base_link_name,
        model_pose=model_pose,
        sdf_model=sdf_model,
    )


def build_model_description(
    model_description: pathlib.Path | str | rod.Model,
    is_urdf: bool | None = None,
    base_link_name: str | None = None,
) -> descriptions.ModelDescription:
    """
    Build a model description from an SDF/URDF resource.

    Args:
        model_description: A path to an SDF/URDF file, a string containing its content,
          or a pre-parsed/pre-built rod model.
        is_urdf: Whether the force parsing the resource as a URDF file. Automatically
            detected if not provided.
        base_link_name:
            The optional name of the base link (root of the kinematic tree). If None,
            the canonical link detected from the model description is used.

    Returns:
        The parsed model description.
    """

    # Parse data from the SDF assuming it contains a single model.
    sdf_data = extract_model_data(
        model_description=model_description, model_name=None, is_urdf=is_urdf
    )

    # Build the intermediate representation used for building a JaxSim model.
    # This process, beyond other operations, removes the fixed joints.
    # Note: if the model is fixed-base, the fixed joint between world and the first
    #       link is removed and the pose of the first link is updated.
    #
    # The whole process is:
    # URDF/SDF ⟶ rod.Model ⟶ ModelDescription ⟶ JaxSimModel.
    graph = descriptions.ModelDescription.build_model_from(
        name=sdf_data.model_name,
        links=sdf_data.link_descriptions,
        joints=sdf_data.joint_descriptions,
        frames=sdf_data.frame_descriptions,
        collisions=sdf_data.collision_shapes,
        fixed_base=sdf_data.fixed_base,
        base_link_name=(
            base_link_name if base_link_name is not None else sdf_data.base_link_name
        ),
        model_pose=sdf_data.model_pose,
        considered_joints=[
            j.name
            for j in sdf_data.joint_descriptions
            if j.jtype is not descriptions.JointType.Fixed
        ],
    )

    # Store the parsed SDF tree as extra info
    graph = dataclasses.replace(graph, _extra_info={"sdf_model": sdf_data.sdf_model})

    return graph
