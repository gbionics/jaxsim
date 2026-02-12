from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import pathlib
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import rod
from jax_dataclasses import Static
from rod.urdf.exporter import UrdfExporter

import jaxsim.api as js
import jaxsim.terrain
import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.api.kin_dyn_parameters import (
    HwLinkMetadata,
    KinDynParameters,
    LinkParameters,
    LinkParametrizableShape,
    ScalingFactors,
)
from jaxsim.math import Adjoint, Cross
from jaxsim.parsers.descriptions import ModelDescription
from jaxsim.parsers.descriptions.joint import JointDescription
from jaxsim.parsers.descriptions.link import LinkDescription
from jaxsim.utils import JaxsimDataclass, Mutability, wrappers

from .common import VelRepr


class IntegratorType(enum.IntEnum):
    """The integrators available for the simulation."""

    SemiImplicitEuler = enum.auto()
    RungeKutta4 = enum.auto()
    RungeKutta4Fast = enum.auto()


@jax_dataclasses.pytree_dataclass(eq=False, unsafe_hash=False)
class JaxSimModel(JaxsimDataclass):
    """
    The JaxSim model defining the kinematics and dynamics of a robot.
    """

    model_name: Static[str]

    time_step: float = dataclasses.field(
        default=0.001,
    )

    terrain: Static[jaxsim.terrain.Terrain] = dataclasses.field(
        default_factory=jaxsim.terrain.FlatTerrain.build, repr=False
    )

    gravity: Static[float] = -jaxsim.math.STANDARD_GRAVITY

    contact_model: Static[jaxsim.rbda.contacts.ContactModel | None] = dataclasses.field(
        default=None, repr=False
    )

    contact_params: jaxsim.rbda.contacts.ContactsParams = dataclasses.field(
        default=None, repr=False
    )

    actuation_params: Static[jaxsim.rbda.actuation.ActuationParams] = dataclasses.field(
        default=None, repr=False
    )

    kin_dyn_parameters: js.kin_dyn_parameters.KinDynParameters | None = (
        dataclasses.field(default=None, repr=False)
    )

    integrator: Static[IntegratorType] = dataclasses.field(
        default=IntegratorType.SemiImplicitEuler, repr=False
    )

    built_from: Static[str | pathlib.Path | rod.Model | None] = dataclasses.field(
        default=None, repr=False
    )

    _description: Static[wrappers.HashlessObject[ModelDescription | None]] = (
        dataclasses.field(default=None, repr=False)
    )

    @property
    def description(self) -> ModelDescription:
        """
        Return the model description.
        """
        return self._description.get()

    def __eq__(self, other: JaxSimModel) -> bool:
        if not isinstance(other, JaxSimModel):
            return False

        if self.model_name != other.model_name:
            return False

        if self.time_step != other.time_step:
            return False

        if self.kin_dyn_parameters != other.kin_dyn_parameters:
            return False

        return True

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.model_name),
                hash(self.time_step),
                hash(self.kin_dyn_parameters),
                hash(self.contact_model),
            )
        )

    # ========================
    # Initialization and state
    # ========================

    @classmethod
    def build_from_model_description(
        cls,
        model_description: str | pathlib.Path | rod.Model,
        *,
        model_name: str | None = None,
        time_step: jtp.FloatLike | None = None,
        terrain: jaxsim.terrain.Terrain | None = None,
        contact_model: jaxsim.rbda.contacts.ContactModel | None = None,
        contact_params: jaxsim.rbda.contacts.ContactsParams | None = None,
        actuation_params: jaxsim.rbda.actuation.ActuationParams | None = None,
        integrator: IntegratorType | None = None,
        is_urdf: bool | None = None,
        considered_joints: Sequence[str] | None = None,
        gravity: jtp.FloatLike = jaxsim.math.STANDARD_GRAVITY,
        constraints: jaxsim.rbda.kinematic_constraints.ConstraintMap | None = None,
        parametrized_links: tuple[str, ...] | None = None,
    ) -> JaxSimModel:
        """
        Build a Model object from a model description.

        Args:
            model_description:
                A path to an SDF/URDF file, a string containing
                its content, or a pre-parsed/pre-built rod model.
            model_name:
                The name of the model. If not specified, it is read from the description.
            time_step:
                The default time step to consider for the simulation. It can be
                manually overridden in the function that steps the simulation.
            terrain: The terrain to consider (the default is a flat infinite plane).
            contact_model:
                The contact model to consider.
                If not specified, a soft contacts model is used.
            contact_params: The parameters of the contact model.
            actuation_params: The parameters of the actuation model.
            integrator: The integrator to use for the simulation.
            is_urdf:
                The optional flag to force the model description to be parsed as a URDF.
                This is usually automatically inferred.
            considered_joints:
                The list of joints to consider. If None, all joints are considered.
            gravity: The gravity constant. Normally passed as a positive value.
            constraints:
                An object of type ConstraintMap containing the kinematic constraints to consider. If None, no constraints are considered.
                Note that constraints can be used only with RelaxedRigidContacts.
            parametrized_links:
                The optional list of links to be parametrized. If None, all links are parametrized.

        Returns:
            The built Model object.
        """

        import jaxsim.parsers.rod

        # Parse the input resource (either a path to file or a string with the URDF/SDF)
        # and build the -intermediate- model description.
        intermediate_description = jaxsim.parsers.rod.build_model_description(
            model_description=model_description, is_urdf=is_urdf
        )

        # Lump links together if not all joints are considered.
        # Note: this procedure assigns a zero position to all joints not considered.
        if considered_joints is not None:
            intermediate_description = intermediate_description.reduce(
                considered_joints=considered_joints
            )

        # Build the model.
        model = cls.build(
            model_description=intermediate_description,
            model_name=model_name,
            time_step=time_step,
            terrain=terrain,
            contact_model=contact_model,
            actuation_params=actuation_params,
            contact_params=contact_params,
            integrator=integrator,
            gravity=-gravity,
            constraints=constraints,
            parametrized_links=parametrized_links,
        )

        # Store the origin of the model, in case downstream logic needs it.
        with model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            model.built_from = model_description

        # Compute the hw parametrization metadata of the model
        # TODO: move the building of the metadata to KinDynParameters.build()
        #       and use the model_description instead of model.built_from.
        with model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            model.kin_dyn_parameters.hw_link_metadata = model.compute_hw_link_metadata(
                parametrized_links=parametrized_links
            )

        return model

    @classmethod
    def build(
        cls,
        model_description: ModelDescription,
        *,
        model_name: str | None = None,
        time_step: jtp.FloatLike | None = None,
        terrain: jaxsim.terrain.Terrain | None = None,
        contact_model: jaxsim.rbda.contacts.ContactModel | None = None,
        contact_params: jaxsim.rbda.contacts.ContactsParams | None = None,
        actuation_params: jaxsim.rbda.actuation.ActuationParams | None = None,
        integrator: IntegratorType | None = None,
        gravity: jtp.FloatLike = jaxsim.math.STANDARD_GRAVITY,
        constraints: jaxsim.rbda.kinematic_constraints.ConstraintMap | None = None,
        parametrized_links: tuple[str, ...] | None = None,
    ) -> JaxSimModel:
        """
        Build a Model object from an intermediate model description.

        Args:
            model_description:
                The intermediate model description defining the kinematics and dynamics
                of the model.
            model_name:
                The name of the model. If not specified, it is read from the description.
            time_step:
                The default time step to consider for the simulation. It can be
                manually overridden in the function that steps the simulation.
            terrain: The terrain to consider (the default is a flat infinite plane).
                The optional name of the model overriding the physics model name.
            contact_model:
                The contact model to consider.
                If not specified, a soft contact model is used.
            contact_params: The parameters of the contact model.
            actuation_params: The parameters of the actuation model.
            integrator: The integrator to use for the simulation.
            gravity: The gravity constant.
            constraints:
                An object of type ConstraintMap containing the kinematic constraints to consider. If None, no constraints are considered.
            parametrized_links:
                The optional list of links to be parametrized. If None, all links are parametrized.

        Returns:
            The built Model object.
        """

        # Set the model name (if not provided, use the one from the model description).
        model_name = model_name if model_name is not None else model_description.name

        # Consider the default terrain (a flat infinite plane) if not specified.
        terrain = (
            terrain
            if terrain is not None
            else JaxSimModel.__dataclass_fields__["terrain"].default_factory()
        )

        # Consider the default time step if not specified.
        time_step = (
            time_step
            if time_step is not None
            else JaxSimModel.__dataclass_fields__["time_step"].default
        )

        # Create the default contact model.
        # It will be populated with an initial estimation of good parameters.
        # While these might not be the best, they are a good starting point.
        contact_model = (
            contact_model
            if contact_model is not None
            else jaxsim.rbda.contacts.SoftContacts.build()
        )

        if contact_params is None:
            contact_params = contact_model._parameters_class()

        if actuation_params is None:
            actuation_params = jaxsim.rbda.actuation.ActuationParams()

        # Consider the default integrator if not specified.
        integrator = (
            integrator
            if integrator is not None
            else JaxSimModel.__dataclass_fields__["integrator"].default
        )

        # Build the model.
        model = cls(
            model_name=model_name,
            kin_dyn_parameters=js.kin_dyn_parameters.KinDynParameters.build(
                model_description=model_description, constraints=constraints
            ),
            time_step=time_step,
            terrain=terrain,
            contact_model=contact_model,
            contact_params=contact_params,
            actuation_params=actuation_params,
            integrator=integrator,
            gravity=gravity,
            # The following is wrapped as hashless since it's a static argument, and we
            # don't want to trigger recompilation if it changes. All relevant parameters
            # needed to compute kinematics and dynamics quantities are stored in the
            # kin_dyn_parameters attribute.
            _description=wrappers.HashlessObject(obj=model_description),
        )

        return model

    def compute_hw_link_metadata(
        self, parametrized_links: tuple[str, ...] | None = None
    ) -> HwLinkMetadata:
        """
        Compute the parametric metadata of the links in the model.

        Args:
            parametrized_links:
                An optional tuple of link names to be parametrized. If None,
                all links will be parametrized.

        Returns:
            An instance of HwLinkMetadata containing the metadata of all links.
        """
        model_description = self.description

        # Get ordered links and joints from the model description
        ordered_links: list[LinkDescription] = sorted(
            list(model_description.links_dict.values()),
            key=lambda l: l.index,
        )
        ordered_joints: list[JointDescription] = sorted(
            list(model_description.joints_dict.values()),
            key=lambda j: j.index,
        )

        # Ensure the model was built from a valid source
        rod_model = None
        match self.built_from:
            case str() | pathlib.Path():
                rod_model = rod.Sdf.load(sdf=self.built_from).models()[0]
                assert rod_model.name == self.name()
            case rod.Model():
                rod_model = self.built_from
            case _:
                logging.debug(
                    f"Invalid type for model.built_from ({type(self.built_from)})."
                    "Skipping for hardware parametrization."
                )
                return HwLinkMetadata.empty()

        # Use URDF frame convention for consistent pose representation
        rod_model.switch_frame_convention(
            frame_convention=rod.FrameConvention.Urdf, explicit_frames=True
        )

        rod_links_dict = {link.name: link for link in rod_model.links()}

        # Initialize lists to collect metadata for all links
        shapes = []
        geoms = []
        densities = []
        L_H_Gs = []
        L_H_vises = []
        L_H_pre_masks = []
        L_H_pre = []

        # Process each link, only parametrizing those in parametrized_links if provided
        for link_description in ordered_links:
            link_name = link_description.name

            if parametrized_links is not None and link_name not in parametrized_links:
                # Mark as unsupported for non-parametrized links
                shapes.append(LinkParametrizableShape.Unsupported)
                geoms.append([0, 0, 0])
                densities.append(0.0)
                L_H_Gs.append(jnp.eye(4))
                L_H_vises.append(jnp.eye(4))
                L_H_pre_masks.append([0] * self.number_of_joints())
                L_H_pre.append([jnp.eye(4)] * self.number_of_joints())
                continue

            rod_link = rod_links_dict.get(link_name)
            link_index = int(js.link.name_to_idx(model=self, link_name=link_name))

            # Get child joints for the link
            child_joints_indices = [
                js.joint.name_to_idx(model=self, joint_name=j.name)
                for j in ordered_joints
                if j.parent.name == link_name
            ]

            # Skip unsupported links
            if not jnp.allclose(
                self.kin_dyn_parameters.joint_model.suc_H_i[link_index],
                jnp.eye(4),
                **(dict(atol=1e-6) if not jax.config.jax_enable_x64 else dict()),
            ):
                logging.debug(
                    f"Skipping link '{link_name}' for hardware parametrization due to unsupported suc_H_link."
                )
                rod_link = None

            # Compute density and dimensions
            mass = float(self.kin_dyn_parameters.link_parameters.mass[link_index])

            # Find the first supported visual
            supported_visual = (
                next(
                    (
                        v
                        for v in rod_link.visuals()
                        if isinstance(
                            v.geometry.geometry(), (rod.Box, rod.Sphere, rod.Cylinder)
                        )
                    ),
                    None,
                )
                if rod_link
                else None
            )

            geometry = (
                supported_visual.geometry.geometry() if supported_visual else None
            )
            if isinstance(geometry, rod.Box):
                lx, ly, lz = geometry.size
                density = mass / (lx * ly * lz)
                geom = [lx, ly, lz]
                shape = LinkParametrizableShape.Box
            elif isinstance(geometry, rod.Sphere):
                r = geometry.radius
                density = mass / (4 / 3 * jnp.pi * r**3)
                geom = [r, 0, 0]
                shape = LinkParametrizableShape.Sphere
            elif isinstance(geometry, rod.Cylinder):
                r, l = geometry.radius, geometry.length
                density = mass / (jnp.pi * r**2 * l)
                geom = [r, l, 0]
                shape = LinkParametrizableShape.Cylinder
            else:
                logging.debug(
                    f"Skipping link '{link_name}' for hardware parametrization due to unsupported geometry."
                )
                density = 0.0
                geom = [0, 0, 0]
                shape = LinkParametrizableShape.Unsupported

            inertial_pose = (
                rod_link.inertial.pose.transform() if rod_link else jnp.eye(4)
            )
            visual_pose = (
                supported_visual.pose.transform() if supported_visual else jnp.eye(4)
            )
            l_h_pre_mask = [
                int(joint_index in child_joints_indices)
                for joint_index in range(self.number_of_joints())
            ]
            l_h_pre = [
                (
                    self.kin_dyn_parameters.joint_model.λ_H_pre[joint_index + 1]
                    if joint_index in child_joints_indices
                    else jnp.eye(4)
                )
                for joint_index in range(self.number_of_joints())
            ]

            shapes.append(shape)
            geoms.append(geom)
            densities.append(density)
            L_H_Gs.append(inertial_pose)
            L_H_vises.append(visual_pose)
            L_H_pre_masks.append(l_h_pre_mask)
            L_H_pre.append(l_h_pre)

        if np.all(np.array(shapes) == LinkParametrizableShape.Unsupported):
            logging.debug(
                "All links were skipped for hardware parametrization. Returning empty metadata."
            )
            return HwLinkMetadata.empty()

        # Stack collected data into JAX arrays
        return HwLinkMetadata(
            link_shape=jnp.array(shapes, dtype=int),
            geometry=jnp.array(geoms, dtype=float),
            density=jnp.array(densities, dtype=float),
            L_H_G=jnp.array(L_H_Gs, dtype=float),
            L_H_vis=jnp.array(L_H_vises, dtype=float),
            L_H_pre_mask=jnp.array(L_H_pre_masks, dtype=bool),
            L_H_pre=jnp.array(L_H_pre, dtype=float),
        )

    def export_updated_model(self) -> str:
        """
        Export the JaxSim model to URDF with the current hardware parameters.

        Returns:
            The URDF string of the updated model.

        Note:
            This method is not meant to be used in JIT-compiled functions.
        """

        if isinstance(jnp.zeros(0), jax.core.Tracer):
            raise RuntimeError("This method cannot be used in JIT-compiled functions")

        # Ensure `built_from` is a ROD model and create `rod_model_output`
        if isinstance(self.built_from, rod.Model):
            rod_model_output = copy.deepcopy(self.built_from)
        elif isinstance(self.built_from, (str, pathlib.Path)):
            rod_model_output = rod.Sdf.load(sdf=self.built_from).models()[0]
        else:
            raise ValueError(
                "The JaxSim model must be built from a valid ROD model source"
            )

        # Switch to URDF frame convention for easier mapping
        rod_model_output.switch_frame_convention(
            frame_convention=rod.FrameConvention.Urdf,
            explicit_frames=True,
            attach_frames_to_links=True,
        )

        # Get links and joints from the ROD model
        links_dict = {link.name: link for link in rod_model_output.links()}
        joints_dict = {joint.name: joint for joint in rod_model_output.joints()}

        # Iterate over the hardware metadata to update the ROD model
        hw_metadata = self.kin_dyn_parameters.hw_link_metadata
        for link_index, link_name in enumerate(self.link_names()):
            if link_name not in links_dict:
                continue

            # Skip links with unsupported shapes
            shape = hw_metadata.link_shape[link_index]
            if shape == LinkParametrizableShape.Unsupported:
                logging.debug(f"Skipping link '{link_name}' with unsupported shape")
                continue

            # Update mass and inertia
            mass = float(self.kin_dyn_parameters.link_parameters.mass[link_index])
            center_of_mass = np.array(
                self.kin_dyn_parameters.link_parameters.center_of_mass[link_index]
            )
            inertia_tensor = LinkParameters.unflatten_inertia_tensor(
                self.kin_dyn_parameters.link_parameters.inertia_elements[link_index]
            )

            links_dict[link_name].inertial.mass = mass
            L_H_COM = np.eye(4)
            L_H_COM[0:3, 3] = center_of_mass
            links_dict[link_name].inertial.pose = rod.Pose.from_transform(
                transform=L_H_COM,
                relative_to=links_dict[link_name].inertial.pose.relative_to,
            )
            links_dict[link_name].inertial.inertia = rod.Inertia.from_inertia_tensor(
                inertia_tensor=inertia_tensor, validate=True
            )

            # Update visual shape
            dims = hw_metadata.geometry[link_index]
            if shape == LinkParametrizableShape.Box:
                links_dict[link_name].visual.geometry.box.size = dims.tolist()
                links_dict[link_name].collision.geometry.box.size = dims.tolist()
            elif shape == LinkParametrizableShape.Sphere:
                links_dict[link_name].visual.geometry.sphere.radius = float(dims[0])
                links_dict[link_name].collision.geometry.sphere.radius = float(dims[0])
            elif shape == LinkParametrizableShape.Cylinder:
                links_dict[link_name].visual.geometry.cylinder.radius = float(dims[0])
                links_dict[link_name].visual.geometry.cylinder.length = float(dims[1])
                links_dict[link_name].collision.geometry.cylinder.radius = float(
                    dims[0]
                )
                links_dict[link_name].collision.geometry.cylinder.length = float(
                    dims[1]
                )
            else:
                logging.debug(f"Skipping unsupported shape for link '{link_name}'")
                continue

            # Update visual pose
            links_dict[link_name].visual.pose = rod.Pose.from_transform(
                transform=np.array(hw_metadata.L_H_vis[link_index]),
                relative_to=link_name,
            )

            links_dict[link_name].collision.pose = rod.Pose.from_transform(
                transform=np.array(hw_metadata.L_H_vis[link_index]),
                relative_to=link_name,
            )

            # Update joint poses
            for joint_index in range(self.number_of_joints()):
                if hw_metadata.L_H_pre_mask[link_index, joint_index]:
                    joint_name = js.joint.idx_to_name(
                        model=self, joint_index=joint_index
                    )
                    if joint_name in joints_dict:
                        joints_dict[joint_name].pose = rod.Pose.from_transform(
                            transform=np.array(
                                hw_metadata.L_H_pre[link_index, joint_index]
                            ),
                            relative_to=link_name,
                        )

        # Restore continuous joint types for joints with infinite limits
        # to ensure valid URDF export (continuous joints should not have limits).
        # Continuous joints are internally represented as revolute with infinite
        # limits, but must be exported as type="continuous" for valid URDF.
        for joint in joints_dict.values():
            # Skip if not a revolute joint with axis and limits
            if not (
                joint.type == "revolute"
                and joint.axis is not None
                and joint.axis.limit is not None
            ):
                continue

            lower, upper = joint.axis.limit.lower, joint.axis.limit.upper

            # Check if both limits are infinite (indicating original continuous joint)
            if not (
                lower is not None
                and upper is not None
                and np.isinf(lower)
                and lower < 0
                and np.isinf(upper)
                and upper > 0
            ):
                continue

            # Restore as continuous joint
            joint.type = "continuous"

            # Create a new Limit object with only effort and velocity
            # (no position limits for continuous joints)
            joint.axis.limit = rod.Limit(
                effort=joint.axis.limit.effort,
                velocity=joint.axis.limit.velocity,
                lower=None,
                upper=None,
            )

        # Export the URDF string
        urdf_string = UrdfExporter(pretty=True).to_urdf_string(sdf=rod_model_output)

        return urdf_string

    # ==========
    # Properties
    # ==========

    def name(self) -> str:
        """
        Return the name of the model.

        Returns:
            The name of the model.
        """

        return self.model_name

    def number_of_links(self) -> int:
        """
        Return the number of links in the model.

        Returns:
            The number of links in the model.

        Note:
            The base link is included in the count and its index is always 0.
        """

        return self.kin_dyn_parameters.number_of_links()

    def number_of_joints(self) -> int:
        """
        Return the number of joints in the model.

        Returns:
            The number of joints in the model.
        """

        return self.kin_dyn_parameters.number_of_joints()

    def number_of_frames(self) -> int:
        """
        Return the number of frames in the model.

        Returns:
            The number of frames in the model.

        """

        return self.kin_dyn_parameters.number_of_frames()

    # =================
    # Base link methods
    # =================

    def floating_base(self) -> bool:
        """
        Return whether the model has a floating base.

        Returns:
            True if the model is floating-base, False otherwise.
        """

        return self.kin_dyn_parameters.joint_model.joint_dofs[0] == 6

    def base_link(self) -> str:
        """
        Return the name of the base link.

        Returns:
            The name of the base link.

        Note:
            By default, the base link is the root of the kinematic tree.
        """

        return self.link_names()[0]

    # =====================
    # Joint-related methods
    # =====================

    def dofs(self) -> int:
        """
        Return the number of degrees of freedom of the model.

        Returns:
            The number of degrees of freedom of the model.

        Note:
            We do not yet support multi-DoF joints, therefore this is always equal to
            the number of joints. In the future, this could be different.
        """

        return sum(self.kin_dyn_parameters.joint_model.joint_dofs[1:])

    def joint_names(self) -> tuple[str, ...]:
        """
        Return the names of the joints in the model.

        Returns:
            The names of the joints in the model.
        """

        return self.kin_dyn_parameters.joint_model.joint_names[1:]

    # ====================
    # Link-related methods
    # ====================

    def link_names(self) -> tuple[str, ...]:
        """
        Return the names of the links in the model.

        Returns:
            The names of the links in the model.
        """

        return self.kin_dyn_parameters.link_names

    # =====================
    # Frame-related methods
    # =====================

    def frame_names(self) -> tuple[str, ...]:
        """
        Return the names of the frames in the model.

        Returns:
            The names of the frames in the model.
        """

        return self.kin_dyn_parameters.frame_parameters.name


# =====================
# Model post-processing
# =====================


def reduce(
    model: JaxSimModel,
    considered_joints: tuple[str, ...],
    locked_joint_positions: dict[str, jtp.FloatLike] | None = None,
) -> JaxSimModel:
    """
    Reduce the model by lumping together the links connected by removed joints.

    Args:
        model: The model to reduce.
        considered_joints: The sequence of joints to consider.
        locked_joint_positions:
            A dictionary containing the positions of the joints to be considered
            in the reduction process. The removed joints in the reduced model
            will have their position locked to their value of this dictionary.
            If a joint is not part of the dictionary, its position is set to zero.
    """

    locked_joint_positions = (
        locked_joint_positions if locked_joint_positions is not None else {}
    )

    # If locked joints are passed, make sure that they are valid.
    if not set(locked_joint_positions).issubset(model.joint_names()):
        new_joints = set(model.joint_names()) - set(locked_joint_positions)
        raise ValueError(f"Passed joints not existing in the model: {new_joints}")

    # Operate on a deep copy of the model description in order to prevent problems
    # when mutable attributes are updated.
    intermediate_description = copy.deepcopy(model.description)

    # Update the initial position of the joints.
    # This is necessary to compute the correct pose of the link pairs connected
    # to removed joints.
    for joint_name in set(model.joint_names()) - set(considered_joints):
        j = intermediate_description.joints_dict[joint_name]
        with j.mutable_context():
            j.initial_position = locked_joint_positions.get(joint_name, 0.0)

    # Reduce the model description.
    # If `considered_joints` contains joints not existing in the model,
    # the method will raise an exception.
    reduced_intermediate_description = intermediate_description.reduce(
        considered_joints=list(considered_joints)
    )

    # Build the reduced model.
    reduced_model = JaxSimModel.build(
        model_description=reduced_intermediate_description,
        model_name=model.name(),
        time_step=model.time_step,
        terrain=model.terrain,
        contact_model=model.contact_model,
        contact_params=model.contact_params,
        actuation_params=model.actuation_params,
        gravity=model.gravity,
        integrator=model.integrator,
        constraints=model.kin_dyn_parameters.constraints,
    )

    with reduced_model.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
        # Store the origin of the model, in case downstream logic needs it.
        reduced_model.built_from = model.built_from

        # Compute the hw parametrization metadata of the reduced model
        # TODO: move the building of the metadata to KinDynParameters.build()
        #       and use the model_description instead of model.built_from.
        reduced_model.kin_dyn_parameters.hw_link_metadata = (
            reduced_model.compute_hw_link_metadata()
        )

    return reduced_model


# ===================
# Inertial properties
# ===================


@jax.jit
@js.common.named_scope
def total_mass(model: JaxSimModel) -> jtp.Float:
    """
    Compute the total mass of the model.

    Args:
        model: The model to consider.

    Returns:
        The total mass of the model.
    """

    return model.kin_dyn_parameters.link_parameters.mass.sum().astype(float)


@jax.jit
@js.common.named_scope
def link_spatial_inertia_matrices(model: JaxSimModel) -> jtp.Array:
    """
    Compute the spatial 6D inertia matrices of all links of the model.

    Args:
        model: The model to consider.

    Returns:
        A 3D array containing the stacked spatial 6D inertia matrices of the links.
    """

    return jax.vmap(js.kin_dyn_parameters.LinkParameters.spatial_inertia)(
        model.kin_dyn_parameters.link_parameters
    )


# ==============================
# Rigid Body Dynamics Algorithms
# ==============================


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
def generalized_free_floating_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating jacobians of all links.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        input_representation:
            The input velocity representation of the free-floating jacobians.
        output_vel_repr:
            The output velocity representation of the free-floating jacobians.

    Returns:
        The `(nL, 6, 6+dofs)` array containing the stacked free-floating
        jacobians of the links. The first axis is the link index.

    Note:
        The v-stacked version of the returned Jacobian array together with the
        flattened 6D forces of the links, are useful to compute the `J.T @ f`
        product of the multi-body EoM.
    """
    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    # Compute the doubly-left free-floating full jacobian.
    B_J_full_WX_B, B_H_L = jaxsim.rbda.jacobian_full_doubly_left(
        model=model,
        joint_positions=data.joint_positions,
    )

    # ======================================================================
    # Update the input velocity representation such that v_WL = J_WL_I @ I_ν
    # ======================================================================

    match input_representation:
        case VelRepr.Inertial:
            W_H_B = data._base_transform
            B_X_W = Adjoint.from_transform(transform=W_H_B, inverse=True)

            B_J_full_WX_I = B_J_full_WX_W = (  # noqa: F841
                B_J_full_WX_B
                @ jax.scipy.linalg.block_diag(B_X_W, jnp.eye(model.dofs()))
            )

        case VelRepr.Body:
            B_J_full_WX_I = B_J_full_WX_B

        case VelRepr.Mixed:
            W_R_B = jaxsim.math.Quaternion.to_dcm(data.base_orientation)
            BW_H_B = jnp.eye(4).at[0:3, 0:3].set(W_R_B)
            B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)

            B_J_full_WX_I = B_J_full_WX_BW = (  # noqa: F841
                B_J_full_WX_B
                @ jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))
            )

        case _:
            raise ValueError(data.velocity_representation)

    # ====================================================================
    # Create stacked Jacobian for each link by filtering the full Jacobian
    # ====================================================================

    κ_bool = model.kin_dyn_parameters.support_body_array_bool

    # Keep only the columns of the full Jacobian corresponding to the support
    # body array of each link.
    B_J_WL_I = jax.vmap(
        lambda κ: jnp.where(
            jnp.hstack([jnp.ones(5), κ]), B_J_full_WX_I, jnp.zeros_like(B_J_full_WX_I)
        )
    )(κ_bool)

    # =======================================================================
    # Update the output velocity representation such that O_v_WL = O_J_WL @ ν
    # =======================================================================

    match output_vel_repr:
        case VelRepr.Inertial:
            W_H_B = data._base_transform
            W_X_B = jaxsim.math.Adjoint.from_transform(W_H_B)

            O_J_WL_I = W_J_WL_I = jax.vmap(  # noqa: F841
                lambda B_J_WL_I: W_X_B @ B_J_WL_I
            )(B_J_WL_I)

        case VelRepr.Body:
            O_J_WL_I = L_J_WL_I = jax.vmap(  # noqa: F841
                lambda B_H_L, B_J_WL_I: jaxsim.math.Adjoint.from_transform(
                    B_H_L, inverse=True
                )
                @ B_J_WL_I
            )(B_H_L, B_J_WL_I)

        case VelRepr.Mixed:
            W_H_B = data._base_transform

            LW_H_L = jax.vmap(
                lambda B_H_L: (W_H_B @ B_H_L).at[0:3, 3].set(jnp.zeros(3))
            )(B_H_L)

            LW_H_B = jax.vmap(
                lambda LW_H_L, B_H_L: LW_H_L @ jaxsim.math.Transform.inverse(B_H_L)
            )(LW_H_L, B_H_L)

            O_J_WL_I = LW_J_WL_I = jax.vmap(  # noqa: F841
                lambda LW_H_B, B_J_WL_I: jaxsim.math.Adjoint.from_transform(LW_H_B)
                @ B_J_WL_I
            )(LW_H_B, B_J_WL_I)

        case _:
            raise ValueError(output_vel_repr)

    return O_J_WL_I


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
def generalized_free_floating_jacobian_derivative(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating jacobian derivatives of all links.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        input_representation:
            The input velocity representation of the free-floating jacobian derivatives.
        output_vel_repr:
            The output velocity representation of the free-floating jacobian derivatives.

    Returns:
        The `(nL, 6, 6+dofs)` array containing the stacked free-floating
        jacobian derivatives of the links. The first axis is the link index.
    """

    input_representation = (
        input_representation
        if input_representation is not None
        else data.velocity_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    # Compute the derivative of the doubly-left free-floating full jacobian.
    B_J̇_full_WX_B, B_H_L = jaxsim.rbda.jacobian_derivative_full_doubly_left(
        model=model,
        joint_positions=data.joint_positions,
        joint_velocities=data.joint_velocities,
    )

    # The derivative of the equation to change the input and output representations
    # of the Jacobian derivative needs the computation of the plain link Jacobian.
    B_J_full_WL_B, _ = jaxsim.rbda.jacobian_full_doubly_left(
        model=model,
        joint_positions=data.joint_positions,
    )

    # Compute the actual doubly-left free-floating jacobian derivative of the link
    # by zeroing the columns not in the path π_B(L) using the boolean κ(i).
    κb = model.kin_dyn_parameters.support_body_array_bool

    # Compute the base transform.
    W_H_B = data._base_transform

    # We add the 5 columns of ones to the Jacobian derivative to account for the
    # base velocity and acceleration (5 + number of links = 6 + number of joints).
    B_J̇_WL_B = (
        jnp.hstack([jnp.ones((κb.shape[0], 5)), κb])[:, jnp.newaxis] * B_J̇_full_WX_B
    )
    B_J_WL_B = (
        jnp.hstack([jnp.ones((κb.shape[0], 5)), κb])[:, jnp.newaxis] * B_J_full_WL_B
    )

    # =====================================================
    # Compute quantities to adjust the input representation
    # =====================================================

    In = jnp.eye(model.dofs())
    On = jnp.zeros(shape=(model.dofs(), model.dofs()))

    match input_representation:
        case VelRepr.Inertial:
            B_X_W = jaxsim.math.Adjoint.from_transform(transform=W_H_B, inverse=True)

            W_v_WB = data.base_velocity(input_representation)
            B_Ẋ_W = -B_X_W @ jaxsim.math.Cross.vx(W_v_WB)

            # Compute the operator to change the representation of ν, and its
            # time derivative.
            T = jax.scipy.linalg.block_diag(B_X_W, In)
            Ṫ = jax.scipy.linalg.block_diag(B_Ẋ_W, On)

        case VelRepr.Body:
            B_X_B = jaxsim.math.Adjoint.from_rotation_and_translation(
                translation=jnp.zeros(3), rotation=jnp.eye(3)
            )

            B_Ẋ_B = jnp.zeros(shape=(6, 6))

            # Compute the operator to change the representation of ν, and its
            # time derivative.
            T = jax.scipy.linalg.block_diag(B_X_B, In)
            Ṫ = jax.scipy.linalg.block_diag(B_Ẋ_B, On)

        case VelRepr.Mixed:
            BW_H_B = W_H_B.at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = jaxsim.math.Adjoint.from_transform(transform=BW_H_B, inverse=True)

            BW_v_WB = data.base_velocity(input_representation)
            BW_v_W_BW = BW_v_WB.at[3:6].set(jnp.zeros(3))

            BW_v_BW_B = BW_v_WB - BW_v_W_BW
            B_Ẋ_BW = -B_X_BW @ jaxsim.math.Cross.vx(BW_v_BW_B)

            # Compute the operator to change the representation of ν, and its
            # time derivative.
            T = jax.scipy.linalg.block_diag(B_X_BW, In)
            Ṫ = jax.scipy.linalg.block_diag(B_Ẋ_BW, On)

        case _:
            raise ValueError(input_representation)

    # ======================================================
    # Compute quantities to adjust the output representation
    # ======================================================

    match output_vel_repr:
        case VelRepr.Inertial:
            O_X_B = W_X_B = jaxsim.math.Adjoint.from_transform(transform=W_H_B)

            B_v_WB = data.base_velocity(VelRepr.Body)

            O_Ẋ_B = W_Ẋ_B = W_X_B @ jaxsim.math.Cross.vx(B_v_WB)  # noqa: F841

        case VelRepr.Body:
            O_X_B = L_X_B = jaxsim.math.Adjoint.from_transform(
                transform=B_H_L, inverse=True
            )

            B_X_L = jaxsim.math.Adjoint.inverse(adjoint=L_X_B)

            B_v_WB = data.base_velocity(VelRepr.Body)
            L_v_WL = jnp.einsum(
                "b6j,j->b6", L_X_B @ B_J_WL_B, data.generalized_velocity(VelRepr.Body)
            )

            O_Ẋ_B = L_Ẋ_B = -L_X_B @ jaxsim.math.Cross.vx(  # noqa: F841
                jnp.einsum("bij,bj->bi", B_X_L, L_v_WL) - B_v_WB
            )

        case VelRepr.Mixed:
            W_H_L = W_H_B @ B_H_L
            LW_H_L = W_H_L.at[:, 0:3, 3].set(jnp.zeros_like(W_H_L[:, 0:3, 3]))
            LW_H_B = LW_H_L @ jaxsim.math.Transform.inverse(B_H_L)

            O_X_B = LW_X_B = jaxsim.math.Adjoint.from_transform(transform=LW_H_B)

            B_X_LW = jaxsim.math.Adjoint.inverse(adjoint=LW_X_B)

            B_v_WB = data.base_velocity(output_representation=VelRepr.Body)

            BW_H_B = W_H_B.at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)
            LW_v_WL = jnp.einsum(
                "bij,bj->bi",
                LW_X_B,
                B_J_WL_B
                @ jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))
                @ data.generalized_velocity(output_representation=VelRepr.Mixed),
            )

            LW_v_W_LW = LW_v_WL.at[:, 3:6].set(jnp.zeros_like(LW_v_WL[:, 3:6]))

            LW_v_LW_L = LW_v_WL - LW_v_W_LW
            LW_v_B_LW = LW_v_WL - jnp.einsum("bij,j->bi", LW_X_B, B_v_WB) - LW_v_LW_L

            O_Ẋ_B = LW_Ẋ_B = -LW_X_B @ jaxsim.math.Cross.vx(  # noqa: F841
                jnp.einsum("bij,bj->bi", B_X_LW, LW_v_B_LW)
            )

        case _:
            raise ValueError(output_vel_repr)

    # =============================================================
    # Express the Jacobian derivative in the target representations
    # =============================================================

    # Sum all the components that form the Jacobian derivative in the target
    # input/output velocity representations.
    O_J̇_WL_I = jnp.zeros_like(B_J̇_WL_B)
    O_J̇_WL_I += O_Ẋ_B @ B_J_WL_B @ T
    O_J̇_WL_I += O_X_B @ B_J̇_WL_B @ T
    O_J̇_WL_I += O_X_B @ B_J_WL_B @ Ṫ

    return O_J̇_WL_I


@functools.partial(jax.jit, static_argnames=["prefer_aba"])
def forward_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    prefer_aba: float = True,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.
        prefer_aba: Whether to prefer the ABA algorithm over the CRB one.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    forward_dynamics_fn = forward_dynamics_aba if prefer_aba else forward_dynamics_crb

    return forward_dynamics_fn(
        model=model,
        data=data,
        joint_forces=joint_forces,
        link_forces=link_forces,
    )


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def forward_dynamics_aba(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    output_representation: VelRepr | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the ABA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.
        output_representation:
            The desired output velocity representation of the free-floating acceleration.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.
    """

    output_representation = (
        output_representation
        if output_representation is not None
        else data.velocity_representation
    )

    # ============
    # Prepare data
    # ============

    # Build joint forces, if not provided.
    τ = (
        jnp.atleast_1d(joint_forces.squeeze())
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions)
    )

    # Build link forces, if not provided.
    f_L = (
        jnp.atleast_2d(link_forces.squeeze())
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    )

    # Create a references object that simplifies converting among representations.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=τ,
        link_forces=f_L,
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Extract the state in inertial-fixed representation.
    W_p_B = data.base_position
    W_v_WB = data.base_velocity(VelRepr.Inertial)
    W_Q_B = data.base_orientation
    s = data.joint_positions
    ṡ = data.joint_velocities

    # Extract the inputs in inertial-fixed representation.
    W_f_L = references._link_forces
    τ = references._joint_force_references

    # ========================
    # Compute forward dynamics
    # ========================

    W_v̇_WB, s̈ = jaxsim.rbda.aba(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        joint_forces=τ,
        link_forces=W_f_L,
        standard_gravity=model.gravity,
    )

    # =============
    # Adjust output
    # =============

    def to_active(
        W_v̇_WB: jtp.Vector, W_H_C: jtp.Matrix, W_v_WB: jtp.Vector, W_v_WC: jtp.Vector
    ) -> jtp.Vector:
        """
        Convert the inertial-fixed apparent base acceleration W_v̇_WB to
        another representation C_v̇_WB expressed in a generic frame C.
        """

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        C_X_W = Adjoint.from_transform(transform=W_H_C, inverse=True)
        return C_X_W @ (W_v̇_WB - Cross.vx(W_v_WC) @ W_v_WB)

    match output_representation:
        case VelRepr.Inertial:
            # In this case C=W
            W_H_C = W_H_W = jnp.eye(4)  # noqa: F841
            W_v_WC = W_v_WW = jnp.zeros(6)  # noqa: F841

        case VelRepr.Body:
            # In this case C=B
            W_H_C = W_H_B = data._base_transform
            W_v_WC = W_v_WB

        case VelRepr.Mixed:
            # In this case C=B[W]
            W_H_B = data._base_transform
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))  # noqa: F841
            W_ṗ_B = data.base_velocity(output_representation)[0:3]
            W_v_WC = W_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)  # noqa: F841

        case _:
            raise ValueError(output_representation)

    # We need to convert the derivative of the base velocity to the active
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    C_v̇_WB = to_active(
        W_v̇_WB=W_v̇_WB,
        W_H_C=W_H_C,
        W_v_WB=W_v_WB,
        W_v_WC=W_v_WC,
    )

    # The ABA algorithm already returns a zero base 6D acceleration for
    # fixed-based models. However, the to_active function introduces an
    # additional acceleration component in Mixed representation.
    # Here below we make sure that the base acceleration is zero.
    C_v̇_WB = C_v̇_WB if model.floating_base() else jnp.zeros(6)

    return C_v̇_WB.astype(float), s̈.astype(float)


@jax.jit
@js.common.named_scope
def forward_dynamics_crb(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_forces: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute the forward dynamics of the model with the CRB algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_forces:
            The joint forces to consider as a vector of shape `(dofs,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `data.velocity_representation`.

    Returns:
        A tuple containing the 6D acceleration in the active representation of the
        base link and the joint accelerations resulting from the application of the
        considered joint forces and external forces.

    Note:
        Compared to ABA, this method could be significantly slower, especially for
        models with a large number of degrees of freedom.
    """

    # ============
    # Prepare data
    # ============

    # Build joint torques if not provided.
    τ = (
        jnp.atleast_1d(joint_forces)
        if joint_forces is not None
        else jnp.zeros_like(data.joint_positions)
    )

    # Build external forces if not provided.
    f = (
        jnp.atleast_2d(link_forces)
        if link_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    # Compute terms of the floating-base EoM.
    M = free_floating_mass_matrix(model=model, data=data)
    h = free_floating_bias_forces(model=model, data=data)
    S = jnp.block([jnp.zeros(shape=(model.dofs(), 6)), jnp.eye(model.dofs())]).T
    J = generalized_free_floating_jacobian(model=model, data=data)

    # TODO: invert the Mss block exploiting sparsity defined by the parent array λ(i)

    # ========================
    # Compute forward dynamics
    # ========================

    if model.floating_base():
        # l: number of links.
        # g: generalized coordinates, 6 + number of joints.
        JTf = jnp.einsum("l6g,l6->g", J, f)
        ν̇ = jnp.linalg.solve(M, S @ τ - h + JTf)

    else:
        # l: number of links.
        # j: number of joints.
        JTf = jnp.einsum("l6j,l6->j", J[:, :, 6:], f)
        s̈ = jnp.linalg.solve(M[6:, 6:], τ - h[6:] + JTf)

        v̇_WB = jnp.zeros(6)
        ν̇ = jnp.hstack([v̇_WB, s̈.squeeze()])

    # =============
    # Adjust output
    # =============

    # Extract the base acceleration in the active representation.
    # Note that this is an apparent acceleration (relevant in Mixed representation),
    # therefore it cannot be always expressed in different frames with just a
    # 6D transformation X.
    v̇_WB = ν̇[0:6].squeeze().astype(float)

    # Extract the joint accelerations.
    s̈ = jnp.atleast_1d(ν̇[6:].squeeze()).astype(float)

    return v̇_WB, s̈


@jax.jit
@js.common.named_scope
def forward_kinematics(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Matrix:
    """
    Compute the forward kinematics of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The nL x 4 x 4 array containing the stacked homogeneous transformations
        of the links. The first axis is the link index.
    """

    W_H_LL, _ = jaxsim.rbda.forward_kinematics_model(
        model=model,
        base_position=data.base_position,
        base_quaternion=data.base_quaternion,
        joint_positions=data.joint_positions,
        joint_velocities=data.joint_velocities,
        base_linear_velocity_inertial=data._base_linear_velocity,
        base_angular_velocity_inertial=data._base_angular_velocity,
    )

    return W_H_LL


def _transform_M_block(M_body: jtp.Matrix, X: jtp.Matrix) -> jtp.Matrix:
    """
    Apply invTᵀ M_body invT with invT = diag(X, I_n), without forming invT.

    Args:
        M_body: (6+n, 6+n) mass matrix (inverse) in body representation.
        X:      (6, 6) adjoint (e.g. B_X_W or B_X_BW).

    Returns:
        M_repr: (6+n, 6+n) mass matrix (inverse) in the new representation.
    """

    # invTᵀ M invT with invT = diag(X, I):
    # Mbb' = Xᵀ Mbb X
    # Mbj' = Xᵀ Mbj
    # Mjb' = Mjb X
    # Mjj' = Mjj
    Mbb_t = X.T @ M_body[:6, :6] @ X
    Mbj_t = X.T @ M_body[:6, 6:]
    Mjb_t = M_body[6:, :6] @ X
    Mjj_t = M_body[6:, 6:]

    top = jnp.concatenate([Mbb_t, Mbj_t], axis=1)
    bottom = jnp.concatenate([Mjb_t, Mjj_t], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def free_floating_mass_matrix(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating mass matrix of the model with the CRBA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the mass matrix.

    Returns:
        The free-floating mass matrix of the model.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    M_body = jaxsim.rbda.crba(
        model=model,
        joint_positions=data.joint_positions,
    )

    match output_representation:
        case VelRepr.Body:
            return M_body

        case VelRepr.Inertial:
            B_X_W = Adjoint.from_transform(transform=data.base_transform, inverse=True)

            return _transform_M_block(M_body, B_X_W)

        case VelRepr.Mixed:
            BW_H_B = data.base_transform.at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)

            return _transform_M_block(M_body, B_X_BW)
        case _:
            raise ValueError(output_representation)


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def free_floating_mass_matrix_inverse(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the inverse of the free-floating mass matrix of the model
    with the CRBA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the inverse mass matrix.

    Returns:
        The inverse of the free-floating mass matrix of the model.
    """
    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    M_inv_body = jaxsim.rbda.mass_inverse(
        model=model,
        base_position=data.base_position,
        base_quaternion=data.base_orientation,
        joint_positions=data.joint_positions,
    )

    match output_representation:
        case VelRepr.Body:
            return M_inv_body
        case VelRepr.Inertial:
            W_X_B = Adjoint.from_transform(transform=data.base_transform)

            return _transform_M_block(M_inv_body, W_X_B.T)
        case VelRepr.Mixed:
            B_H_BW = data.base_transform.at[0:3, 3].set(jnp.zeros(3))
            BW_X_B = Adjoint.from_transform(transform=B_H_BW)

            return _transform_M_block(M_inv_body, BW_X_B.T)
        case _:
            raise ValueError(output_representation)


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def free_floating_coriolis_matrix(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the free-floating Coriolis matrix of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the Coriolis matrix.

    Returns:
        The free-floating Coriolis matrix of the model.

    Note:
        This function, contrarily to other quantities of the equations of motion,
        does not exploit any iterative algorithm. Therefore, the computation of
        the Coriolis matrix may be much slower than other quantities.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    # We perform all the calculation in body-fixed.
    # The Coriolis matrix computed in this representation is converted later
    # to the active representation stored in data.
    B_ν = data.generalized_velocity(VelRepr.Body)

    # Doubly-left free-floating Jacobian.
    L_J_WL_B = generalized_free_floating_jacobian(
        model=model, data=data, input_representation=VelRepr.Body
    )

    # Doubly-left free-floating Jacobian derivative.
    L_J̇_WL_B = generalized_free_floating_jacobian_derivative(
        model=model, data=data, input_representation=VelRepr.Body
    )

    L_M_L = link_spatial_inertia_matrices(model=model)

    # Body-fixed link velocities.
    # Note: we could have called link.velocity() instead of computing it ourselves,
    # but since we need the link Jacobians later, we can save a double calculation.
    L_v_WL = jax.vmap(lambda J: J @ B_ν)(L_J_WL_B)

    # Compute the contribution of each link to the Coriolis matrix.
    def compute_link_contribution(M, v, J, J̇) -> jtp.Array:
        return J.T @ ((Cross.vx_star(v) @ M + M @ Cross.vx(v)) @ J + M @ J̇)

    C_B_links = jax.vmap(compute_link_contribution)(
        L_M_L,
        L_v_WL,
        L_J_WL_B,
        L_J̇_WL_B,
    )

    # We need to adjust the Coriolis matrix for fixed-base models.
    # In this case, the base link does not contribute to the matrix, and we need to zero
    # the off-diagonal terms mapping joint quantities onto the base configuration.
    if model.floating_base():
        C_B = C_B_links.sum(axis=0)
    else:
        C_B = C_B_links[1:].sum(axis=0)
        C_B = C_B.at[0:6, 6:].set(0.0)
        C_B = C_B.at[6:, 0:6].set(0.0)

    # Adjust the representation of the Coriolis matrix.
    # Refer to https://github.com/traversaro/traversaro-phd-thesis, Section 3.6.
    match output_representation:
        case VelRepr.Body:
            return C_B

        case VelRepr.Inertial:
            n = model.dofs()
            W_H_B = data._base_transform
            B_X_W = jaxsim.math.Adjoint.from_transform(W_H_B, inverse=True)
            B_T_W = jax.scipy.linalg.block_diag(B_X_W, jnp.eye(n))

            W_v_WB = data.base_velocity(VelRepr.Inertial)
            B_Ẋ_W = -B_X_W @ jaxsim.math.Cross.vx(W_v_WB)

            B_Ṫ_W = jax.scipy.linalg.block_diag(B_Ẋ_W, jnp.zeros(shape=(n, n)))

            M = free_floating_mass_matrix(
                model=model, data=data, output_representation=VelRepr.Body
            )

            C = B_T_W.T @ (M @ B_Ṫ_W + C_B @ B_T_W)

            return C

        case VelRepr.Mixed:
            n = model.dofs()
            BW_H_B = data._base_transform.at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = jaxsim.math.Adjoint.from_transform(transform=BW_H_B, inverse=True)
            B_T_BW = jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(n))

            BW_v_WB = data.base_velocity(VelRepr.Mixed)
            BW_v_W_BW = BW_v_WB.at[3:6].set(jnp.zeros(3))

            BW_v_BW_B = BW_v_WB - BW_v_W_BW
            B_Ẋ_BW = -B_X_BW @ jaxsim.math.Cross.vx(BW_v_BW_B)

            B_Ṫ_BW = jax.scipy.linalg.block_diag(B_Ẋ_BW, jnp.zeros(shape=(n, n)))

            M = free_floating_mass_matrix(
                model=model, data=data, output_representation=VelRepr.Body
            )

            C = B_T_BW.T @ (M @ B_Ṫ_BW + C_B @ B_T_BW)

            return C

        case _:
            raise ValueError(output_representation)


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def inverse_dynamics(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    joint_accelerations: jtp.VectorLike | None = None,
    base_acceleration: jtp.VectorLike | None = None,
    link_forces: jtp.MatrixLike | None = None,
    output_representation: VelRepr | None = None,
) -> tuple[jtp.Vector, jtp.Vector]:
    """
    Compute inverse dynamics with the RNEA algorithm.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        joint_accelerations:
            The joint accelerations to consider as a vector of shape `(dofs,)`.
        base_acceleration:
            The base acceleration to consider as a vector of shape `(6,)`.
        link_forces:
            The link 6D forces to consider as a matrix of shape `(nL, 6)`.
            The frame in which they are expressed must be `output_representation`.
        output_representation:
            The desired output velocity representation of the base force.

    Returns:
        A tuple containing the 6D force in the active representation applied to the
        base to obtain the considered base acceleration, and the joint forces to apply
        to obtain the considered joint accelerations.
    """

    output_representation = (
        output_representation
        if output_representation is not None
        else data.velocity_representation
    )

    # ============
    # Prepare data
    # ============

    # Build joint accelerations, if not provided.
    s̈ = (
        jnp.atleast_1d(jnp.array(joint_accelerations).squeeze())
        if joint_accelerations is not None
        else jnp.zeros_like(data.joint_positions)
    )

    # Build base acceleration, if not provided.
    v̇_WB = (
        jnp.array(base_acceleration).squeeze()
        if base_acceleration is not None
        else jnp.zeros(6)
    )

    # Build link forces, if not provided.
    f_L = (
        jnp.atleast_2d(jnp.array(link_forces).squeeze())
        if link_forces is not None
        else jnp.zeros(shape=(model.number_of_links(), 6))
    )

    def to_inertial(C_v̇_WB, W_H_C, C_v_WB, W_v_WC):
        """
        Convert the active representation of the base acceleration C_v̇_WB
        expressed in a generic frame C to the inertial-fixed representation W_v̇_WB.
        """

        W_X_C = Adjoint.from_transform(transform=W_H_C)
        C_X_W = Adjoint.from_transform(transform=W_H_C, inverse=True)
        C_v_WC = C_X_W @ W_v_WC

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        return W_X_C @ (C_v̇_WB + Cross.vx(C_v_WC) @ C_v_WB)

    match output_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)  # noqa: F841
            W_v_WC = W_v_WW = jnp.zeros(6)  # noqa: F841

        case VelRepr.Body:
            W_H_C = W_H_B = data._base_transform
            W_v_WC = W_v_WB = data.base_velocity(VelRepr.Inertial)

        case VelRepr.Mixed:
            W_H_B = data._base_transform
            W_H_C = W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))  # noqa: F841
            W_ṗ_B = data.base_velocity(output_representation)[0:3]
            W_v_WC = W_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)  # noqa: F841

        case _:
            raise ValueError(output_representation)

    # We need to convert the derivative of the base acceleration to the Inertial
    # representation. In Mixed representation, this conversion is not a plain
    # transformation with just X, but it also involves a cross product in ℝ⁶.
    W_v̇_WB = to_inertial(
        C_v̇_WB=v̇_WB,
        W_H_C=W_H_C,
        C_v_WB=data.base_velocity(output_representation),
        W_v_WC=W_v_WC,
    )

    # Create a references object that simplifies converting among representations.
    references = js.references.JaxSimModelReferences.build(
        model=model,
        data=data,
        link_forces=f_L,
        velocity_representation=output_representation,
    )

    # Extract the state in inertial-fixed representation.
    W_p_B = data.base_position
    W_v_WB = data.base_velocity(VelRepr.Inertial)
    W_Q_B = data.base_quaternion
    s = data.joint_positions
    ṡ = data.joint_velocities

    # Extract the inputs in inertial-fixed representation.
    W_f_L = references._link_forces

    # ========================
    # Compute inverse dynamics
    # ========================

    W_f_B, τ = jaxsim.rbda.rnea(
        model=model,
        base_position=W_p_B,
        base_quaternion=W_Q_B,
        joint_positions=s,
        base_linear_velocity=W_v_WB[0:3],
        base_angular_velocity=W_v_WB[3:6],
        joint_velocities=ṡ,
        base_linear_acceleration=W_v̇_WB[0:3],
        base_angular_acceleration=W_v̇_WB[3:6],
        joint_accelerations=s̈,
        link_forces=W_f_L,
        standard_gravity=model.gravity,
    )

    # =============
    # Adjust output
    # =============

    # Express W_f_B in the active representation.
    f_B = js.data.JaxSimModelData.inertial_to_other_representation(
        array=W_f_B,
        other_representation=output_representation,
        transform=data._base_transform,
        is_force=True,
    ).squeeze()

    return f_B.astype(float), τ.astype(float)


@jax.jit
@js.common.named_scope
def free_floating_gravity_forces(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Vector:
    r"""
    Compute the free-floating gravity forces :math:`g(\mathbf{q})` of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the gravity forces.

    Returns:
        The free-floating gravity forces of the model.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    # Build a new state with zeroed velocities.
    data_rnea = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=output_representation,
        base_position=data.base_position,
        base_quaternion=data.base_quaternion,
        joint_positions=data.joint_positions,
    )

    return jnp.hstack(
        inverse_dynamics(
            model=model,
            data=data_rnea,
            # Set zero inputs:
            joint_accelerations=jnp.atleast_1d(jnp.zeros(model.dofs())),
            base_acceleration=jnp.zeros(6),
            link_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
            output_representation=output_representation,
        )
    ).astype(float)


@jax.jit
@js.common.named_scope
def free_floating_bias_forces(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Vector:
    r"""
    Compute the free-floating bias forces :math:`h(\mathbf{q}, \boldsymbol{\nu})`
    of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the bias forces.

    Returns:
        The free-floating bias forces of the model.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    # Set the generalized position and generalized velocity.
    base_linear_velocity, base_angular_velocity = None, None
    if model.floating_base():
        base_velocity = data.base_velocity(output_representation)
        base_linear_velocity = base_velocity[:3]
        base_angular_velocity = base_velocity[3:]

    data_rnea = js.data.JaxSimModelData.build(
        model=model,
        velocity_representation=output_representation,
        base_position=data.base_position,
        base_quaternion=data.base_quaternion,
        joint_positions=data.joint_positions,
        joint_velocities=data.joint_velocities,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
    )

    return jnp.hstack(
        inverse_dynamics(
            model=model,
            data=data_rnea,
            # Set zero inputs:
            joint_accelerations=jnp.atleast_1d(jnp.zeros(model.dofs())),
            base_acceleration=jnp.zeros(6),
            link_forces=jnp.zeros(shape=(model.number_of_links(), 6)),
            output_representation=output_representation,
        )
    ).astype(float)


# ==========================
# Other kinematic quantities
# ==========================


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def locked_spatial_inertia(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the locked 6D inertia matrix of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the locked inertia matrix.

    Returns:
        The locked 6D inertia matrix of the model.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    return total_momentum_jacobian(
        model=model,
        data=data,
        input_representation=output_representation,
        output_vel_repr=output_representation,
    )[:, 0:6]


@jax.jit
@js.common.named_scope
def total_momentum(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    output_representation: VelRepr | None = None,
) -> jtp.Vector:
    """
    Compute the total momentum of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the total momentum.

    Returns:
        The total momentum of the model in the active velocity representation.
    """

    ν = data.generalized_velocity(output_representation)
    Jh = total_momentum_jacobian(
        model=model, data=data, output_vel_repr=output_representation
    )

    return Jh @ ν


@functools.partial(jax.jit, static_argnames=["input_representation", "output_vel_repr"])
def total_momentum_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    input_representation: VelRepr | None = None,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the jacobian of the total momentum.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        input_representation: The input velocity representation of the data.
        output_vel_repr: The output velocity representation of the jacobian.

    Returns:
        The jacobian of the total momentum of the model in the active representation.
    """

    input_representation = (
        data.velocity_representation
        if input_representation is None
        else input_representation
    )

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else input_representation
    )

    if output_vel_repr is input_representation:
        return free_floating_mass_matrix(
            model=model, data=data, output_representation=output_vel_repr
        )[0:6]

    B_Jh_B = free_floating_mass_matrix(
        model=model, data=data, output_representation=VelRepr.Body
    )[0:6]

    match input_representation:
        case VelRepr.Body:
            B_Jh = B_Jh_B

        case VelRepr.Inertial:
            B_X_W = Adjoint.from_transform(transform=data._base_transform, inverse=True)
            B_Jh = B_Jh_B @ jax.scipy.linalg.block_diag(B_X_W, jnp.eye(model.dofs()))

        case VelRepr.Mixed:
            BW_H_B = data._base_transform.at[0:3, 3].set(jnp.zeros(3))
            B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)
            B_Jh = B_Jh_B @ jax.scipy.linalg.block_diag(B_X_BW, jnp.eye(model.dofs()))

        case _:
            raise ValueError(input_representation)

    match output_vel_repr:
        case VelRepr.Body:
            return B_Jh

        case VelRepr.Inertial:
            W_H_B = data._base_transform
            B_Xv_W = Adjoint.from_transform(transform=W_H_B, inverse=True)
            W_Xf_B = B_Xv_W.T
            W_Jh = W_Xf_B @ B_Jh
            return W_Jh

        case VelRepr.Mixed:
            BW_H_B = data._base_transform.at[0:3, 3].set(jnp.zeros(3))
            B_Xv_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)
            BW_Xf_B = B_Xv_BW.T
            BW_Jh = BW_Xf_B @ B_Jh
            return BW_Jh

        case _:
            raise ValueError(output_vel_repr)


@jax.jit
@js.common.named_scope
def average_velocity(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Vector:
    """
    Compute the average velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the average velocity.

    Returns:
        The average velocity of the model computed in the base frame and expressed
        in the active representation.
    """

    ν = data.generalized_velocity(output_representation)
    J = average_velocity_jacobian(
        model=model, data=data, output_vel_repr=output_representation
    )

    return J @ ν


@functools.partial(jax.jit, static_argnames=["output_vel_repr"])
def average_velocity_jacobian(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_vel_repr: VelRepr | None = None,
) -> jtp.Matrix:
    """
    Compute the Jacobian of the average velocity of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_vel_repr: The output velocity representation of the jacobian.

    Returns:
        The Jacobian of the average centroidal velocity of the model in the desired
        representation.
    """

    output_vel_repr = (
        output_vel_repr if output_vel_repr is not None else data.velocity_representation
    )

    # Depending on the velocity representation, the frame G is either G[W] or G[B].
    G_J = js.com.average_centroidal_velocity_jacobian(model=model, data=data)

    match output_vel_repr:
        case VelRepr.Inertial:
            GW_J = G_J
            W_p_CoM = js.com.com_position(model=model, data=data)

            W_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM)
            W_X_GW = Adjoint.from_transform(transform=W_H_GW)

            return W_X_GW @ GW_J

        case VelRepr.Body:
            GB_J = G_J
            W_p_B = data.base_position
            W_p_CoM = js.com.com_position(model=model, data=data)
            B_R_W = jaxsim.math.Quaternion.to_dcm(data.base_orientation).transpose()

            B_H_GB = jnp.eye(4).at[0:3, 3].set(B_R_W @ (W_p_CoM - W_p_B))
            B_X_GB = Adjoint.from_transform(transform=B_H_GB)

            return B_X_GB @ GB_J

        case VelRepr.Mixed:
            GW_J = G_J
            W_p_B = data.base_position
            W_p_CoM = js.com.com_position(model=model, data=data)

            BW_H_GW = jnp.eye(4).at[0:3, 3].set(W_p_CoM - W_p_B)
            BW_X_GW = Adjoint.from_transform(transform=BW_H_GW)

            return BW_X_GW @ GW_J


# ========================
# Other dynamic quantities
# ========================


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def link_bias_accelerations(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    output_representation: VelRepr | None = None,
) -> jtp.Vector:
    r"""
    Compute the bias accelerations of the links of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        output_representation:
            The output velocity representation of the bias accelerations.

    Returns:
        The bias accelerations of the links of the model.

    Note:
        This function computes the component of the total 6D acceleration not due to
        the joint or base acceleration.
        It is often called :math:`\dot{J} \boldsymbol{\nu}`.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    # ================================================
    # Compute the body-fixed zero base 6D acceleration
    # ================================================

    # Compute the base transform.
    W_H_B = data._base_transform

    def other_representation_to_inertial(
        C_v̇_WB: jtp.Vector, C_v_WB: jtp.Vector, W_H_C: jtp.Matrix, W_v_WC: jtp.Vector
    ) -> jtp.Vector:
        """
        Convert the active representation of the base acceleration C_v̇_WB
        expressed in a generic frame C to the inertial-fixed representation W_v̇_WB.
        """

        W_X_C = Adjoint.from_transform(transform=W_H_C)
        C_X_W = Adjoint.from_transform(transform=W_H_C, inverse=True)

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        return W_X_C @ (C_v̇_WB + jaxsim.math.Cross.vx(C_X_W @ W_v_WC) @ C_v_WB)

    # Here we initialize a zero 6D acceleration in the active representation, and
    # convert it to inertial-fixed. This is a useful intermediate representation
    # because the apparent acceleration W_v̇_WB is equal to the intrinsic acceleration
    # W_a_WB, and intrinsic accelerations can be expressed in different frames through
    # a simple C_X_W 6D transform.
    match output_representation:
        case VelRepr.Inertial:
            W_H_C = W_H_W = jnp.eye(4)  # noqa: F841
            W_v_WC = W_v_WW = jnp.zeros(6)  # noqa: F841
            C_v_WB = W_v_WB = data.base_velocity(VelRepr.Inertial)

        case VelRepr.Body:
            W_H_C = W_H_B
            W_v_WC = W_v_WB = data.base_velocity(VelRepr.Inertial)  # noqa: F841
            C_v_WB = B_v_WB = data.base_velocity(VelRepr.Body)

        case VelRepr.Mixed:
            W_H_BW = W_H_B.at[0:3, 0:3].set(jnp.eye(3))
            W_H_C = W_H_BW
            W_ṗ_B = data.base_velocity(VelRepr.Mixed)[0:3]
            BW_v_W_BW = jnp.zeros(6).at[0:3].set(W_ṗ_B)
            W_X_BW = jaxsim.math.Adjoint.from_transform(transform=W_H_BW)
            W_v_WC = W_v_W_BW = W_X_BW @ BW_v_W_BW  # noqa: F841
            C_v_WB = BW_v_WB = data.base_velocity(VelRepr.Mixed)  # noqa: F841

        case _:
            raise ValueError(output_representation)

    # Convert a zero 6D acceleration from the active representation to inertial-fixed.
    W_v̇_WB = other_representation_to_inertial(
        C_v̇_WB=jnp.zeros(6), C_v_WB=C_v_WB, W_H_C=W_H_C, W_v_WC=W_v_WC
    )

    # ===================================
    # Initialize buffers and prepare data
    # ===================================

    # Get the parent array λ(i).
    # Note: λ(0) must not be used, it's initialized to -1.
    λ = model.kin_dyn_parameters.parent_array

    # Compute 6D transforms of the base velocity.
    B_X_W = jaxsim.math.Adjoint.from_transform(transform=W_H_B, inverse=True)

    # Compute the parent-to-child adjoints and the motion subspaces of the joints.
    # These transforms define the relative kinematics of the entire model, including
    # the base transform for both floating-base and fixed-base models.
    i_X_λi = model.kin_dyn_parameters.joint_transforms(
        joint_positions=data.joint_positions, base_transform=W_H_B
    )

    # Extract the joint motion subspaces.
    S = model.kin_dyn_parameters.motion_subspaces

    # Allocate the buffer to store the body-fixed link velocities.
    L_v_WL = jnp.zeros(shape=(model.number_of_links(), 6))

    # Store the base velocity.
    B_v_WB = data.base_velocity(VelRepr.Body)
    L_v_WL = L_v_WL.at[0].set(B_v_WB)

    # Get the joint velocities.
    ṡ = data.joint_velocities

    # Allocate the buffer to store the body-fixed link accelerations,
    # and initialize the base acceleration.
    L_v̇_WL = jnp.zeros(shape=(model.number_of_links(), 6))
    L_v̇_WL = L_v̇_WL.at[0].set(B_X_W @ W_v̇_WB)

    # ======================================
    # Propagate accelerations and velocities
    # ======================================

    # The computation of the bias forces is similar to the forward pass of RNEA,
    # this time with zero base and joint accelerations. Furthermore, here we do
    # not remove gravity during the propagation.

    # Initialize the loop.
    Carry = tuple[jtp.Matrix, jtp.Matrix]
    carry0: Carry = (L_v_WL, L_v̇_WL)

    def propagate_accelerations(carry: Carry, i: jtp.Int) -> tuple[Carry, None]:
        # Initialize index and unpack the carry.
        ii = i - 1
        v, a = carry

        # Get the motion subspace of the joint.
        Si = S[i].squeeze()

        # Project the joint velocity into its motion subspace.
        vJ = Si * ṡ[ii]

        # Propagate the link body-fixed velocity.
        v_i = i_X_λi[i] @ v[λ[i]] + vJ
        v = v.at[i].set(v_i)

        # Propagate the link body-fixed acceleration considering zero joint acceleration.
        s̈ = 0.0
        a_i = i_X_λi[i] @ a[λ[i]] + Si * s̈ + jaxsim.math.Cross.vx(v[i]) @ vJ
        a = a.at[i].set(a_i)

        return (v, a), None

    # Compute the body-fixed velocity and body-fixed apparent acceleration of the links.
    (L_v_WL, L_v̇_WL), _ = (
        jax.lax.scan(
            f=propagate_accelerations,
            init=carry0,
            xs=jnp.arange(start=1, stop=model.number_of_links()),
        )
        if model.number_of_links() > 1
        else [(L_v_WL, L_v̇_WL), None]
    )

    # ===================================================================
    # Convert the body-fixed 6D acceleration to the active representation
    # ===================================================================

    def body_to_other_representation(
        L_v̇_WL: jtp.Vector, L_v_WL: jtp.Vector, C_H_L: jtp.Matrix, L_v_CL: jtp.Vector
    ) -> jtp.Vector:
        """
        Convert the body-fixed apparent acceleration L_v̇_WL to
        another representation C_v̇_WL expressed in a generic frame C.
        """

        # In Mixed representation, we need to include a cross product in ℝ⁶.
        # In Inertial and Body representations, the cross product is always zero.
        C_X_L = jaxsim.math.Adjoint.from_transform(transform=C_H_L)
        return C_X_L @ (L_v̇_WL + jaxsim.math.Cross.vx(L_v_CL) @ L_v_WL)

    match output_representation:
        case VelRepr.Body:
            C_H_L = L_H_L = jnp.stack(  # noqa: F841
                [jnp.eye(4)] * model.number_of_links()
            )
            L_v_CL = L_v_LL = jnp.zeros(  # noqa: F841
                shape=(model.number_of_links(), 6)
            )

        case VelRepr.Inertial:
            C_H_L = W_H_L = data._link_transforms
            L_v_CL = L_v_WL

        case VelRepr.Mixed:
            W_H_L = data._link_transforms
            LW_H_L = jax.vmap(lambda W_H_L: W_H_L.at[0:3, 3].set(jnp.zeros(3)))(W_H_L)
            C_H_L = LW_H_L
            L_v_CL = L_v_LW_L = jax.vmap(  # noqa: F841
                lambda v: v.at[0:3].set(jnp.zeros(3))
            )(L_v_WL)

        case _:
            raise ValueError(output_representation)

    # Convert from body-fixed to the active representation.
    O_v̇_WL = jax.vmap(body_to_other_representation)(
        L_v̇_WL=L_v̇_WL, L_v_WL=L_v_WL, C_H_L=C_H_L, L_v_CL=L_v_CL
    )

    return O_v̇_WL


# ======
# Energy
# ======


@jax.jit
@js.common.named_scope
def mechanical_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the mechanical energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The mechanical energy of the model.
    """

    K = kinetic_energy(model=model, data=data)
    U = potential_energy(model=model, data=data)

    return (K + U).astype(float)


@jax.jit
@js.common.named_scope
def kinetic_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the kinetic energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The kinetic energy of the model.
    """

    B_ν = data.generalized_velocity(VelRepr.Body)
    M_B = free_floating_mass_matrix(
        model=model, data=data, output_representation=VelRepr.Body
    )

    K = 0.5 * B_ν.T @ M_B @ B_ν
    return K.squeeze().astype(float)


@jax.jit
@js.common.named_scope
def potential_energy(model: JaxSimModel, data: js.data.JaxSimModelData) -> jtp.Float:
    """
    Compute the potential energy of the model.

    Args:
        model: The model to consider.
        data: The data of the considered model.

    Returns:
        The potential energy of the model.
    """

    m = total_mass(model=model)
    W_p̃_CoM = jnp.hstack([js.com.com_position(model=model, data=data), 1])
    return jnp.sum((m * W_p̃_CoM)[2] * model.gravity)


# ===================
# Hw parametrization
# ===================


@jax.jit
@js.common.named_scope
def update_hw_parameters(
    model: JaxSimModel, scaling_factors: ScalingFactors
) -> JaxSimModel:
    """
    Update the hardware parameters of the model by scaling the parameters of the links.

    This function applies scaling factors to the hardware metadata of the links,
    updating their shape, dimensions, density, and other related parameters. It
    recalculates the mass and inertia tensors of the links based on the updated
    metadata and adjusts the joint model transforms accordingly.

    Args:
        model: The JaxSimModel object to update.
        scaling_factors: A ScalingFactors object containing scaling factors for
                         dimensions and density of the links.

    Returns:
        The updated JaxSimModel object with modified hardware parameters.
    """

    kin_dyn_params: KinDynParameters = model.kin_dyn_parameters
    link_parameters: LinkParameters = kin_dyn_params.link_parameters
    hw_link_metadata: HwLinkMetadata = kin_dyn_params.hw_link_metadata

    has_joints = model.number_of_joints() > 0

    supported_case = lambda hw_metadata, scaling_factors: HwLinkMetadata.apply_scaling(
        hw_metadata=hw_metadata, scaling_factors=scaling_factors, has_joints=has_joints
    )
    unsupported_case = lambda hw_metadata, scaling_factors: hw_metadata

    # Apply scaling to hw_link_metadata using vmap
    updated_hw_link_metadata = jax.vmap(
        lambda hw_metadata, multipliers: jax.lax.cond(
            hw_metadata.link_shape == LinkParametrizableShape.Unsupported,
            unsupported_case,
            supported_case,
            hw_metadata,
            multipliers,
        )
    )(hw_link_metadata, scaling_factors)

    # Compute mass and inertia once and unpack the results
    m_updated, I_com_updated = HwLinkMetadata.compute_mass_and_inertia(
        updated_hw_link_metadata
    )

    # Rotate the inertia tensor at CoM with the link orientation, and store
    # it in KynDynParameters.
    I_L_updated = jax.vmap(
        lambda metadata, I_com: metadata.L_H_G[:3, :3]
        @ I_com
        @ metadata.L_H_G[:3, :3].T
    )(updated_hw_link_metadata, I_com_updated)

    # Update link parameters
    updated_link_parameters = link_parameters.replace(
        mass=m_updated,
        inertia_elements=jax.vmap(LinkParameters.flatten_inertia_tensor)(I_L_updated),
        center_of_mass=jax.vmap(lambda metadata: metadata.L_H_G[:3, 3])(
            updated_hw_link_metadata
        ),
    )

    if kin_dyn_params.contact_parameters.body:
        # Compute the contact parameters
        points = HwLinkMetadata.compute_contact_points(
            original_contact_params=kin_dyn_params.contact_parameters,
            link_shapes=updated_hw_link_metadata.link_shape,
            original_com_positions=link_parameters.center_of_mass,
            updated_com_positions=updated_link_parameters.center_of_mass,
            scaling_factors=scaling_factors,
        )

        # Update contact parameters
        updated_contact_parameters = kin_dyn_params.contact_parameters.replace(
            point=points
        )
    else:
        updated_contact_parameters = kin_dyn_params.contact_parameters

    # Update joint model transforms (λ_H_pre)
    def update_λ_H_pre(joint_index):
        # Extract the transforms and masks for the current joint index across all links
        L_H_pre_for_joint = updated_hw_link_metadata.L_H_pre[:, joint_index]
        L_H_pre_mask_for_joint = updated_hw_link_metadata.L_H_pre_mask[:, joint_index]

        # Select the first valid transform (if any) using the mask
        first_valid_index = jnp.argmax(L_H_pre_mask_for_joint)
        selected_transform = L_H_pre_for_joint[first_valid_index]

        # Check if any valid transform exists
        has_valid_transform = L_H_pre_mask_for_joint.any()

        # Fallback to the original λ_H_pre if no valid transform exists
        fallback_transform = kin_dyn_params.joint_model.λ_H_pre[joint_index + 1]

        # Return the selected transform or fallback
        return jnp.where(has_valid_transform, selected_transform, fallback_transform)

    if has_joints:
        # Apply the update function to all joint indices
        updated_λ_H_pre = jax.vmap(update_λ_H_pre)(
            jnp.arange(kin_dyn_params.number_of_joints())
        )

        # NOTE: λ_H_pre should be of len (1+n_joints) with the 0-th element equal
        # to identity to represent the world-to-base tree transform. See JointModel class
        updated_λ_H_pre_with_base = jnp.concatenate(
            (jnp.eye(4).reshape(1, 4, 4), updated_λ_H_pre), axis=0
        )

        # Replace the joint model with the updated transforms
        updated_joint_model = kin_dyn_params.joint_model.replace(
            λ_H_pre=updated_λ_H_pre_with_base
        )

    else:
        # If there are no joints, we can just use the identity transform
        updated_joint_model = kin_dyn_params.joint_model

    # Replace the kin_dyn_parameters with updated values
    updated_kin_dyn_params = kin_dyn_params.replace(
        link_parameters=updated_link_parameters,
        contact_parameters=updated_contact_parameters,
        hw_link_metadata=updated_hw_link_metadata,
        joint_model=updated_joint_model,
    )

    # Return the updated model
    return model.replace(kin_dyn_parameters=updated_kin_dyn_params)


# ==========
# Simulation
# ==========


@functools.partial(jax.jit, static_argnames=["output_representation"])
@js.common.named_scope
def step(
    model: JaxSimModel,
    data: js.data.JaxSimModelData,
    *,
    link_forces: jtp.MatrixLike | None = None,
    joint_force_references: jtp.VectorLike | None = None,
    output_representation: VelRepr | None = None,
) -> js.data.JaxSimModelData:
    """
    Perform a simulation step.

    Args:
        model: The model to consider.
        data: The data of the considered model.
        dt: The time step to consider. If not specified, it is read from the model.
        link_forces:
            The 6D forces to apply to the links expressed in same representation of data.
        joint_force_references: The joint force references to consider.
        output_representation:
            The output velocity representation of the data object after the step.
            Defaults to the same representation of data.

    Returns:
        The new data of the model after the simulation step.

    Note:
        In order to reduce the occurrences of frame conversions performed internally,
        it is recommended to use inertial-fixed velocity representation. This can be
        particularly useful for automatically differentiated logic.
    """

    output_representation = (
        data.velocity_representation
        if output_representation is None
        else output_representation
    )

    # TODO: some contact models here may want to perform a dynamic filtering of
    # the enabled collidable points

    # Extract the inputs
    O_f_L_external = jnp.atleast_2d(
        jnp.array(link_forces, dtype=float).squeeze()
        if link_forces is not None
        else jnp.zeros((model.number_of_links(), 6))
    )

    # Get the external forces in inertial-fixed representation.
    W_f_L_external = js.data.JaxSimModelData.other_representation_to_inertial(
        O_f_L_external,
        other_representation=output_representation,
        transform=data._link_transforms,
        is_force=True,
    )

    τ_references = jnp.atleast_1d(
        jnp.array(joint_force_references, dtype=float).squeeze()
        if joint_force_references is not None
        else jnp.zeros(model.dofs())
    )

    # ================================
    # Compute the total joint torques
    # ================================

    τ_total = js.actuation_model.compute_resultant_torques(
        model, data, joint_force_references=τ_references
    )

    # =============================
    # Advance the simulation state
    # =============================

    from .integrators import _INTEGRATORS_MAP

    integrator_fn = _INTEGRATORS_MAP[model.integrator]

    data_tf = integrator_fn(
        model=model,
        data=data,
        link_forces=W_f_L_external,
        joint_torques=τ_total,
    )

    data_tf = model.contact_model.update_velocity_after_impact(
        model=model, data=data_tf
    )

    return data_tf
