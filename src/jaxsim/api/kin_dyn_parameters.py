from __future__ import annotations

import dataclasses
from itertools import starmap
from typing import ClassVar

import jax.lax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import numpy.typing as npt
from jax_dataclasses import Static

import jaxsim
import jaxsim.typing as jtp
from jaxsim.math import Inertia, JointModel, supported_joint_motion
from jaxsim.math.adjoint import Adjoint
from jaxsim.parsers.descriptions import JointDescription, JointType, ModelDescription
from jaxsim.utils import HashedNumpyArray, JaxsimDataclass


@jax_dataclasses.pytree_dataclass(eq=False, unsafe_hash=False)
class KinDynParameters(JaxsimDataclass):
    r"""
    Class storing the kinematic and dynamic parameters of a model.

    Attributes:
        link_names: The names of the links.
        parent_array: The parent array :math:`\lambda(i)` of the model.
        support_body_array_bool:
            The boolean support parent array :math:`\kappa_{b}(i)` of the model.
        link_parameters: The parameters of the links.
        frame_parameters: The parameters of the frames.
        contact_parameters: The parameters of the collidable points.
        joint_model: The joint model of the model.
        joint_parameters: The parameters of the joints.
        hw_link_metadata: The hardware parameters of the model links.
        constraints: The kinematic constraints of the model. They can be used only with Relaxed-Rigid contact model.
    """

    # Static
    link_names: Static[tuple[str]]
    _parent_array: Static[HashedNumpyArray]
    _support_body_array_bool: Static[HashedNumpyArray]
    _motion_subspaces: Static[HashedNumpyArray]

    # Links
    link_parameters: LinkParameters

    # Contacts
    contact_parameters: ContactParameters

    # Frames
    frame_parameters: FrameParameters

    # Joints
    joint_model: JointModel
    joint_parameters: JointParameters | None

    # Model hardware parameters
    hw_link_metadata: HwLinkMetadata | None = dataclasses.field(default=None)

    # Kinematic constraints
    constraints: ConstraintMap | None = dataclasses.field(default=None)

    @property
    def motion_subspaces(self) -> jtp.Matrix:
        r"""
        Return the motion subspaces :math:`\mathbf{S}(s)` of the joints.
        """
        return self._motion_subspaces.get()

    @property
    def parent_array(self) -> jtp.Vector:
        r"""
        Return the parent array :math:`\lambda(i)` of the model.
        """
        return self._parent_array.get()

    @property
    def support_body_array_bool(self) -> jtp.Matrix:
        r"""
        Return the boolean support parent array :math:`\kappa_{b}(i)` of the model.
        """
        return self._support_body_array_bool.get()

    @staticmethod
    def build(
        model_description: ModelDescription, constraints: ConstraintMap | None
    ) -> KinDynParameters:
        """
        Construct the kinematic and dynamic parameters of the model.

        Args:
            model_description: The parsed model description to consider.
            constraints: An object of type ConstraintMap specifying the kinematic constraint of the model.

        Returns:
            The kinematic and dynamic parameters of the model.

        Note:
            This class is meant to ease the management of parametric models in
            an automatic differentiation context.
        """

        # Extract the links ordered by their index.
        # The link index corresponds to the body index ∈ [0, num_bodies - 1].
        ordered_links = sorted(
            list(model_description.links_dict.values()),
            key=lambda l: l.index,
        )

        # Extract the joints ordered by their index.
        # The joint index matches the index of its child link, therefore it starts
        # from 1. Keep this in mind since this 1-indexing might introduce bugs.
        ordered_joints = sorted(
            list(model_description.joints_dict.values()),
            key=lambda j: j.index,
        )

        # ================
        # Links properties
        # ================

        # Create a list of link parameters objects.
        link_parameters_list = [
            LinkParameters.build_from_spatial_inertia(index=link.index, M=link.inertia)
            for link in ordered_links
        ]

        # Create a vectorized object of link parameters.
        link_parameters = jax.tree.map(lambda *l: jnp.stack(l), *link_parameters_list)

        # =================
        # Joints properties
        # =================

        # Create a list of joint parameters objects.
        joint_parameters_list = [
            JointParameters.build_from_joint_description(joint_description=joint)
            for joint in ordered_joints
        ]

        # Create a vectorized object of joint parameters.
        joint_parameters = (
            jax.tree.map(lambda *l: jnp.stack(l), *joint_parameters_list)
            if ordered_joints
            else JointParameters(
                index=jnp.array([], dtype=int),
                friction_static=jnp.array([], dtype=float),
                friction_viscous=jnp.array([], dtype=float),
                position_limits_min=jnp.array([], dtype=float),
                position_limits_max=jnp.array([], dtype=float),
                position_limit_spring=jnp.array([], dtype=float),
                position_limit_damper=jnp.array([], dtype=float),
            )
        )

        # Create an object that defines the joint model (parent-to-child transforms).
        joint_model = JointModel.build(description=model_description)

        # ===================
        # Contacts properties
        # ===================

        # Create the object storing the parameters of collidable points.
        # Note that, contrarily to LinkParameters and JointsParameters, this object
        # is not created with vmap. This is because the "body" attribute of the object
        # must be Static for JIT-related reasons, and tree_map would not consider it
        # as a leaf.
        contact_parameters = ContactParameters.build_from(
            model_description=model_description
        )

        # =================
        # Frames properties
        # =================

        # Create the object storing the parameters of frames.
        # Note that, contrarily to LinkParameters and JointsParameters, this object
        # is not created with vmap. This is because the "name" attribute of the object
        # must be Static for JIT-related reasons, and tree_map would not consider it
        # as a leaf.
        frame_parameters = FrameParameters.build_from(
            model_description=model_description
        )

        # ===============
        # Tree properties
        # ===============

        # Build the parent array λ(i) of the model.
        # Note: the parent of the base link is not set since it's not defined.
        parent_array_dict = {
            link.index: model_description.links_dict[link.parent_name].index
            for link in ordered_links
            if link.parent_name is not None
        }
        parent_array = jnp.array([-1, *list(parent_array_dict.values())], dtype=int)

        # Instead of building the support parent array κ(i) for each link of the model,
        # that has a variable length depending on the number of links connecting the
        # root to the i-th link, we build the corresponding boolean version.
        # Given a link index i, the boolean support parent array κb(i) is an array
        # with the same number of elements of λ(i) having the i-th element set to True
        # if the i-th link is in the support parent array κ(i), False otherwise.
        # We store the boolean κb(i) as static attribute of the PyTree so that
        # algorithms that need to access it can be jit-compiled.
        def κb(link_index: jtp.IntLike) -> jtp.Vector:
            κb = jnp.zeros(len(ordered_links), dtype=bool)

            carry0 = κb, link_index

            def scan_body(carry: tuple, i: jtp.Int) -> tuple[tuple, None]:
                κb, active_link_index = carry

                κb, active_link_index = jax.lax.cond(
                    pred=(i == active_link_index),
                    false_fun=lambda: (κb, active_link_index),
                    true_fun=lambda: (
                        κb.at[active_link_index].set(True),
                        parent_array[active_link_index],
                    ),
                )

                return (κb, active_link_index), None

            (κb, _), _ = jax.lax.scan(
                f=scan_body,
                init=carry0,
                xs=jnp.flip(jnp.arange(start=0, stop=len(ordered_links))),
            )

            return κb

        support_body_array_bool = jax.vmap(κb)(
            jnp.arange(start=0, stop=len(ordered_links))
        )

        def motion_subspace(joint_type: int, axis: npt.ArrayLike) -> npt.ArrayLike:
            S = {
                JointType.Fixed: np.zeros(shape=(6, 1)),
                JointType.Revolute: np.vstack(np.hstack([np.zeros(3), axis.axis])),
                JointType.Prismatic: np.vstack(np.hstack([axis.axis, np.zeros(3)])),
            }

            return S[joint_type]

        S_J = jnp.array(
            list(
                starmap(
                    motion_subspace,
                    zip(
                        joint_model.joint_types[1:], joint_model.joint_axis, strict=True
                    ),
                )
            )
            if len(joint_model.joint_axis) != 0
            else jnp.empty((0, 6, 1))
        )

        motion_subspaces = jnp.vstack([jnp.zeros((6, 1))[jnp.newaxis, ...], S_J])

        # ===========
        # Constraints
        # ===========

        constraints = ConstraintMap() if constraints is None else constraints

        # =================================
        # Build and return KinDynParameters
        # =================================

        return KinDynParameters(
            link_names=tuple(l.name for l in ordered_links),
            _parent_array=HashedNumpyArray(array=parent_array),
            _support_body_array_bool=HashedNumpyArray(array=support_body_array_bool),
            _motion_subspaces=HashedNumpyArray(array=motion_subspaces),
            link_parameters=link_parameters,
            joint_model=joint_model,
            joint_parameters=joint_parameters,
            contact_parameters=contact_parameters,
            frame_parameters=frame_parameters,
            constraints=constraints,
        )

    def __eq__(self, other: KinDynParameters) -> bool:
        if not isinstance(other, KinDynParameters):
            return False

        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.number_of_links()),
                hash(self.number_of_joints()),
                hash(self.frame_parameters.name),
                hash(self.frame_parameters.body),
                hash(self._parent_array),
                hash(self._support_body_array_bool),
            )
        )

    # =============================
    # Helpers to extract parameters
    # =============================

    def number_of_links(self) -> int:
        """
        Return the number of links of the model.

        Returns:
            The number of links of the model.
        """

        return len(self.link_names)

    def number_of_joints(self) -> int:
        """
        Return the number of joints of the model.

        Returns:
            The number of joints of the model.
        """

        return len(self.joint_model.joint_names) - 1

    def number_of_frames(self) -> int:
        """
        Return the number of frames of the model.

        Returns:
            The number of frames of the model.
        """

        return len(self.frame_parameters.name)

    def support_body_array(self, link_index: jtp.IntLike) -> jtp.Vector:
        r"""
        Return the support parent array :math:`\kappa(i)` of a link.

        Args:
            link_index: The index of the link.

        Returns:
            The support parent array :math:`\kappa(i)` of the link.

        Note:
            This method returns a variable-length vector. In jit-compiled functions,
            it's better to use the (static) boolean version `support_body_array_bool`.
        """

        return jnp.array(
            jnp.where(self.support_body_array_bool[link_index])[0], dtype=int
        )

    # ========================
    # Quantities used by RBDAs
    # ========================

    @jax.jit
    def links_spatial_inertia(self) -> jtp.Array:
        """
        Return the spatial inertia of all links of the model.

        Returns:
            The spatial inertia of all links of the model.
        """

        return jax.vmap(LinkParameters.spatial_inertia)(self.link_parameters)

    @jax.jit
    def tree_transforms(self) -> jtp.Array:
        r"""
        Return the tree transforms of the model.

        Returns:
            The transforms
            :math:`{}^{\text{pre}(i)} H_{\lambda(i)}`
            of all joints of the model.
        """

        pre_Xi_λ = jax.vmap(
            lambda i: self.joint_model.parent_H_predecessor(joint_index=i)
            .inverse()
            .adjoint()
        )(jnp.arange(1, self.number_of_joints() + 1))

        return jnp.vstack(
            [
                jnp.zeros(shape=(1, 6, 6), dtype=float),
                pre_Xi_λ,
            ]
        )

    @jax.jit
    def joint_transforms(
        self, joint_positions: jtp.VectorLike, base_transform: jtp.MatrixLike
    ) -> jtp.Array:
        r"""
        Return the transforms of the joints.

        Args:
            joint_positions: The joint positions.
            base_transform: The homogeneous matrix defining the base pose.

        Returns:
            The stacked transforms
            :math:`{}^{i} \mathbf{H}_{\lambda(i)}(s)`
            of each joint.
        """

        # Rename the base transform.
        W_H_B = base_transform

        # Extract the parent-to-predecessor fixed transforms of the joints.
        λ_H_pre = jnp.vstack(
            [
                jnp.eye(4)[jnp.newaxis],
                self.joint_model.λ_H_pre[1 : 1 + self.number_of_joints()],
            ]
        )
        if self.number_of_joints() == 0:
            pre_H_suc_J = jnp.empty((0, 4, 4))
        else:
            pre_H_suc_J = jax.vmap(supported_joint_motion)(
                joint_types=jnp.array(self.joint_model.joint_types[1:]).astype(int),
                joint_positions=jnp.array(joint_positions),
                joint_axes=jnp.array([j.axis for j in self.joint_model.joint_axis]),
            )

        # Extract the transforms and motion subspaces of the joints.
        # We stack the base transform W_H_B at index 0, and a dummy motion subspace
        # for either the fixed or free-floating joint connecting the world to the base.
        pre_H_suc = jnp.vstack([W_H_B[jnp.newaxis, ...], pre_H_suc_J])

        # Extract the successor-to-child fixed transforms.
        # Note that here we include also the index 0 since suc_H_child[0] stores the
        # optional pose of the base link w.r.t. the root frame of the model.
        # This is supported by SDF when the base link <pose> element is defined.
        suc_H_i = self.joint_model.suc_H_i[jnp.arange(0, 1 + self.number_of_joints())]

        # Compute the overall transforms from the parent to the child of each joint by
        # composing all the components of our joint model.
        i_X_λ = jax.vmap(
            lambda λ_Hi_pre, pre_Hi_suc, suc_Hi_i: Adjoint.from_transform(
                transform=λ_Hi_pre @ pre_Hi_suc @ suc_Hi_i, inverse=True
            )
        )(λ_H_pre, pre_H_suc, suc_H_i)

        return i_X_λ

    # ============================
    # Helpers to update parameters
    # ============================

    def set_link_mass(
        self, link_index: jtp.IntLike, mass: jtp.FloatLike
    ) -> KinDynParameters:
        """
        Set the mass of a link.

        Args:
            link_index: The index of the link.
            mass: The mass of the link.

        Returns:
            The updated kinematic and dynamic parameters of the model.
        """

        link_parameters = self.link_parameters.replace(
            mass=self.link_parameters.mass.at[link_index].set(mass)
        )

        return self.replace(link_parameters=link_parameters)

    def set_link_inertia(
        self, link_index: jtp.IntLike, inertia: jtp.MatrixLike
    ) -> KinDynParameters:
        r"""
        Set the inertia tensor of a link.

        Args:
            link_index: The index of the link.
            inertia: The :math:`3 \times 3` inertia tensor of the link.

        Returns:
            The updated kinematic and dynamic parameters of the model.
        """

        inertia_elements = LinkParameters.flatten_inertia_tensor(I=inertia)

        link_parameters = self.link_parameters.replace(
            inertia_elements=self.link_parameters.inertia_elements.at[link_index].set(
                inertia_elements
            )
        )

        return self.replace(link_parameters=link_parameters)


@jax_dataclasses.pytree_dataclass
class JointParameters(JaxsimDataclass):
    """
    Class storing the parameters of a joint.

    Attributes:
        index: The index of the joint.
        friction_static: The static friction of the joint.
        friction_viscous: The viscous friction of the joint.
        position_limits_min: The lower position limit of the joint.
        position_limits_max: The upper position limit of the joint.
        position_limit_spring: The spring constant of the position limit.
        position_limit_damper: The damper constant of the position limit.

    Note:
        This class is used inside KinDynParameters to store the vectorized set
        of joint parameters.
    """

    index: jtp.Int

    friction_static: jtp.Float
    friction_viscous: jtp.Float

    position_limits_min: jtp.Float
    position_limits_max: jtp.Float

    position_limit_spring: jtp.Float
    position_limit_damper: jtp.Float

    @staticmethod
    def build_from_joint_description(
        joint_description: JointDescription,
    ) -> JointParameters:
        """
        Build a JointParameters object from a joint description.

        Args:
            joint_description: The joint description to consider.

        Returns:
            The JointParameters object.
        """

        s_min = joint_description.position_limit[0]
        s_max = joint_description.position_limit[1]

        position_limits_min = jnp.minimum(s_min, s_max)
        position_limits_max = jnp.maximum(s_min, s_max)

        friction_static = jnp.array(joint_description.friction_static).squeeze()
        friction_viscous = jnp.array(joint_description.friction_viscous).squeeze()

        position_limit_spring = jnp.array(
            joint_description.position_limit_spring
        ).squeeze()

        position_limit_damper = jnp.array(
            joint_description.position_limit_damper
        ).squeeze()

        return JointParameters(
            index=jnp.array(joint_description.index).squeeze().astype(int),
            friction_static=friction_static.astype(float),
            friction_viscous=friction_viscous.astype(float),
            position_limits_min=position_limits_min.astype(float),
            position_limits_max=position_limits_max.astype(float),
            position_limit_spring=position_limit_spring.astype(float),
            position_limit_damper=position_limit_damper.astype(float),
        )


@jax_dataclasses.pytree_dataclass
class LinkParameters(JaxsimDataclass):
    r"""
    Class storing the parameters of a link.

    Attributes:
        index: The index of the link.
        mass: The mass of the link.
        inertia_elements:
            The unique elements of the :math:`3 \times 3` inertia tensor of the link.
        center_of_mass:
            The translation :math:`{}^L \mathbf{p}_{\text{CoM}}` between the origin
            of the link frame and the link's center of mass, expressed in the
            coordinates of the link frame.

    Note:
        This class is used inside KinDynParameters to store the vectorized set
        of link parameters.
    """

    index: jtp.Int

    mass: jtp.Float
    center_of_mass: jtp.Vector
    inertia_elements: jtp.Vector

    @staticmethod
    def build_from_spatial_inertia(index: jtp.IntLike, M: jtp.Matrix) -> LinkParameters:
        r"""
        Build a LinkParameters object from a :math:`6 \times 6` spatial inertia matrix.

        Args:
            index: The index of the link.
            M: The :math:`6 \times 6` spatial inertia matrix of the link.

        Returns:
            The LinkParameters object.
        """

        # Extract the link parameters from the 6D spatial inertia.
        m, L_p_CoM, I_CoM = Inertia.to_params(M=M)

        # Extract only the necessary elements of the inertia tensor.
        inertia_elements = I_CoM[jnp.triu_indices(3)]

        return LinkParameters(
            index=jnp.array(index).squeeze().astype(int),
            mass=jnp.array(m).squeeze().astype(float),
            center_of_mass=jnp.atleast_1d(jnp.array(L_p_CoM).squeeze()).astype(float),
            inertia_elements=jnp.atleast_1d(inertia_elements.squeeze()).astype(float),
        )

    @staticmethod
    def build_from_inertial_parameters(
        index: jtp.IntLike, m: jtp.FloatLike, I: jtp.MatrixLike, c: jtp.VectorLike
    ) -> LinkParameters:
        r"""
        Build a LinkParameters object from the inertial parameters of a link.

        Args:
            index: The index of the link.
            m: The mass of the link.
            I: The :math:`3 \times 3` inertia tensor of the link.
            c: The translation between the link frame and the link's center of mass.

        Returns:
            The LinkParameters object.
        """

        # Extract only the necessary elements of the inertia tensor.
        inertia_elements = I[jnp.triu_indices(3)]

        return LinkParameters(
            index=jnp.array(index).squeeze().astype(int),
            mass=jnp.array(m).squeeze().astype(float),
            center_of_mass=jnp.atleast_1d(c.squeeze()).astype(float),
            inertia_elements=jnp.atleast_1d(inertia_elements.squeeze()).astype(float),
        )

    @staticmethod
    def build_from_flat_parameters(
        index: jtp.IntLike, parameters: jtp.VectorLike
    ) -> LinkParameters:
        """
        Build a LinkParameters object from a flat vector of parameters.

        Args:
            index: The index of the link.
            parameters: The flat vector of parameters.

        Returns:
            The LinkParameters object.
        """
        index = jnp.array(index).squeeze().astype(int)

        m = jnp.array(parameters[0]).squeeze().astype(float)
        c = jnp.atleast_1d(parameters[1:4].squeeze()).astype(float)
        inertia_elements = jnp.atleast_1d(parameters[4:].squeeze()).astype(float)

        return LinkParameters(
            index=index, mass=m, inertia_elements=inertia_elements, center_of_mass=c
        )

    @staticmethod
    def flat_parameters(params: LinkParameters) -> jtp.Vector:
        """
        Return the parameters of a link as a flat vector.

        Args:
            params: The link parameters.

        Returns:
            The parameters of the link as a flat vector.
        """

        return (
            jnp.hstack(
                [
                    params.mass,
                    params.center_of_mass.squeeze(),
                    params.inertia_elements,
                ]
            )
            .squeeze()
            .astype(float)
        )

    @staticmethod
    def inertia_tensor(params: LinkParameters) -> jtp.Matrix:
        r"""
        Return the :math:`3 \times 3` inertia tensor of a link.

        Args:
            params: The link parameters.

        Returns:
            The :math:`3 \times 3` inertia tensor of the link.
        """

        return LinkParameters.unflatten_inertia_tensor(
            inertia_elements=params.inertia_elements
        )

    @staticmethod
    def spatial_inertia(params: LinkParameters) -> jtp.Matrix:
        r"""
        Return the :math:`6 \times 6` spatial inertia matrix of a link.

        Args:
            params: The link parameters.

        Returns:
            The :math:`6 \times 6` spatial inertia matrix of the link.
        """

        return Inertia.to_sixd(
            mass=params.mass,
            I=LinkParameters.inertia_tensor(params),
            com=params.center_of_mass,
        )

    @staticmethod
    def flatten_inertia_tensor(I: jtp.Matrix) -> jtp.Vector:
        r"""
        Flatten a :math:`3 \times 3` inertia tensor into a vector of unique elements.

        Args:
            I: The :math:`3 \times 3` inertia tensor.

        Returns:
            The vector of unique elements of the inertia tensor.
        """

        return jnp.atleast_1d(I[jnp.triu_indices(3)].squeeze())

    @staticmethod
    def unflatten_inertia_tensor(inertia_elements: jtp.Vector) -> jtp.Matrix:
        r"""
        Unflatten a vector of unique elements into a :math:`3 \times 3` inertia tensor.

        Args:
            inertia_elements: The vector of unique elements of the inertia tensor.

        Returns:
            The :math:`3 \times 3` inertia tensor.
        """

        I = jnp.zeros([3, 3]).at[jnp.triu_indices(3)].set(inertia_elements.squeeze())
        return jnp.atleast_2d(jnp.where(I, I, I.T)).astype(float)


@jax_dataclasses.pytree_dataclass
class ContactParameters(JaxsimDataclass):
    """
    Class storing the contact parameters of a model.

    Attributes:
        body:
            A tuple of integers representing, for each collidable point, the index of
            the body (link) to which it is rigidly attached to.
        point:
            The translations between the link frame and the collidable point, expressed
            in the coordinates of the parent link frame.
        enabled:
            A tuple of booleans representing, for each collidable point, whether it is
            enabled or not in contact models.

    Note:
        Contrarily to LinkParameters and JointParameters, this class is not meant
        to be created with vmap. This is because the `body` attribute must be `Static`.
    """

    body: Static[tuple[int, ...]] = dataclasses.field(default_factory=tuple)

    point: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.array([]))

    enabled: Static[tuple[bool, ...]] = dataclasses.field(default_factory=tuple)

    @property
    def indices_of_enabled_collidable_points(self) -> npt.NDArray:
        """
        Return the indices of the enabled collidable points.
        """
        return np.where(np.array(self.enabled))[0]

    @staticmethod
    def build_from(model_description: ModelDescription) -> ContactParameters:
        """
        Build a ContactParameters object from a model description.

        Args:
            model_description: The model description to consider.

        Returns:
            The ContactParameters object.
        """

        if len(model_description.collision_shapes) == 0:
            return ContactParameters()

        # Get all the links so that we can take their updated index.
        links_dict = {link.name: link for link in model_description}

        # Get all the enabled collidable points of the model.
        collidable_points = model_description.all_enabled_collidable_points()

        # Extract the positions L_p_C of the collidable points w.r.t. the link frames
        # they are rigidly attached to.
        points = jnp.vstack([cp.position for cp in collidable_points])

        # Extract the indices of the links to which the collidable points are rigidly
        # attached to.
        link_index_of_points = tuple(
            links_dict[cp.parent_link.name].index for cp in collidable_points
        )

        # Build the ContactParameters object.
        cp = ContactParameters(
            point=points,
            body=link_index_of_points,
            enabled=tuple(True for _ in link_index_of_points),
        )

        assert cp.point.shape[1] == 3, cp.point.shape[1]
        assert cp.point.shape[0] == len(cp.body), cp.point.shape[0]

        return cp


@jax_dataclasses.pytree_dataclass
class FrameParameters(JaxsimDataclass):
    """
    Class storing the frame parameters of a model.

    Attributes:
        name: A tuple of strings defining the frame names.
        body:
            A vector of integers representing, for each frame, the index of
            the body (link) to which it is rigidly attached to.
        transform: The transforms of the frames w.r.t. their parent link.

    Note:
        Contrarily to LinkParameters and JointParameters, this class is not meant
        to be created with vmap. This is because the `name` attribute must be `Static`.
    """

    name: Static[tuple[str, ...]] = dataclasses.field(default_factory=tuple)

    body: Static[tuple[int, ...]] = dataclasses.field(default_factory=tuple)

    transform: jtp.Array = dataclasses.field(default_factory=lambda: jnp.array([]))

    @staticmethod
    def build_from(model_description: ModelDescription) -> FrameParameters:
        """
        Build a FrameParameters object from a model description.

        Args:
            model_description: The model description to consider.

        Returns:
            The FrameParameters object.
        """

        if len(model_description.frames) == 0:
            return FrameParameters()

        # Extract the frame names.
        names = tuple(frame.name for frame in model_description.frames)

        # For each frame, extract the index of the link to which it is attached to.
        parent_link_index_of_frames = tuple(
            model_description.links_dict[frame.parent_name].index
            for frame in model_description.frames
        )

        # For each frame, extract the transform w.r.t. its parent link.
        transforms = jnp.atleast_3d(
            jnp.stack([frame.pose for frame in model_description.frames])
        )

        # Build the FrameParameters object.
        fp = FrameParameters(
            name=names,
            transform=transforms.astype(float),
            body=parent_link_index_of_frames,
        )

        assert fp.transform.shape[1:] == (4, 4), fp.transform.shape[1:]
        assert fp.transform.shape[0] == len(fp.body), fp.transform.shape[0]

        return fp


@dataclasses.dataclass(frozen=True)
class LinkParametrizableShape:
    """
    Enum-like class listing the supported shapes for HW parametrization.
    """

    Unsupported: ClassVar[int] = -1
    Box: ClassVar[int] = 0
    Cylinder: ClassVar[int] = 1
    Sphere: ClassVar[int] = 2
    Mesh: ClassVar[int] = 3


@jax_dataclasses.pytree_dataclass
class HwLinkMetadata(JaxsimDataclass):
    """
    Class storing the hardware parameters of a link.

    Attributes:
        link_shape: The shape of the link.
            0 = box, 1 = cylinder, 2 = sphere, 3 = mesh, -1 = unsupported.
        geometry: The dimensions of the link.
            box: [lx,ly,lz], cylinder: [r,l,0], sphere: [r,0,0], mesh: [lx,ly,lz] (bounding box).
        density: The density of the link.
        L_H_G: The homogeneous transformation matrix from the link frame to the CoM frame G.
        L_H_vis: The homogeneous transformation matrix from the link frame to the visual frame.
        L_H_pre_mask: The mask indicating the link's child joint indices.
        L_H_pre: The homogeneous transforms for child joints.
    """

    link_shape: jtp.Vector
    geometry: jtp.Vector
    density: jtp.Float
    L_H_G: jtp.Matrix
    L_H_vis: jtp.Matrix
    L_H_pre_mask: jtp.Vector
    L_H_pre: jtp.Matrix

    @classmethod
    def empty(cls) -> HwLinkMetadata:
        """Return hardware metadata representing the absence of links."""
        return cls(
            link_shape=jnp.array([], dtype=int),
            geometry=jnp.array([], dtype=float),
            density=jnp.array([], dtype=float),
            L_H_G=jnp.array([], dtype=float),
            L_H_vis=jnp.array([], dtype=float),
            L_H_pre_mask=jnp.array([], dtype=bool),
            L_H_pre=jnp.array([], dtype=float),
        )

    @staticmethod
    def compute_mass_and_inertia(
        hw_link_metadata: HwLinkMetadata,
    ) -> tuple[jtp.Float, jtp.Matrix]:
        """
        Compute the mass and inertia of a hardware link based on its metadata.

        This function calculates the mass and inertia tensor of a hardware link
        using its shape, dimensions, and density. The computation is performed
        by using shape-specific methods.

        Args:
            hw_link_metadata: Metadata describing the hardware link,
                including its shape, dimensions, and density.

        Returns:
            tuple: A tuple containing:
                - mass: The computed mass of the hardware link.
                - inertia: The computed inertia tensor of the hardware link.
        """

        def box(dims, density) -> tuple[jtp.Float, jtp.Matrix]:
            lx, ly, lz = dims

            mass = density * lx * ly * lz

            inertia = jnp.array(
                [
                    [mass * (ly**2 + lz**2) / 12, 0, 0],
                    [0, mass * (lx**2 + lz**2) / 12, 0],
                    [0, 0, mass * (lx**2 + ly**2) / 12],
                ]
            )
            return mass, inertia

        def cylinder(dims, density) -> tuple[jtp.Float, jtp.Matrix]:
            r, l, _ = dims

            mass = density * (jnp.pi * r**2 * l)

            inertia = jnp.array(
                [
                    [mass * (3 * r**2 + l**2) / 12, 0, 0],
                    [0, mass * (3 * r**2 + l**2) / 12, 0],
                    [0, 0, mass * (r**2) / 2],
                ]
            )

            return mass, inertia

        def sphere(dims, density) -> tuple[jtp.Float, jtp.Matrix]:
            r = dims[0]

            mass = density * (4 / 3 * jnp.pi * r**3)

            inertia = jnp.eye(3) * (2 / 5 * mass * r**2)

            return mass, inertia

        def compute_mass_inertia(shape_idx, dims, density):
            return jax.lax.switch(shape_idx, (box, cylinder, sphere), dims, density)

        mass, inertia = jax.vmap(compute_mass_inertia)(
            hw_link_metadata.link_shape,
            hw_link_metadata.geometry,
            hw_link_metadata.density,
        )

        return mass, inertia

    @staticmethod
    def _convert_scaling_to_3d_vector(
        link_shapes: jtp.Int, scaling_factors: jtp.Vector
    ) -> jtp.Vector:
        """
        Convert scaling factors for specific shape dimensions into a 3D scaling vector.

        Args:
            link_shapes: The link_shapes of the link (e.g., box, sphere, cylinder, mesh).
            scaling_factors: The scaling factors for the shape dimensions.

        Returns:
            A 3D scaling vector to apply to position vectors.

        Note:
            The scaling factors are applied as follows to generate the 3D scale vector:
            - Box: [lx, ly, lz]
            - Cylinder: [r, r, l]
            - Sphere: [r, r, r]
            - Mesh: [sx, sy, sz]
        """

        # Index mapping for each shape type (link_shapes x 3 dims)
        # Box: [lx, ly, lz] -> [0, 1, 2]
        # Cylinder: [r, r, l] -> [0, 0, 1]
        # Sphere: [r, r, r] -> [0, 0, 0]
        # Mesh: [sx, sy, sz] -> [0, 1, 2]
        shape_indices = jnp.array(
            [
                [0, 1, 2],  # Box
                [0, 0, 1],  # Cylinder
                [0, 0, 0],  # Sphere
                [0, 1, 2],  # Mesh
            ]
        )

        # For each link, get the index vector for its shape
        per_link_indices = shape_indices[link_shapes]

        # Gather dims per link according to per_link_indices
        return scaling_factors.dims[per_link_indices.squeeze()]

    @staticmethod
    def compute_contact_points(
        original_contact_params: jtp.Vector,
        link_shapes: jtp.Vector,
        original_com_positions: jtp.Vector,
        updated_com_positions: jtp.Vector,
        scaling_factors: ScalingFactors,
    ) -> jtp.Matrix:
        """
        Compute the new contact points based on the original contact parameters and
        the scaling factors.

        Args:
            original_contact_params: The original contact parameters.
            link_shapes: The shape types of the links (e.g., box, sphere, cylinder).
            original_com_positions: The original center of mass positions of the links.
            updated_com_positions: The updated center of mass positions of the links.
            scaling_factors: The scaling factors for the link dimensions.

        Returns:
            The new contact points positions in the parent link frame.
        """

        parent_link_indices = np.array(original_contact_params.body)

        # Translate the original contact point positions in the origin, so
        # that we can apply the scaling factors.
        L_p_Ci = (
            original_contact_params.point - original_com_positions[parent_link_indices]
        )

        # Extract the shape types of the parent links.
        parent_link_shapes = jnp.array(link_shapes[parent_link_indices])

        def sphere(parent_idx, L_p_C):
            r = scaling_factors.dims[parent_idx][0]
            return L_p_C * r

        def cylinder(parent_idx, L_p_C):
            # TODO: Cylinder collisions are not currently supported in JaxSim.
            return L_p_C

        def box(parent_idx, L_p_C):
            lx, ly, lz = scaling_factors.dims[parent_idx]
            return jnp.hstack(
                [
                    L_p_C[0] * lx,
                    L_p_C[1] * ly,
                    L_p_C[2] * lz,
                ]
            )

        def mesh(parent_idx, L_p_C):
            # Element-wise scaling like box
            sx, sy, sz = scaling_factors.dims[parent_idx]
            return jnp.hstack(
                [
                    L_p_C[0] * sx,
                    L_p_C[1] * sy,
                    L_p_C[2] * sz,
                ]
            )

        new_positions = jax.vmap(
            lambda shape_idx, parent_idx, L_p_C: jax.lax.switch(
                shape_idx, (box, cylinder, sphere, mesh), parent_idx, L_p_C
            )
        )(
            parent_link_shapes,
            parent_link_indices,
            L_p_Ci,
        )

        return new_positions + updated_com_positions[parent_link_indices]

    @staticmethod
    def compute_inertia_link(I_com, L_H_G) -> jtp.Matrix:
        """
        Compute the inertia tensor of the link based on its shape and mass.
        """

        L_R_G = L_H_G[:3, :3]
        return L_R_G @ I_com @ L_R_G.T

    @staticmethod
    def apply_scaling(
        has_joints: bool,
        hw_metadata: HwLinkMetadata,
        scaling_factors: ScalingFactors,
    ) -> HwLinkMetadata:
        """
        Apply scaling to the hardware parameters and return a new HwLinkMetadata object.

        Args:
            has_joints: A boolean indicating if the model has joints.
            hw_metadata: the original HwLinkMetadata object.
            scaling_factors: the scaling factors to apply.
            has_joints: whether the model has at least one joint.

        Returns:
            A new HwLinkMetadata object with updated parameters.
        """

        scale_vector = HwLinkMetadata._convert_scaling_to_3d_vector(
            hw_metadata.link_shape, scaling_factors
        )

        # =================================
        # Update the kinematics of the link
        # =================================

        # Get the nominal transforms
        L_H_G = hw_metadata.L_H_G
        L_H_vis = hw_metadata.L_H_vis
        L_H_pre_array = hw_metadata.L_H_pre
        L_H_pre_mask = hw_metadata.L_H_pre_mask

        # Express the transforms in the G frame
        G_H_L = jaxsim.math.Transform.inverse(L_H_G)
        G_H_vis = G_H_L @ L_H_vis

        G_H_pre_array = (
            jax.vmap(lambda L_H_pre: G_H_L @ L_H_pre)(L_H_pre_array)
            if has_joints
            else L_H_pre_array
        )

        # Apply the scaling to the position vectors
        G_H̅_vis = G_H_vis.at[:3, 3].set(scale_vector * G_H_vis[:3, 3])

        # Apply scaling to the position vectors in G_H_pre_array based on the mask
        G_H̅_pre_array = (
            G_H_pre_array.at[:, :3, 3].set(
                jnp.where(
                    L_H_pre_mask[:, None],
                    scale_vector[None, :] * G_H_pre_array[:, :3, 3],
                    G_H_pre_array[:, :3, 3],
                )
            )
            if has_joints
            else G_H_pre_array
        )

        # Get back to the link frame
        L_H̅_G = L_H_G.at[:3, 3].set(scale_vector * L_H_G[:3, 3])
        L_H̅_vis = L_H̅_G @ G_H̅_vis
        L_H̅_pre_array = (
            jax.vmap(lambda G_H̅_pre: L_H̅_G @ G_H̅_pre)(G_H̅_pre_array)
            if has_joints
            else G_H̅_pre_array
        )

        # ===========================
        # Update the shape parameters
        # ===========================

        updated_geoms = hw_metadata.geometry * scaling_factors.dims

        # =============================
        # Scale the density of the link
        # =============================

        updated_density = hw_metadata.density * scaling_factors.density

        # =============================
        # Return updated HwLinkMetadata
        # =============================

        return hw_metadata.replace(
            geometry=updated_geoms,
            density=updated_density,
            L_H_G=L_H̅_G,
            L_H_vis=L_H̅_vis,
            L_H_pre=L_H̅_pre_array,
        )


@jax_dataclasses.pytree_dataclass
class ScalingFactors(JaxsimDataclass):
    """
    Class storing scaling factors for hardware parameters.

    Attributes:
        dims: Scaling factors for shape dimensions.
        density: Scaling factor for density.
    """

    dims: jtp.Vector
    density: jtp.Float


@dataclasses.dataclass(frozen=True)
class ConstraintType:
    """
    Enumeration of all supported constraint types.
    """

    Weld: ClassVar[int] = 0
    # TODO: handle Connect constraint
    # Connect: ClassVar[int] = 1


@jax_dataclasses.pytree_dataclass
class ConstraintMap(JaxsimDataclass):
    """
    Class storing the kinematic constraints of a model.
    """

    frame_idxs_1: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    frame_idxs_2: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    constraint_types: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    K_P: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=float)
    )
    K_D: jtp.Float = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=float)
    )
    # Precomputed parent link indices for each constraint pair
    parent_link_idxs_1: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )
    parent_link_idxs_2: jtp.Int = dataclasses.field(
        default_factory=lambda: jnp.array([], dtype=int)
    )

    def add_constraint(
        self,
        model: jaxsim.api.model.JaxSimModel,
        frame_idx_1: int,
        frame_idx_2: int,
        constraint_type: int,
        K_P: float | None = None,
        K_D: float | None = None,
    ) -> ConstraintMap:
        """
        Add a constraint to the constraint map.

        Args:
            model: The model for which the constraints are added.
            frame_idx_1: The index of the first frame.
            frame_idx_2: The index of the second frame.
            constraint_type: The type of constraint.
            K_P: The proportional gain for Baumgarte stabilization (default: 1000).
            K_D: The derivative gain for Baumgarte stabilization (default: 2 * sqrt(K_P)).

        Returns:
            A new ConstraintMap instance with the added constraint.

        Note:
            Since this method returns a new instance of ConstraintMap with the new constraint,
            it will trigger recompilations in JIT-compiled functions.
        """

        # Set default values for Baumgarte coefficients if not provided
        if K_P is None:
            K_P = jnp.array([1000.0])
        if K_D is None:
            K_D = 2 * jnp.sqrt(K_P)

        # Create new arrays with the input elements appended
        new_frame_idxs_1 = jnp.append(self.frame_idxs_1, frame_idx_1)
        new_frame_idxs_2 = jnp.append(self.frame_idxs_2, frame_idx_2)
        new_constraint_types = jnp.append(self.constraint_types, constraint_type)
        new_K_P = jnp.append(self.K_P, K_P)
        new_K_D = jnp.append(self.K_D, K_D)

        # Compute parent link indices (now always available since model is required)
        parent_link_idx_1 = jaxsim.api.frame.idx_of_parent_link(
            model, frame_index=frame_idx_1
        )
        parent_link_idx_2 = jaxsim.api.frame.idx_of_parent_link(
            model, frame_index=frame_idx_2
        )
        new_parent_link_idxs_1 = jnp.append(self.parent_link_idxs_1, parent_link_idx_1)
        new_parent_link_idxs_2 = jnp.append(self.parent_link_idxs_2, parent_link_idx_2)

        # Return a new ConstraintMap object with updated attributes
        return ConstraintMap(
            frame_idxs_1=new_frame_idxs_1,
            frame_idxs_2=new_frame_idxs_2,
            constraint_types=new_constraint_types,
            K_P=new_K_P,
            K_D=new_K_D,
            parent_link_idxs_1=new_parent_link_idxs_1,
            parent_link_idxs_2=new_parent_link_idxs_2,
        )
