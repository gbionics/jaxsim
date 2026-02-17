from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax_dataclasses

import jaxsim.api as js
import jaxsim.typing as jtp
from jaxsim import exceptions
from jaxsim.utils.tracing import not_tracing

from .common import VelRepr

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxSimModelReferences(js.common.ModelDataWithVelocityRepresentation):
    """
    Class containing the references for a `JaxSimModel` object.

    Attributes:
        _link_forces: The link 6D forces in inertial-fixed representation.
        _joint_force_references: The joint force references.
    """

    _link_forces: jtp.Matrix
    _joint_force_references: jtp.Vector

    @staticmethod
    def zero(
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData | None = None,
        velocity_representation: VelRepr = VelRepr.Inertial,
    ) -> JaxSimModelReferences:
        """
        Create a `JaxSimModelReferences` object with zero references.

        Args:
            model: The model for which to create the zero references.
            data:
                The data of the model, only needed if the velocity representation is
                not inertial-fixed.
            velocity_representation: The velocity representation to use.

        Returns:
            A `JaxSimModelReferences` object with zero state.
        """

        return JaxSimModelReferences.build(
            model=model, data=data, velocity_representation=velocity_representation
        )

    @staticmethod
    def build(
        model: js.model.JaxSimModel,
        joint_force_references: jtp.VectorLike | None = None,
        link_forces: jtp.MatrixLike | None = None,
        data: js.data.JaxSimModelData | None = None,
        velocity_representation: VelRepr | None = None,
    ) -> JaxSimModelReferences:
        """
        Create a `JaxSimModelReferences` object with the given references.

        Args:
            model: The model for which to create the state.
            joint_force_references: The joint force references.
            link_forces: The link 6D forces in the desired representation.
            data:
                The data of the model, only needed if the velocity representation is
                not inertial-fixed.
            velocity_representation: The velocity representation to use.

        Returns:
            A `JaxSimModelReferences` object with the given references.
        """

        # Create or adjust joint force references.
        joint_force_references = jnp.atleast_1d(
            jnp.array(joint_force_references, dtype=float).squeeze()
            if joint_force_references is not None
            else jnp.zeros(model.dofs())
        ).astype(float)

        # Create or adjust link forces.
        f_L = jnp.atleast_2d(
            jnp.array(link_forces, dtype=float).squeeze()
            if link_forces is not None
            else jnp.zeros((model.number_of_links(), 6))
        ).astype(float)

        # Select the velocity representation.
        velocity_representation = (
            velocity_representation
            if velocity_representation is not None
            else getattr(data, "velocity_representation", VelRepr.Inertial)
        )

        # Create a zero references object.
        references = JaxSimModelReferences(
            _link_forces=f_L,
            _joint_force_references=joint_force_references,
            velocity_representation=velocity_representation,
        )

        # If the velocity representation is inertial-fixed, we can return
        # the references directly, as we store the link forces in this frame.
        if velocity_representation is VelRepr.Inertial:
            return references

        # Store the joint force references.
        references = references.set_joint_force_references(
            forces=joint_force_references,
            model=model,
            joint_names=model.joint_names(),
        )

        # Apply the link forces.
        references = references.apply_link_forces(
            forces=f_L,
            model=model,
            data=data,
            link_names=model.link_names(),
            additive=False,
        )

        return references

    def valid(self, model: js.model.JaxSimModel | None = None) -> bool:
        """
        Check if the current references are valid for the given model.

        Args:
            model: The model to check against.

        Returns:
            `True` if the current references are valid for the given model,
            `False` otherwise.
        """

        if model is None:
            return True

        shape = self._joint_force_references.shape
        expected_shape = (model.dofs(),)

        if shape != expected_shape:
            return False

        shape = self._link_forces.shape
        expected_shape = (model.number_of_links(), 6)

        if shape != expected_shape:
            return False

        return True

    # ==================
    # Extract quantities
    # ==================
    @js.common.named_scope
    @functools.partial(jax.jit, static_argnames=["link_names", "output_representation"])
    def link_forces(
        self,
        model: js.model.JaxSimModel | None = None,
        data: js.data.JaxSimModelData | None = None,
        link_names: tuple[str, ...] | None = None,
        output_representation: VelRepr | None = None,
    ) -> jtp.Matrix:
        """
        Return the link forces expressed in the specified representation.

        Args:
            model: The model to consider.
            data: The data of the considered model.
            link_names: The names of the links corresponding to the forces.
            output_representation:
                The desired output representation. If None, uses self.velocity_representation.

        Returns:
            If no model and no link names are provided, the link forces as a
            `(n_links,6)` matrix corresponding to the default link serialization
            of the original model used to build the actuation object.
            If a model is provided and no link names are provided, the link forces
            as a `(n_links,6)` matrix corresponding to the serialization of the
            provided model.
            If both a model and link names are provided, the link forces as a
            `(len(link_names),6)` matrix corresponding to the serialization of
            the passed link names vector.

        Note:
            The returned link forces are those passed as user inputs when integrating
            the dynamics of the model. They are summed with other forces related
            e.g. to the contact model and other kinematic constraints.
        """

        W_f_L = self._link_forces

        # Determine the output representation
        output_repr = (
            output_representation
            if output_representation is not None
            else self.velocity_representation
        )

        # Return all link forces in the desired representation using the implicit
        # serialization.
        if model is None:
            if output_repr is not VelRepr.Inertial:
                msg = "Missing model to use a representation different from {}"
                raise ValueError(msg.format(VelRepr.Inertial.name))

            if link_names is not None:
                raise ValueError("Link names cannot be provided without a model")

            return W_f_L

        # If we have the model, we can extract the link names, if not provided.
        link_idxs = (
            js.link.names_to_idxs(link_names=link_names, model=model)
            if link_names is not None
            else jnp.arange(model.number_of_links())
        )

        # In inertial-fixed representation, we already have the link forces.
        if output_repr is VelRepr.Inertial:
            return W_f_L[link_idxs, :]

        if data is None:
            msg = "Missing model data to use a representation different from {}"
            raise ValueError(msg.format(VelRepr.Inertial.name))

        if not_tracing(self._link_forces) and not data.valid(model=model):
            raise ValueError("The provided data is not valid for the model")

        # Helper function to convert a single 6D force to the desired representation
        # considering as body the link (i.e. L_f_L and LW_f_L).
        def convert(W_f_L: jtp.MatrixLike, W_H_L: jtp.ArrayLike) -> jtp.Matrix:

            return jax.vmap(
                lambda W_f_L, W_H_L: JaxSimModelReferences.inertial_to_other_representation(
                    array=W_f_L,
                    other_representation=output_repr,
                    transform=W_H_L,
                    is_force=True,
                )
            )(W_f_L, W_H_L)

        # The f_L output is either L_f_L or LW_f_L, depending on the representation.
        W_H_L = data._link_transforms
        f_L = convert(W_f_L=W_f_L[link_idxs, :], W_H_L=W_H_L[link_idxs, :, :])

        return f_L

    def joint_force_references(
        self,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> jtp.Vector:
        """
        Return the joint force references.

        Args:
            model: The model to consider.
            joint_names: The names of the joints corresponding to the forces.

        Returns:
            If no model and no joint names are provided, the joint forces as a
            `(DoFs,)` vector corresponding to the default joint serialization
            of the original model used to build the actuation object.
            If a model is provided and no joint names are provided, the joint forces
            as a `(DoFs,)` vector corresponding to the serialization of the
            provided model.
            If both a model and joint names are provided, the joint forces as a
            `(len(joint_names),)` vector corresponding to the serialization of
            the passed joint names vector.

        Note:
            The returned joint forces are those passed as user inputs when integrating
            the dynamics of the model. They are summed with other joint forces related
            e.g. to the enforcement of other kinematic constraints. Keep also in mind
            that the presence of joint friction and other similar effects can make the
            actual joint forces different from the references.
        """

        if model is None:
            if joint_names is not None:
                raise ValueError("Joint names cannot be provided without a model")

            return self._joint_force_references

        if not_tracing(self._joint_force_references) and not self.valid(model=model):
            msg = "The actuation object is not compatible with the provided model"
            raise ValueError(msg)

        joint_idxs = (
            js.joint.names_to_idxs(joint_names=joint_names, model=model)
            if joint_names is not None
            else jnp.arange(model.number_of_joints())
        )

        return jnp.atleast_1d(
            self._joint_force_references[joint_idxs].squeeze()
        ).astype(float)

    # ================
    # Store quantities
    # ================
    @js.common.named_scope
    @functools.partial(jax.jit, static_argnames=["joint_names"])
    def set_joint_force_references(
        self,
        forces: jtp.VectorLike,
        model: js.model.JaxSimModel | None = None,
        joint_names: tuple[str, ...] | None = None,
    ) -> Self:
        """
        Set the joint force references.

        Args:
            forces: The joint force references.
            model:
                The model to consider, only needed if a joint serialization different
                from the implicit one is used.
            joint_names: The names of the joints corresponding to the forces.

        Returns:
            A new `JaxSimModelReferences` object with the given joint force references.
        """

        forces = jnp.atleast_1d(jnp.array(forces, dtype=float).squeeze())

        def replace(forces: jtp.Vector) -> JaxSimModelReferences:
            return self.replace(
                validate=True,
                _joint_force_references=jnp.atleast_1d(forces.squeeze()).astype(float),
            )

        if model is None:
            return replace(forces=forces)

        if not_tracing(forces) and not self.valid(model=model):
            msg = "The references object is not compatible with the provided model"
            raise ValueError(msg)

        joint_idxs = (
            js.joint.names_to_idxs(joint_names=joint_names, model=model)
            if joint_names is not None
            else jnp.arange(model.number_of_joints())
        )

        return replace(forces=self._joint_force_references.at[joint_idxs].set(forces))

    @js.common.named_scope
    @functools.partial(
        jax.jit, static_argnames=["link_names", "additive", "input_representation"]
    )
    def apply_link_forces(
        self,
        forces: jtp.MatrixLike,
        model: js.model.JaxSimModel | None = None,
        data: js.data.JaxSimModelData | None = None,
        link_names: tuple[str, ...] | str | None = None,
        additive: bool = False,
        input_representation: VelRepr | None = None,
    ) -> Self:
        """
        Apply the link forces.

        Args:
            forces: The link 6D forces in the specified input representation.
            model:
                The model to consider, only needed if a link serialization different
                from the implicit one is used.
            data:
                The data of the considered model, only needed if the input
                representation is not inertial-fixed.
            link_names: The names of the links corresponding to the forces.
            additive:
                Whether to add the forces to the existing ones instead of replacing them.
            input_representation:
                The representation of the input forces. If None, uses self.velocity_representation.

        Returns:
            A new `JaxSimModelReferences` object with the given link forces.

        Note:
            The link forces must be expressed in the specified input representation.
            They are always converted and stored internally in inertial-fixed representation.
        """

        f_L = jnp.atleast_2d(forces).astype(float)

        # Determine the input representation
        input_repr = (
            input_representation
            if input_representation is not None
            else self.velocity_representation
        )

        # Helper function to replace the link forces.
        def replace(forces: jtp.MatrixLike) -> JaxSimModelReferences:
            return self.replace(
                validate=True,
                _link_forces=jnp.atleast_2d(forces.squeeze()).astype(float),
            )

        # In this case, we allow only to set the inertial 6D forces to all links
        # using the implicit link serialization.
        if model is None:
            if input_repr is not VelRepr.Inertial:
                msg = "Missing model to use a representation different from {}"
                raise ValueError(msg.format(VelRepr.Inertial.name))

            if link_names is not None:
                raise ValueError("Link names cannot be provided without a model")

            W_f_L = f_L

            W_f0_L = jnp.zeros_like(W_f_L) if not additive else self._link_forces

            return replace(forces=W_f0_L + W_f_L)

        if link_names is not None and len(link_names) != f_L.shape[0]:
            msg = "The number of link names ({}) must match the number of forces ({})"
            raise ValueError(msg.format(len(link_names), f_L.shape[0]))

        # Extract the link indices.
        link_idxs = (
            js.link.names_to_idxs(link_names=link_names, model=model)
            if link_names is not None
            else jnp.arange(model.number_of_links())
        )

        # Compute the bias depending on whether we either set or add the link forces.
        W_f0_L = (
            jnp.zeros_like(f_L) if not additive else self._link_forces[link_idxs, :]
        )

        # If inertial-fixed representation, we can directly store the link forces.
        if input_repr is VelRepr.Inertial:
            W_f_L = f_L
            return replace(
                forces=self._link_forces.at[link_idxs, :].set(W_f0_L + W_f_L)
            )

        if data is None:
            msg = "Missing model data to use a representation different from {}"
            raise ValueError(msg.format(VelRepr.Inertial.name))

        if not_tracing(forces) and not data.valid(model=model):
            raise ValueError("The provided data is not valid for the model")

        W_H_L = data._link_transforms

        # Convert a single 6D force to the inertial representation
        # considering as body the link (i.e. L_f_L and LW_f_L).
        # The f_L input is either L_f_L or LW_f_L, depending on the input representation.
        W_f_L = JaxSimModelReferences.other_representation_to_inertial(
            array=f_L,
            other_representation=input_repr,
            transform=W_H_L[link_idxs] if model.number_of_links() > 1 else W_H_L,
            is_force=True,
        )

        return replace(forces=self._link_forces.at[link_idxs, :].set(W_f0_L + W_f_L))

    def apply_frame_forces(
        self,
        forces: jtp.MatrixLike,
        model: js.model.JaxSimModel,
        data: js.data.JaxSimModelData,
        frame_names: tuple[str, ...] | str | None = None,
        additive: bool = False,
    ) -> Self:
        """
        Apply the frame forces.

        Args:
            forces: The frame 6D forces in the active representation.
            model:
                The model to consider, only needed if a frame serialization different
                from the implicit one is used.
            data:
                The data of the considered model, only needed if the velocity
                representation is not inertial-fixed.
            frame_names: The names of the frames corresponding to the forces.
            additive:
                Whether to add the forces to the existing ones instead of replacing them.

        Returns:
            A new `JaxSimModelReferences` object with the given frame forces.

        Note:
            The frame forces must be expressed in the active representation.
            Then, we always convert and store forces in inertial-fixed representation.
        """

        f_F = jnp.atleast_2d(forces).astype(float)

        if len(frame_names) != f_F.shape[0]:
            msg = "The number of frame names ({}) must match the number of forces ({})"
            raise ValueError(msg.format(len(frame_names), f_F.shape[0]))

        # Extract the frame indices.
        frame_idxs = (
            js.frame.names_to_idxs(frame_names=frame_names, model=model)
            if frame_names is not None
            else jnp.arange(len(model.frame_names()))
        )

        parent_link_idxs = jnp.array(model.kin_dyn_parameters.frame_parameters.body)[
            frame_idxs - model.number_of_links()
        ]

        exceptions.raise_value_error_if(
            condition=not data.valid(model=model),
            msg="The provided data is not valid for the model",
        )
        W_H_Fi = jax.vmap(
            lambda frame_idx: js.frame.transform(
                model=model, data=data, frame_index=frame_idx
            )
        )(frame_idxs)

        # Helper function to convert a single 6D force to the inertial representation
        # considering as body the frame (i.e. L_f_F and LW_f_F).
        def to_inertial(f_F: jtp.MatrixLike, W_H_F: jtp.MatrixLike) -> jtp.Matrix:
            return JaxSimModelReferences.other_representation_to_inertial(
                array=f_F,
                other_representation=self.velocity_representation,
                transform=W_H_F,
                is_force=True,
            )

        match self.velocity_representation:
            case VelRepr.Inertial:
                W_f_F = f_F

            case VelRepr.Body | VelRepr.Mixed:
                W_f_F = jax.vmap(to_inertial)(f_F, W_H_Fi)

            case _:
                raise ValueError("Invalid velocity representation.")

        # Sum the forces on the parent links.
        mask = parent_link_idxs[:, jnp.newaxis] == jnp.arange(model.number_of_links())
        W_f_L = mask.T @ W_f_F

        # Apply the forces, specifying that they are already in Inertial representation.
        # The output will maintain the current velocity representation.
        return self.apply_link_forces(
            model=model,
            data=data,
            forces=W_f_L,
            additive=additive,
            input_representation=VelRepr.Inertial,
        )
