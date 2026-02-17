from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence

try:
    from typing import Self, override
except ImportError:
    from typing_extensions import override, Self

import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform
import jax_dataclasses

import jaxsim.api as js
import jaxsim.math
import jaxsim.rbda
import jaxsim.typing as jtp

from . import common
from .common import VelRepr


@jax_dataclasses.pytree_dataclass
class JaxSimModelData(common.ModelDataWithVelocityRepresentation):
    """
    Class storing the state of the physics model dynamics.

    Attributes:
        joint_positions: The vector of joint positions.
        joint_velocities: The vector of joint velocities.
        base_position: The 3D position of the base link.
        base_quaternion: The quaternion defining the orientation of the base link.
        base_linear_velocity:
            The linear velocity of the base link in inertial-fixed representation.
        base_angular_velocity:
            The angular velocity of the base link in inertial-fixed representation.
        base_transform: The base transform.
        joint_transforms: The joint transforms.
        link_transforms: The link transforms.
        link_velocities: The link velocities in inertial-fixed representation.
    """

    # Joint state
    _joint_positions: jtp.Vector
    _joint_velocities: jtp.Vector

    # Base state
    _base_quaternion: jtp.Vector
    _base_linear_velocity: jtp.Vector
    _base_angular_velocity: jtp.Vector
    _base_position: jtp.Vector

    # Cached computations.
    _base_transform: jtp.Matrix = dataclasses.field(repr=False, default=None)
    _joint_transforms: jtp.Matrix = dataclasses.field(repr=False, default=None)
    _link_transforms: jtp.Matrix = dataclasses.field(repr=False, default=None)
    _link_velocities: jtp.Matrix = dataclasses.field(repr=False, default=None)

    # Extended state for soft and rigid contact models.
    contact_state: dict[str, jtp.Array] = dataclasses.field(default_factory=dict)

    @staticmethod
    def build(
        model: js.model.JaxSimModel,
        base_position: jtp.VectorLike | None = None,
        base_quaternion: jtp.VectorLike | None = None,
        joint_positions: jtp.VectorLike | None = None,
        base_linear_velocity: jtp.VectorLike | None = None,
        base_angular_velocity: jtp.VectorLike | None = None,
        joint_velocities: jtp.VectorLike | None = None,
        contact_state: dict[str, jtp.Array] | None = None,
        velocity_representation: VelRepr = VelRepr.Mixed,
    ) -> JaxSimModelData:
        """
        Create a `JaxSimModelData` object with the given state.

        Args:
            model: The model for which to create the state.
            base_position: The base position.
            base_quaternion: The base orientation as a quaternion.
            joint_positions: The joint positions.
            base_linear_velocity:
                The base linear velocity in the selected representation.
            base_angular_velocity:
                The base angular velocity in the selected representation.
            joint_velocities: The joint velocities.
            velocity_representation: The velocity representation to use. It defaults to mixed if not provided.
            contact_state: The optional contact state.

        Returns:
            A `JaxSimModelData` initialized with the given state.
        """

        base_position = jnp.array(
            base_position if base_position is not None else jnp.zeros(3),
            dtype=float,
        ).squeeze()

        base_quaternion = jnp.array(
            (
                base_quaternion
                if base_quaternion is not None
                else jnp.array([1.0, 0, 0, 0])
            ),
            dtype=float,
        ).squeeze()

        base_linear_velocity = jnp.array(
            base_linear_velocity if base_linear_velocity is not None else jnp.zeros(3),
            dtype=float,
        ).squeeze()

        base_angular_velocity = jnp.array(
            (
                base_angular_velocity
                if base_angular_velocity is not None
                else jnp.zeros(3)
            ),
            dtype=float,
        ).squeeze()

        joint_positions = jnp.atleast_1d(
            jnp.array(
                (
                    joint_positions
                    if joint_positions is not None
                    else jnp.zeros(model.dofs())
                ),
                dtype=float,
            ).squeeze()
        )

        joint_velocities = jnp.atleast_1d(
            jnp.array(
                (
                    joint_velocities
                    if joint_velocities is not None
                    else jnp.zeros(model.dofs())
                ),
                dtype=float,
            ).squeeze()
        )

        W_H_B = jaxsim.math.Transform.from_quaternion_and_translation(
            translation=base_position, quaternion=base_quaternion
        )

        W_v_WB = JaxSimModelData.other_representation_to_inertial(
            array=jnp.hstack([base_linear_velocity, base_angular_velocity]),
            other_representation=velocity_representation,
            transform=W_H_B,
            is_force=False,
        ).astype(float)

        joint_transforms = model.kin_dyn_parameters.joint_transforms(
            joint_positions=joint_positions, base_transform=W_H_B
        )

        link_transforms, link_velocities_inertial = (
            jaxsim.rbda.forward_kinematics_model(
                model=model,
                base_position=base_position,
                base_quaternion=base_quaternion,
                joint_positions=joint_positions,
                base_linear_velocity_inertial=W_v_WB[0:3],
                base_angular_velocity_inertial=W_v_WB[3:6],
                joint_velocities=joint_velocities,
            )
        )

        contact_state = contact_state or {}

        if isinstance(model.contact_model, jaxsim.rbda.contacts.SoftContacts):
            contact_state["tangential_deformation"] = contact_state.get(
                "tangential_deformation",
                jnp.zeros_like(model.kin_dyn_parameters.contact_parameters.point),
            )

        model_data = JaxSimModelData(
            velocity_representation=velocity_representation,
            _base_quaternion=base_quaternion,
            _base_position=base_position,
            _joint_positions=joint_positions,
            _base_linear_velocity=W_v_WB[0:3],
            _base_angular_velocity=W_v_WB[3:6],
            _joint_velocities=joint_velocities,
            _base_transform=W_H_B,
            _joint_transforms=joint_transforms,
            _link_transforms=link_transforms,
            _link_velocities=link_velocities_inertial,
            contact_state=contact_state,
        )

        if not model_data.valid(model=model):
            raise ValueError(
                "The built state is not compatible with the model.", model_data
            )

        return model_data

    @staticmethod
    def zero(
        model: js.model.JaxSimModel,
        velocity_representation: VelRepr = VelRepr.Mixed,
    ) -> JaxSimModelData:
        """
        Create a `JaxSimModelData` object with zero state.

        Args:
            model: The model for which to create the state.
            velocity_representation: The velocity representation to use. It defaults to mixed if not provided.

        Returns:
            A `JaxSimModelData` initialized with zero state.
        """
        return JaxSimModelData.build(
            model=model, velocity_representation=velocity_representation
        )

    # ==================
    # Extract quantities
    # ==================

    @property
    def joint_positions(self) -> jtp.Vector:
        """
        Get the joint positions.

        Returns:
            The joint positions.
        """
        return self._joint_positions

    @property
    def joint_velocities(self) -> jtp.Vector:
        """
        Get the joint velocities.

        Returns:
            The joint velocities.
        """
        return self._joint_velocities

    @property
    def base_quaternion(self) -> jtp.Vector:
        """
        Get the base quaternion.

        Returns:
            The base quaternion.
        """
        return self._base_quaternion

    @property
    def base_position(self) -> jtp.Vector:
        """
        Get the base position.

        Returns:
            The base position.
        """
        return self._base_position

    @property
    def base_orientation(self) -> jtp.Matrix:
        """
        Get the base orientation.

        Returns:
            The base orientation.
        """

        # Extract the base quaternion.
        W_Q_B = self.base_quaternion

        # Always normalize the quaternion to avoid numerical issues.
        # If the active scheme does not integrate the quaternion on its manifold,
        # we introduce a Baumgarte stabilization to let the quaternion converge to
        # a unit quaternion. In this case, it is not guaranteed that the quaternion
        # stored in the state is a unit quaternion.
        norm = jaxsim.math.safe_norm(W_Q_B, axis=-1, keepdims=True)
        W_Q_B = W_Q_B / (norm + jnp.finfo(float).eps * (norm == 0))
        return W_Q_B

    def base_velocity(self, output_representation: VelRepr | None = None) -> jtp.Vector:
        """
        Get the base 6D velocity.

        Args:
            output_representation: The desired output representation.
                If None, uses self.velocity_representation.

        Returns:
            The base 6D velocity in the specified representation.
        """

        output_repr = (
            output_representation
            if output_representation is not None
            else self.velocity_representation
        )

        W_v_WB = jnp.concatenate(
            [self._base_linear_velocity, self._base_angular_velocity], axis=-1
        )

        W_H_B = self._base_transform

        return (
            JaxSimModelData.inertial_to_other_representation(
                array=W_v_WB,
                other_representation=output_repr,
                transform=W_H_B,
                is_force=False,
            )
            .squeeze()
            .astype(float)
        )

    @property
    def generalized_position(self) -> tuple[jtp.Matrix, jtp.Vector]:
        r"""
        Get the generalized position
        :math:`\mathbf{q} = ({}^W \mathbf{H}_B, \mathbf{s}) \in \text{SO}(3) \times \mathbb{R}^n`.

        Returns:
            A tuple containing the base transform and the joint positions.
        """

        return self._base_transform, self.joint_positions

    def generalized_velocity(
        self, output_representation: VelRepr | None = None
    ) -> jtp.Vector:
        r"""
        Get the generalized velocity.

        :math:`\boldsymbol{\nu} = (\boldsymbol{v}_{W,B};\, \boldsymbol{\omega}_{W,B};\, \mathbf{s}) \in \mathbb{R}^{6+n}`

        Args:
            output_representation: The desired output representation.
                If None, uses self.velocity_representation.

        Returns:
            The generalized velocity in the specified representation.
        """

        return (
            jnp.hstack(
                [
                    self.base_velocity(output_representation=output_representation),
                    self.joint_velocities,
                ]
            )
            .squeeze()
            .astype(float)
        )

    @property
    def base_transform(self) -> jtp.Matrix:
        """
        Get the base transform.

        Returns:
            The base transform.
        """
        return self._base_transform

    # ================
    # Store quantities
    # ================

    @js.common.named_scope
    @jax.jit
    def reset_base_quaternion(
        self, model: js.model.JaxSimModel, base_quaternion: jtp.VectorLike
    ) -> Self:
        """
        Reset the base quaternion.

        Args:
            model: The JaxSim model to use.
            base_quaternion: The base orientation as a quaternion.

        Returns:
            The updated `JaxSimModelData` object.
        """

        W_Q_B = jnp.array(base_quaternion, dtype=float)

        norm = jaxsim.math.safe_norm(W_Q_B, axis=-1)
        W_Q_B = W_Q_B / (norm + jnp.finfo(float).eps * (norm == 0))

        return self.replace(model=model, base_quaternion=W_Q_B)

    @js.common.named_scope
    @jax.jit
    def reset_base_pose(
        self, model: js.model.JaxSimModel, base_pose: jtp.MatrixLike
    ) -> Self:
        """
        Reset the base pose.

        Args:
            model: The JaxSim model to use.
            base_pose: The base pose as an SE(3) matrix.

        Returns:
            The updated `JaxSimModelData` object.
        """

        base_pose = jnp.array(base_pose)
        W_p_B = base_pose[0:3, 3]
        W_Q_B = jaxsim.math.Quaternion.from_dcm(dcm=base_pose[0:3, 0:3])
        return self.replace(
            model=model,
            base_position=W_p_B,
            base_quaternion=W_Q_B,
        )

    @override
    def replace(
        self,
        model: js.model.JaxSimModel,
        joint_positions: jtp.Vector | None = None,
        joint_velocities: jtp.Vector | None = None,
        base_quaternion: jtp.Vector | None = None,
        base_linear_velocity: jtp.Vector | None = None,
        base_angular_velocity: jtp.Vector | None = None,
        base_position: jtp.Vector | None = None,
        *,
        input_representation: VelRepr | None = None,
        contact_state: dict[str, jtp.Array] | None = None,
        validate: bool = False,
    ) -> Self:
        """
        Replace the attributes of the `JaxSimModelData` object.
        """

        input_representation = (
            input_representation
            if input_representation is not None
            else self.velocity_representation
        )

        if joint_positions is None:
            joint_positions = self.joint_positions
        if joint_velocities is None:
            joint_velocities = self.joint_velocities
        if base_quaternion is None:
            base_quaternion = self.base_quaternion
        if base_position is None:
            base_position = self.base_position
        if contact_state is None:
            contact_state = self.contact_state

        # Normalize the quaternion to avoid numerical issues.
        base_quaternion_norm = jaxsim.math.safe_norm(
            base_quaternion, axis=-1, keepdims=True
        )
        base_quaternion = base_quaternion / jnp.where(
            base_quaternion_norm == 0, 1.0, base_quaternion_norm
        )

        joint_positions = jnp.atleast_2d(joint_positions.squeeze()).astype(float)
        joint_velocities = jnp.atleast_2d(joint_velocities.squeeze()).astype(float)
        base_quaternion = jnp.atleast_2d(base_quaternion.squeeze()).astype(float)
        base_position = jnp.atleast_2d(base_position.squeeze()).astype(float)

        base_transform = jaxsim.math.Transform.from_quaternion_and_translation(
            translation=base_position, quaternion=base_quaternion
        ).reshape((-1, 4, 4))

        joint_transforms = jax.vmap(model.kin_dyn_parameters.joint_transforms)(
            joint_positions=joint_positions,
            base_transform=base_transform,
        )

        if base_linear_velocity is None and base_angular_velocity is None:
            base_linear_velocity_inertial = self._base_linear_velocity
            base_angular_velocity_inertial = self._base_angular_velocity
        else:
            if base_linear_velocity is None:
                base_linear_velocity = self.base_velocity[:3]
            if base_angular_velocity is None:
                base_angular_velocity = self.base_velocity[3:]

            base_linear_velocity = jnp.atleast_1d(base_linear_velocity.squeeze())
            base_angular_velocity = jnp.atleast_1d(base_angular_velocity.squeeze())

            W_v_WB = JaxSimModelData.other_representation_to_inertial(
                array=jnp.hstack([base_linear_velocity, base_angular_velocity]),
                other_representation=input_representation,
                transform=base_transform,
                is_force=False,
            ).astype(float)

            base_linear_velocity_inertial, base_angular_velocity_inertial = (
                W_v_WB[..., :3],
                W_v_WB[..., 3:],
            )

        link_transforms, link_velocities = jax.vmap(
            jaxsim.rbda.forward_kinematics_model, in_axes=(None,)
        )(
            model,
            base_position=base_position,
            base_quaternion=base_quaternion,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_linear_velocity_inertial=jnp.atleast_2d(base_linear_velocity_inertial),
            base_angular_velocity_inertial=jnp.atleast_2d(
                base_angular_velocity_inertial
            ),
        )

        # Adjust the output shapes.
        joint_positions = joint_positions.reshape(self._joint_positions.shape)
        joint_velocities = joint_velocities.reshape(self._joint_velocities.shape)
        base_quaternion = base_quaternion.reshape(self._base_quaternion.shape)
        base_linear_velocity_inertial = base_linear_velocity_inertial.reshape(
            self._base_linear_velocity.shape
        )
        base_angular_velocity_inertial = base_angular_velocity_inertial.reshape(
            self._base_angular_velocity.shape
        )
        base_position = base_position.reshape(self._base_position.shape)
        base_transform = base_transform.reshape(self._base_transform.shape)
        joint_transforms = joint_transforms.reshape(self._joint_transforms.shape)
        link_transforms = link_transforms.reshape(self._link_transforms.shape)
        link_velocities = link_velocities.reshape(self._link_velocities.shape)

        return super().replace(
            _joint_positions=joint_positions,
            _joint_velocities=joint_velocities,
            _base_quaternion=base_quaternion,
            _base_linear_velocity=base_linear_velocity_inertial,
            _base_angular_velocity=base_angular_velocity_inertial,
            _base_position=base_position,
            _base_transform=base_transform,
            _joint_transforms=joint_transforms,
            _link_transforms=link_transforms,
            _link_velocities=link_velocities,
            contact_state=contact_state,
            validate=validate,
        )

    def valid(self, model: js.model.JaxSimModel) -> bool:
        """
        Check if the `JaxSimModelData` is valid for a given `JaxSimModel`.

        Args:
            model: The `JaxSimModel` to validate the `JaxSimModelData` against.

        Returns:
            `True` if the `JaxSimModelData` is valid for the given model,
            `False` otherwise.
        """
        if self._joint_positions.shape != (model.dofs(),):
            return False
        if self._joint_velocities.shape != (model.dofs(),):
            return False
        if self._base_position.shape != (3,):
            return False
        if self._base_quaternion.shape != (4,):
            return False
        if self._base_linear_velocity.shape != (3,):
            return False
        if self._base_angular_velocity.shape != (3,):
            return False

        return True


@functools.partial(jax.jit, static_argnames=["velocity_representation", "base_rpy_seq"])
def random_model_data(
    model: js.model.JaxSimModel,
    *,
    key: jax.Array | None = None,
    velocity_representation: VelRepr | None = None,
    base_pos_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = ((-1, -1, 0.5), 1.0),
    base_rpy_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-jnp.pi, jnp.pi),
    base_rpy_seq: str = "XYZ",
    joint_pos_bounds: (
        tuple[
            jtp.FloatLike | Sequence[jtp.FloatLike],
            jtp.FloatLike | Sequence[jtp.FloatLike],
        ]
        | None
    ) = None,
    base_vel_lin_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
    base_vel_ang_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
    joint_vel_bounds: tuple[
        jtp.FloatLike | Sequence[jtp.FloatLike],
        jtp.FloatLike | Sequence[jtp.FloatLike],
    ] = (-1.0, 1.0),
) -> JaxSimModelData:
    """
    Randomly generate a `JaxSimModelData` object.

    Args:
        model: The target model for the random data.
        key: The random key.
        velocity_representation: The velocity representation to use.
        base_pos_bounds: The bounds for the base position.
        base_rpy_bounds:
            The bounds for the euler angles used to build the base orientation.
        base_rpy_seq:
            The sequence of axes for rotation (using `Rotation` from scipy).
        joint_pos_bounds:
            The bounds for the joint positions (reading the joint limits if None).
        base_vel_lin_bounds: The bounds for the base linear velocity.
        base_vel_ang_bounds: The bounds for the base angular velocity.
        joint_vel_bounds: The bounds for the joint velocities.

    Returns:
        A `JaxSimModelData` object with random data.
    """

    key = key if key is not None else jax.random.PRNGKey(seed=0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, num=6)

    p_min = jnp.array(base_pos_bounds[0], dtype=float)
    p_max = jnp.array(base_pos_bounds[1], dtype=float)
    rpy_min = jnp.array(base_rpy_bounds[0], dtype=float)
    rpy_max = jnp.array(base_rpy_bounds[1], dtype=float)
    v_min = jnp.array(base_vel_lin_bounds[0], dtype=float)
    v_max = jnp.array(base_vel_lin_bounds[1], dtype=float)
    ω_min = jnp.array(base_vel_ang_bounds[0], dtype=float)
    ω_max = jnp.array(base_vel_ang_bounds[1], dtype=float)
    ṡ_min, ṡ_max = joint_vel_bounds

    base_position = jax.random.uniform(key=k1, shape=(3,), minval=p_min, maxval=p_max)

    base_quaternion = jaxsim.math.Quaternion.to_wxyz(
        xyzw=jax.scipy.spatial.transform.Rotation.from_euler(
            seq=base_rpy_seq,
            angles=jax.random.uniform(
                key=k2, shape=(3,), minval=rpy_min, maxval=rpy_max
            ),
        ).as_quat()
    )

    (
        joint_positions,
        joint_velocities,
        base_linear_velocity,
        base_angular_velocity,
    ) = (None,) * 4

    if model.number_of_joints() > 0:

        s_min, s_max = (
            jnp.array(joint_pos_bounds, dtype=float)
            if joint_pos_bounds is not None
            else (None, None)
        )

        joint_positions = (
            js.joint.random_joint_positions(model=model, key=k3)
            if (s_min is None or s_max is None)
            else jax.random.uniform(
                key=k3, shape=(model.dofs(),), minval=s_min, maxval=s_max
            )
        )

        joint_velocities = jax.random.uniform(
            key=k4, shape=(model.dofs(),), minval=ṡ_min, maxval=ṡ_max
        )

    if model.floating_base():
        base_linear_velocity = jax.random.uniform(
            key=k5, shape=(3,), minval=v_min, maxval=v_max
        )

        base_angular_velocity = jax.random.uniform(
            key=k6, shape=(3,), minval=ω_min, maxval=ω_max
        )

    return JaxSimModelData.build(
        model=model,
        base_position=base_position,
        base_quaternion=base_quaternion,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        **(
            {"velocity_representation": velocity_representation}
            if velocity_representation is not None
            else {}
        ),
    )
