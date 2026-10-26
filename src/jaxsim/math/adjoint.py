import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

from .skew import Skew


class Adjoint:
    """
    A utility class for adjoint matrix operations.
    """

    @staticmethod
    def from_quaternion_and_translation(
        quaternion: jtp.Vector | None = None,
        translation: jtp.Vector | None = None,
        inverse: bool = False,
        normalize_quaternion: bool = False,
    ) -> jtp.Matrix:
        """
        Create an adjoint matrix from a quaternion and a translation.

        Args:
            quaternion: A quaternion vector (4D) representing orientation.
            translation: A translation vector (3D).
            inverse: Whether to compute the inverse adjoint.
            normalize_quaternion: Whether to normalize the quaternion before creating the adjoint.

        Returns:
            The adjoint matrix.
        """
        quaternion = quaternion if quaternion is not None else jnp.array([1.0, 0, 0, 0])
        translation = translation if translation is not None else jnp.zeros(3)
        assert quaternion.size == 4
        assert translation.size == 3

        Q_sixd = jaxlie.SO3(wxyz=quaternion)
        Q_sixd = Q_sixd if not normalize_quaternion else Q_sixd.normalize()

        return Adjoint.from_rotation_and_translation(
            rotation=Q_sixd.as_matrix(), translation=translation, inverse=inverse
        )

    @staticmethod
    def from_transform(transform: jtp.MatrixLike, inverse: bool = False) -> jtp.Matrix:
        """
        Create an adjoint matrix from a transformation matrix.

        Args:
            transform: A 4x4 transformation matrix.
            inverse: Whether to compute the inverse adjoint.

        Returns:
            The 6x6 adjoint matrix.
        """

        A_H_B = transform

        return (
            jaxlie.SE3.from_matrix(matrix=A_H_B).adjoint()
            if not inverse
            else jaxlie.SE3.from_matrix(matrix=A_H_B).inverse().adjoint()
        )

    @staticmethod
    def from_rotation_and_translation(
        rotation: jtp.Matrix | None = None,
        translation: jtp.Vector | None = None,
        inverse: bool = False,
    ) -> jtp.Matrix:
        """
        Create an adjoint matrix from a rotation matrix and a translation vector.

        Args:
            rotation: A 3x3 rotation matrix.
            translation: A translation vector (3D).
            inverse: Whether to compute the inverse adjoint. Default is False.

        Returns:
            The adjoint matrix.
        """
        rotation = rotation if rotation is not None else jnp.eye(3)
        translation = translation if translation is not None else jnp.zeros(3)

        assert rotation.shape == (3, 3)
        assert translation.size == 3

        A_R_B = rotation.squeeze()
        A_o_B = translation.squeeze()

        if not inverse:
            X = A_X_B = jnp.vstack(  # ruff: ignore[unused-variable]
                [
                    jnp.block([A_R_B, Skew.wedge(A_o_B) @ A_R_B]),
                    jnp.block([jnp.zeros(shape=(3, 3)), A_R_B]),
                ]
            )
        else:
            X = B_X_A = jnp.vstack(  # ruff: ignore[unused-variable]
                [
                    jnp.block([A_R_B.T, -A_R_B.T @ Skew.wedge(A_o_B)]),
                    jnp.block([jnp.zeros(shape=(3, 3)), A_R_B.T]),
                ]
            )

        return X

    @staticmethod
    def to_transform(adjoint: jtp.Matrix) -> jtp.Matrix:
        """
        Convert an adjoint matrix to a transformation matrix.

        Args:
            adjoint: The adjoint matrix (6x6).

        Returns:
            The transformation matrix (4x4).
        """
        X = adjoint.squeeze()
        assert X.shape == (6, 6)

        R = X[0:3, 0:3]
        o_x_R = X[0:3, 3:6]

        H = jnp.vstack(
            [
                jnp.block([R, Skew.vee(matrix=o_x_R @ R.T)]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

        return H

    @staticmethod
    def inverse(adjoint: jtp.Matrix) -> jtp.Matrix:
        """
        Compute the inverse of an adjoint matrix.

        Args:
            adjoint: The adjoint matrix.

        Returns:
            The inverse adjoint matrix.
        """
        A_X_B = adjoint.reshape(-1, 6, 6)

        A_R_B_T = jnp.swapaxes(A_X_B[..., 0:3, 0:3], -2, -1)
        A_T_B = A_X_B[..., 0:3, 3:6]

        return jnp.concatenate(
            [
                jnp.concatenate(
                    [A_R_B_T, -A_R_B_T @ A_T_B @ A_R_B_T],
                    axis=-1,
                ),
                jnp.concatenate([jnp.zeros_like(A_R_B_T), A_R_B_T], axis=-1),
            ],
            axis=-2,
        ).reshape(adjoint.shape)
