import jax.lax
import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

from .utils import safe_norm


class Quaternion:
    """
    A utility class for quaternion operations.
    """

    @staticmethod
    def to_xyzw(wxyz: jtp.Vector) -> jtp.Vector:
        """
        Convert a quaternion from WXYZ to XYZW representation.

        Args:
            wxyz: Quaternion in WXYZ representation.

        Returns:
            Quaternion in XYZW representation.
        """
        return wxyz.squeeze()[jnp.array([1, 2, 3, 0])]

    @staticmethod
    def to_wxyz(xyzw: jtp.Vector) -> jtp.Vector:
        """
        Convert a quaternion from XYZW to WXYZ representation.

        Args:
            xyzw: Quaternion in XYZW representation.

        Returns:
            Quaternion in WXYZ representation.
        """
        return xyzw.squeeze()[jnp.array([3, 0, 1, 2])]

    @staticmethod
    def to_dcm(quaternion: jtp.Vector) -> jtp.Matrix:
        """
        Convert a quaternion to a direction cosine matrix (DCM).

        Args:
            quaternion: Quaternion in XYZW representation.

        Returns:
            The Direction cosine matrix (DCM).
        """
        return jaxlie.SO3(wxyz=quaternion).as_matrix()

    @staticmethod
    def from_dcm(dcm: jtp.Matrix) -> jtp.Vector:
        """
        Convert a direction cosine matrix (DCM) to a quaternion.

        Args:
            dcm: Direction cosine matrix (DCM).

        Returns:
            Quaternion in WXYZ representation.
        """
        return jaxlie.SO3.from_matrix(matrix=dcm).wxyz

    @staticmethod
    def derivative(
        quaternion: jtp.Vector,
        omega: jtp.Vector,
        K: float = 0.1,
    ) -> jtp.Vector:
        """
        Compute the derivative of a quaternion given angular velocity.

        Args:
            quaternion: Quaternion in XYZW representation.
            omega: Angular velocity vector.
            K (float): A scaling factor.

        Returns:
            The derivative of the quaternion.
        """
        ω = omega.squeeze()
        q = quaternion.squeeze()

        # Construct pure quaternion: (scalar damping term, angular velocity components)
        ω_quat = jnp.hstack([K * safe_norm(ω) * (1 - safe_norm(quaternion)), ω])

        # Apply quaternion multiplication based on frame representation
        i_idx = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
        j_idx = jnp.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 0, 1, 3], [3, 0, 2, 1]])
        sign_matrix = jnp.array(
            [
                [1, -1, -1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, -1],
            ]
        )

        # Compute quaternion derivative via Einstein summation
        q_outer = jnp.einsum("...i,...j->...ij", q, ω_quat)

        Qd = jnp.sum(
            sign_matrix * q_outer[..., i_idx, j_idx],
            axis=-1,
        )

        return 0.5 * Qd

    @staticmethod
    def integration(
        quaternion: jtp.VectorLike,
        dt: jtp.FloatLike,
        omega: jtp.VectorLike,
        omega_in_body_fixed: jtp.BoolLike = False,
    ) -> jtp.Vector:
        """
        Integrate a quaternion in SO(3) given an angular velocity.

        Args:
            quaternion: The quaternion to integrate.
            dt: The time step.
            omega: The angular velocity vector.
            omega_in_body_fixed:
                Whether the angular velocity is in body-fixed representation
                as opposed to the default inertial-fixed representation.

        Returns:
            The integrated quaternion.
        """

        ω_AB = jnp.array(omega).squeeze().astype(float)
        A_Q_B = jnp.array(quaternion).squeeze().astype(float)

        # Build the initial SO(3) quaternion.
        W_Q_B_t0 = jaxlie.SO3(wxyz=A_Q_B)

        # Integrate the quaternion on the manifold.
        W_Q_B_tf = jax.lax.select(
            pred=omega_in_body_fixed,
            on_true=(W_Q_B_t0 @ jaxlie.SO3.exp(tangent=dt * ω_AB)).wxyz,
            on_false=(jaxlie.SO3.exp(tangent=dt * ω_AB) @ W_Q_B_t0).wxyz,
        )

        return W_Q_B_tf
