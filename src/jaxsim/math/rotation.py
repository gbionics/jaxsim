import jax.numpy as jnp
import jaxlie

import jaxsim.typing as jtp

from .safe import safe_normalize
from .skew import Skew
from .utils import safe_norm


class Rotation:
    """
    A utility class for rotation matrix operations.
    """

    @staticmethod
    def x(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the X-axis.

        Args:
            theta: Rotation angle in radians.

        Returns:
            The 3D rotation matrix.
        """

        return jaxlie.SO3.from_x_radians(theta=theta).as_matrix()

    @staticmethod
    def y(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the Y-axis.

        Args:
            theta: Rotation angle in radians.

        Returns:
            The 3D rotation matrix.
        """

        return jaxlie.SO3.from_y_radians(theta=theta).as_matrix()

    @staticmethod
    def z(theta: jtp.Float) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix around the Z-axis.

        Args:
            theta: Rotation angle in radians.

        Returns:
            The 3D rotation matrix.
        """

        return jaxlie.SO3.from_z_radians(theta=theta).as_matrix()

    @staticmethod
    def from_axis_angle(vector: jtp.Vector) -> jtp.Matrix:
        """
        Generate a 3D rotation matrix from an axis-angle representation.

        Args:
            vector: Axis-angle representation or the rotation as a 3D vector.

        Returns:
            The SO(3) rotation matrix.
        """

        vector = vector.squeeze()

        theta = safe_norm(vector)

        s = jnp.sin(theta)
        c = jnp.cos(theta)

        c1 = 2 * jnp.sin(theta / 2.0) ** 2

        u, _ = safe_normalize(vector)
        u = jnp.vstack(u.squeeze())

        R = c * jnp.eye(3) - s * Skew.wedge(u) + c1 * u @ u.T

        return R.transpose()

    @staticmethod
    def log_vee(R: jnp.ndarray) -> jtp.Vector:
        """
        Compute the logarithm map of an SO(3) rotation matrix.

        Args:
            R: The SO(3) rotation matrix.

        Returns:
            The corresponding so(3) tangent vector.
        """

        return jaxlie.SO3.from_matrix(R).log()
