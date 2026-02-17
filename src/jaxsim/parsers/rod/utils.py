import os
import pathlib
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import rod
import trimesh
from rod.utils.resolve_uris import resolve_local_uri

import jaxsim.typing as jtp
from jaxsim import logging
from jaxsim.math import Adjoint, Inertia
from jaxsim.parsers import descriptions
from jaxsim.parsers.rod import meshes

MeshMappingMethod = TypeVar("MeshMappingMethod", bound=Callable[..., npt.NDArray])


def from_sdf_inertial(inertial: rod.Inertial) -> jtp.Matrix:
    """
    Extract the 6D inertia matrix from an SDF inertial element.

    Args:
        inertial: The SDF inertial element.

    Returns:
        The 6D inertia matrix of the link expressed in the link frame.
    """

    # Extract the "mass" element.
    m = inertial.mass

    # Extract the "inertia" element.
    inertia_element = inertial.inertia

    ixx = inertia_element.ixx
    iyy = inertia_element.iyy
    izz = inertia_element.izz
    ixy = inertia_element.ixy if inertia_element.ixy is not None else 0.0
    ixz = inertia_element.ixz if inertia_element.ixz is not None else 0.0
    iyz = inertia_element.iyz if inertia_element.iyz is not None else 0.0

    # Build the 3x3 inertia matrix expressed in the CoM.
    I_CoM = np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )

    # Build the 6x6 generalized inertia at the CoM.
    M_CoM = Inertia.to_sixd(mass=m, com=np.zeros(3), I=I_CoM)

    # Compute the transform from the inertial frame (CoM) to the link frame.
    L_H_CoM = inertial.pose.transform() if inertial.pose is not None else np.eye(4)

    # We need its inverse.
    CoM_X_L = Adjoint.from_transform(transform=L_H_CoM, inverse=True)

    # Express the CoM inertia matrix in the link frame L.
    M_L = CoM_X_L.T @ M_CoM @ CoM_X_L

    return M_L.astype(dtype=float)


def joint_to_joint_type(joint: rod.Joint) -> int:
    """
    Extract the joint type from an SDF joint.

    Args:
        joint: The parsed SDF joint.

    Returns:
        The integer corresponding to the joint type.
    """

    axis = joint.axis
    joint_type = joint.type

    if joint_type == "fixed":
        return descriptions.JointType.Fixed

    if not (axis.xyz is not None and axis.xyz.xyz is not None):
        raise ValueError("Failed to read axis xyz data")

    # Make sure that the axis is a unary vector.
    axis_xyz = np.array(axis.xyz.xyz).astype(float)
    axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)

    if joint_type in {"revolute", "continuous"}:
        return descriptions.JointType.Revolute

    if joint_type == "prismatic":
        return descriptions.JointType.Prismatic

    raise ValueError("Joint not supported", axis_xyz, joint_type)


def create_box_collision(
    collision: rod.Collision, link_description: descriptions.LinkDescription
) -> descriptions.BoxCollision:
    """
    Create a box collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The box collision description.
    """

    x, y, z = collision.geometry.box.size

    center = np.array([x / 2, y / 2, z / 2])

    # Define the bottom corners.
    bottom_corners = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0]])

    # Conditionally add the top corners based on the environment variable.
    top_corners = (
        np.array([[0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
        if os.environ.get("JAXSIM_COLLISION_USE_BOTTOM_ONLY", "0").lower()
        in {
            "false",
            "0",
        }
        else []
    )

    # Combine and shift by the center
    box_corners = np.vstack([bottom_corners, *top_corners]) - center

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([center, 1.0]))[0:-1]
    box_corners_wrt_link = (
        H @ np.hstack([box_corners, np.vstack([1.0] * box_corners.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=np.array(corner),
            enabled=True,
        )
        for corner in box_corners_wrt_link.T
    ]

    return descriptions.BoxCollision(
        collidable_points=collidable_points, center=center_wrt_link
    )


def create_sphere_collision(
    collision: rod.Collision, link_description: descriptions.LinkDescription
) -> descriptions.SphereCollision:
    """
    Create a sphere collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.

    Returns:
        The sphere collision description.
    """

    # From https://stackoverflow.com/a/26127012
    def fibonacci_sphere(samples: int) -> npt.NDArray:
        # Get the golden ratio in radians.
        phi = np.pi * (3.0 - np.sqrt(5.0))

        # Generate the points.
        points = [
            np.array(
                [
                    np.cos(phi * i)
                    * np.sqrt(1 - (y := 1 - 2 * i / (samples - 1)) ** 2),
                    y,
                    np.sin(phi * i) * np.sqrt(1 - y**2),
                ]
            )
            for i in range(samples)
        ]

        # Filter to keep only the bottom half if required.
        if os.environ.get("JAXSIM_COLLISION_USE_BOTTOM_ONLY", "0").lower() in {
            "true",
            "1",
        }:
            # Keep only the points with z <= 0.
            points = [point for point in points if point[2] <= 0]

        return np.vstack(points)

    r = collision.geometry.sphere.radius

    sphere_points = r * fibonacci_sphere(
        samples=int(os.getenv(key="JAXSIM_COLLISION_SPHERE_POINTS", default="50"))
    )

    H = collision.pose.transform() if collision.pose is not None else np.eye(4)

    center_wrt_link = (H @ np.hstack([0, 0, 0, 1.0]))[0:-1]

    sphere_points_wrt_link = (
        H @ np.hstack([sphere_points, np.vstack([1.0] * sphere_points.shape[0])]).T
    )[0:3, :]

    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=np.array(point),
            enabled=True,
        )
        for point in sphere_points_wrt_link.T
    ]

    return descriptions.SphereCollision(
        collidable_points=collidable_points, center=center_wrt_link
    )


def create_mesh_collision(
    collision: rod.Collision,
    link_description: descriptions.LinkDescription,
    method: MeshMappingMethod = None,
) -> descriptions.MeshCollision:
    """
    Create a mesh collision from an SDF collision element.

    Args:
        collision: The SDF collision element.
        link_description: The link description.
        method: The method to use for mesh wrapping.

    Returns:
        The mesh collision description.
    """

    file = pathlib.Path(resolve_local_uri(uri=collision.geometry.mesh.uri))
    file_type = file.suffix.replace(".", "")
    mesh = trimesh.load_mesh(file, file_type=file_type)

    if mesh.is_empty:
        raise RuntimeError(f"Failed to process '{file}' with trimesh")

    mesh.apply_scale(collision.geometry.mesh.scale)
    logging.info(
        msg=f"Loading mesh {collision.geometry.mesh.uri} with scale {collision.geometry.mesh.scale}, file type '{file_type}'"
    )

    if method is None:
        method = meshes.VertexExtraction()
        logging.debug("Using default Vertex Extraction method for mesh wrapping")
    else:
        logging.debug(f"Using method {method} for mesh wrapping")

    points = method(mesh=mesh)
    logging.debug(f"Extracted {len(points)} points from mesh")

    W_H_L = collision.pose.transform() if collision.pose is not None else np.eye(4)

    # Extract translation from transformation matrix
    W_p_L = W_H_L[:3, 3]
    mesh_points_wrt_link = points @ W_H_L[:3, :3].T + W_p_L
    collidable_points = [
        descriptions.CollidablePoint(
            parent_link=link_description,
            position=point,
            enabled=True,
        )
        for point in mesh_points_wrt_link
    ]

    return descriptions.MeshCollision(collidable_points=collidable_points, center=W_p_L)


def prepare_mesh_for_parametrization(
    mesh_uri: str, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> dict:
    """
    Load and prepare a mesh for parametric scaling with exact inertia computation.

    This function loads a mesh, ensures it's watertight (crucial for volume/inertia
    calculation), centers it, and returns the data needed for parametric scaling.

    Args:
        mesh_uri: URI/path to the mesh file.
        scale: Initial scale factors to apply (from SDF/URDF).

    Returns:
        A dictionary containing:
            - 'vertices': Centered mesh vertices as numpy array (Nx3)
            - 'faces': Triangle faces as numpy array (Mx3 integer indices)
            - 'offset': Original mesh centroid offset as numpy array (3,)
            - 'uri': The mesh URI for reference
            - 'is_watertight': Boolean indicating if mesh is watertight
            - 'volume': The volume of the mesh (after scaling)
    """

    # Load mesh
    file = pathlib.Path(resolve_local_uri(uri=mesh_uri))
    file_type = file.suffix.replace(".", "")
    mesh = trimesh.load_mesh(file, file_type=file_type)

    if mesh.is_empty:
        raise RuntimeError(f"Failed to process '{file}' with trimesh")

    # Apply initial scale from SDF/URDF
    mesh.apply_scale(scale)

    # Check and fix watertightness
    is_watertight = mesh.is_watertight
    if not is_watertight:
        logging.warning(
            f"Mesh {mesh_uri} is not watertight. Computing convex hull for valid inertia."
        )
        mesh = mesh.convex_hull
        is_watertight = True

    # Store original centroid as offset
    offset = mesh.centroid.copy()

    # Center the mesh
    mesh.vertices -= offset

    return {
        "vertices": np.array(mesh.vertices, dtype=np.float64),
        "faces": np.array(mesh.faces, dtype=np.int32),
        "offset": np.array(offset, dtype=np.float64),
        "uri": mesh_uri,
        "is_watertight": is_watertight,
        "volume": mesh.volume,
    }
