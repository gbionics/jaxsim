from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax_dataclasses
import numpy as np
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.math import Adjoint
from jaxsim.utils import JaxsimDataclass


@jax_dataclasses.pytree_dataclass(eq=False, unsafe_hash=False)
class LinkDescription(JaxsimDataclass):
    """
    In-memory description of a robot link.

    Attributes:
        name: The name of the link.
        mass: The mass of the link.
        inertia: The inertia tensor of the link.
        index: An optional index for the link (it gets automatically assigned).
        parent: The parent link of this link.
        pose: The pose transformation matrix of the link.
        children: The children links.
    """

    name: Static[str]
    mass: float = dataclasses.field(repr=False)
    inertia: jtp.Matrix = dataclasses.field(repr=False)
    index: int | None = None
    parent_name: Static[str | None] = dataclasses.field(default=None, repr=False)
    pose: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.eye(4), repr=False)

    children: Static[tuple[LinkDescription]] = dataclasses.field(
        default_factory=list, repr=False
    )

    def __hash__(self) -> int:

        from jaxsim.utils.wrappers import HashedNumpyArray

        return hash(
            (
                hash(self.name),
                hash(float(self.mass)),
                HashedNumpyArray.hash_of_array(self.inertia),
                hash(int(self.index)) if self.index is not None else 0,
                HashedNumpyArray.hash_of_array(self.pose),
                hash(tuple(self.children)),
                hash(self.parent_name) if self.parent_name is not None else 0,
            )
        )

    def __eq__(self, other: LinkDescription) -> bool:

        if not isinstance(other, LinkDescription):
            return False

        if not (
            self.name == other.name
            and np.allclose(self.mass, other.mass)
            and np.allclose(self.inertia, other.inertia)
            and self.index == other.index
            and np.allclose(self.pose, other.pose)
            and self.children == other.children
            and self.parent_name == other.parent_name
        ):
            return False

        return True

    @property
    def name_and_index(self) -> str:
        """
        A formatted string with the link's name and index.

        Returns:
            str: The formatted string.

        """
        return f"#{self.index}_<{self.name}>"

    def lump_with(
        self, link: LinkDescription, lumped_H_removed: jtp.Matrix
    ) -> LinkDescription:
        """
        Combine the current link with another link, preserving mass and inertia.

        Args:
            link: The link to combine with.
            lumped_H_removed: The transformation matrix between the two links.

        Returns:
            The combined link.
        """

        # Get the 6D inertia of the link to remove.
        I_removed = link.inertia

        # Create the SE3 object. Note the inverse.
        r_X_l = Adjoint.from_transform(transform=lumped_H_removed, inverse=True)

        # Move the inertia
        I_removed_in_lumped_frame = r_X_l.transpose() @ I_removed @ r_X_l

        # Create the new combined link
        lumped_link = self.replace(
            mass=self.mass + link.mass,
            inertia=self.inertia + I_removed_in_lumped_frame,
        )

        return lumped_link
