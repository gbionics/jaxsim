from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Generic, TypeVar

import jax
import jax_dataclasses
import numpy as np
import numpy.typing as npt

T = TypeVar("T")


@dataclasses.dataclass
class HashlessObject(Generic[T]):
    """
    A class that wraps an object and makes it hashless.

    This is useful for creating particular JAX pytrees.
    For example, to create a pytree with a static leaf that is ignored
    by JAX when it compares two instances to trigger a JIT recompilation.
    """

    obj: T

    def get(self: HashlessObject[T]) -> T:
        """
        Get the wrapped object.
        """
        return self.obj

    def __hash__(self) -> int:

        return 0

    def __eq__(self, other: HashlessObject[T]) -> bool:

        if not isinstance(other, HashlessObject) and isinstance(
            other.get(), type(self.get())
        ):
            return False

        return hash(self) == hash(other)


@dataclasses.dataclass
class CustomHashedObject(Generic[T]):
    """
    A class that wraps an object and computes its hash with a custom hash function.
    """

    obj: T

    hash_function: Callable[[T], int] = hash

    def get(self: CustomHashedObject[T]) -> T:
        """
        Get the wrapped object.
        """
        return self.obj

    def __hash__(self) -> int:

        return self.hash_function(self.obj)

    def __eq__(self, other: CustomHashedObject[T]) -> bool:

        if not isinstance(other, CustomHashedObject) and isinstance(
            other.get(), type(self.get())
        ):
            return False

        return hash(self) == hash(other)


@jax_dataclasses.pytree_dataclass
class HashedNumpyArray:
    """
    A class that wraps a numpy array and makes it hashable.

    This is useful for creating particular JAX pytrees.
    For example, to create a pytree with a plain NumPy or JAX NumPy array as static leaf.

    Note:
        Calculating with the wrapper class the hash of a very large array can be
        very expensive. If the array is large and only the equality operator is needed,
        set `large_array=True` to use a faster comparison method.
    """

    array: jax.Array | npt.NDArray

    precision: float | None = dataclasses.field(
        default=1e-9, repr=False, compare=False, hash=False
    )

    large_array: jax_dataclasses.Static[bool] = dataclasses.field(
        default=False, repr=False, compare=False, hash=False
    )

    def get(self) -> jax.Array | npt.NDArray:
        """
        Get the wrapped array.
        """
        return self.array

    def __hash__(self) -> int:

        return HashedNumpyArray.hash_of_array(
            array=self.array, precision=self.precision
        )

    def __eq__(self, other: HashedNumpyArray) -> bool:

        if not isinstance(other, HashedNumpyArray):
            return False

        if self.large_array:
            return np.allclose(
                self.array,
                other.array,
                **(dict(atol=self.precision) if self.precision is not None else {}),
            )

        return hash(self) == hash(other)

    @staticmethod
    def hash_of_array(
        array: jax.Array | npt.NDArray, precision: float | None = 1e-9
    ) -> int:
        """
        Calculate the hash of a NumPy array.

        Args:
            array: The array to hash.
            precision: Optionally limit the precision over which the hash is computed.

        Returns:
            The hash of the array.
        """

        array = np.array(array, dtype=float).flatten()

        array = np.where(np.isnan(array), hash(np.nan), array)
        array = np.where(array == np.inf, hash(np.inf), array)
        array = np.where(array == -np.inf, hash(-np.inf), array)

        if precision is not None:

            integer1 = (array * precision).astype(int)
            integer2 = (array - integer1 / precision).astype(int)

            decimal_array = ((array - integer1 * 1e9 - integer2) / precision).astype(
                int
            )

            array = np.hstack([integer1, integer2, decimal_array]).astype(int)

        return hash(tuple(array.tolist()))
