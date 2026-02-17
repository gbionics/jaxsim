import abc
import contextlib
import dataclasses
import functools
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar

import jax.flatten_util
import jax_dataclasses

import jaxsim.typing as jtp

from . import Mutability

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class JaxsimDataclass(abc.ABC):
    """Class extending `jax_dataclasses.pytree_dataclass` instances with utilities."""

    # This attribute is set by jax_dataclasses
    __mutability__: ClassVar[Mutability] = Mutability.FROZEN

    @contextlib.contextmanager
    def editable(self: Self, validate: bool = True) -> Iterator[Self]:
        """
        Context manager to operate on a mutable copy of the object.

        Args:
            validate: Whether to validate the output PyTree upon exiting the context.

        Yields:
            A mutable copy of the object.

        Note:
            This context manager is useful to operate on an r/w copy of a PyTree making
            sure that the output object does not trigger JIT recompilations.
        """

        mutability = (
            Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
        )

        with self.copy().mutable_context(mutability=mutability) as obj:
            yield obj

    @contextlib.contextmanager
    def mutable_context(
        self: Self,
        mutability: Mutability = Mutability.MUTABLE,
        restore_after_exception: bool = True,
    ) -> Iterator[Self]:
        """
        Context manager to temporarily change the mutability of the object.

        Args:
            mutability: The mutability to set.
            restore_after_exception:
                Whether to restore the original object in case of an exception
                occurring within the context.

        Yields:
            The object with the new mutability.

        Note:
            This context manager is useful to operate in place on a PyTree without
            the need to make a copy while optionally keeping active the checks on
            the PyTree structure, shapes, and dtypes.
        """

        if restore_after_exception:
            self_copy = self.copy()

        original_mutability = self.mutability()

        original_dtypes = JaxsimDataclass.get_leaf_dtypes(tree=self)
        original_shapes = JaxsimDataclass.get_leaf_shapes(tree=self)
        original_weak_types = JaxsimDataclass.get_leaf_weak_types(tree=self)
        original_structure = jax.tree.structure(tree=self)

        def restore_self() -> None:
            self.set_mutability(mutability=Mutability.MUTABLE_NO_VALIDATION)
            for f in dataclasses.fields(self_copy):
                setattr(self, f.name, getattr(self_copy, f.name))

        try:
            self.set_mutability(mutability=mutability)
            yield self

            if mutability is not Mutability.MUTABLE_NO_VALIDATION:
                new_structure = jax.tree.structure(tree=self)
                if original_structure != new_structure:
                    msg = "Pytree structure has changed from {} to {}"
                    raise ValueError(msg.format(original_structure, new_structure))

                new_shapes = JaxsimDataclass.get_leaf_shapes(tree=self)
                if original_shapes != new_shapes:
                    msg = "Leaves shapes have changed from {} to {}"
                    raise ValueError(msg.format(original_shapes, new_shapes))

                new_dtypes = JaxsimDataclass.get_leaf_dtypes(tree=self)
                if original_dtypes != new_dtypes:
                    msg = "Leaves dtypes have changed from {} to {}"
                    raise ValueError(msg.format(original_dtypes, new_dtypes))

                new_weak_types = JaxsimDataclass.get_leaf_weak_types(tree=self)
                if original_weak_types != new_weak_types:
                    msg = "Leaves weak types have changed from {} to {}"
                    raise ValueError(msg.format(original_weak_types, new_weak_types))

        except Exception as e:
            if restore_after_exception:
                restore_self()
            self.set_mutability(original_mutability)
            raise e

        finally:
            self.set_mutability(original_mutability)

    @staticmethod
    def get_leaf_shapes(tree: jtp.PyTree) -> tuple[tuple[int, ...] | None]:
        """
        Get the leaf shapes of a PyTree.

        Args:
            tree: The PyTree to consider.

        Returns:
            A tuple containing the leaf shapes of the PyTree or `None` is the leaf is
            not a numpy-like array.
        """

        return tuple(
            map(
                lambda leaf: getattr(leaf, "shape", None),
                jax.tree.leaves(tree),
            )
        )

    @staticmethod
    def get_leaf_dtypes(tree: jtp.PyTree) -> tuple:
        """
        Get the leaf dtypes of a PyTree.

        Args:
            tree: The PyTree to consider.

        Returns:
            A tuple containing the leaf dtypes of the PyTree or `None` is the leaf is
            not a numpy-like array.
        """

        return tuple(
            map(
                lambda leaf: getattr(leaf, "dtype", None),
                jax.tree.leaves(tree),
            )
        )

    @staticmethod
    def get_leaf_weak_types(tree: jtp.PyTree) -> tuple[bool, ...]:
        """
        Get the leaf weak types of a PyTree.

        Args:
            tree: The PyTree to consider.

        Returns:
            A tuple marking whether the leaf contains a JAX array with weak type.
        """

        return tuple(
            map(
                lambda leaf: getattr(leaf, "weak_type", None),
                jax.tree.leaves(tree),
            )
        )

    @staticmethod
    def check_compatibility(*trees: Sequence[Any]) -> None:
        """
        Check whether the PyTrees are compatible in structure, shape, and dtype.

        Args:
            *trees: The PyTrees to compare.

        Raises:
            ValueError: If the PyTrees have incompatible structures, shapes, or dtypes.
        """

        target_structure = jax.tree.structure(trees[0])

        compatible_structure = functools.reduce(
            lambda compatible, tree: compatible
            and jax.tree.structure(tree) == target_structure,
            trees[1:],
            True,
        )

        if not compatible_structure:
            raise ValueError(
                f"Pytrees have incompatible structures.\n"
                f"Original: {', '.join(map(str, [jax.tree.structure(tree) for tree in trees[1:]]))}\n"
                f"Target: {target_structure}"
            )

        target_shapes = JaxsimDataclass.get_leaf_shapes(trees[0])

        compatible_shapes = functools.reduce(
            lambda compatible, tree: compatible
            and JaxsimDataclass.get_leaf_shapes(tree) == target_shapes,
            trees[1:],
            True,
        )

        if not compatible_shapes:
            raise ValueError("Pytrees have incompatible shapes.")

        target_dtypes = JaxsimDataclass.get_leaf_dtypes(trees[0])

        compatible_dtypes = functools.reduce(
            lambda compatible, tree: compatible
            and JaxsimDataclass.get_leaf_dtypes(tree) == target_dtypes,
            trees[1:],
            True,
        )

        if not compatible_dtypes:
            raise ValueError("Pytrees have incompatible dtypes.")

    def is_mutable(self, validate: bool = False) -> bool:
        """
        Check whether the object is mutable.

        Args:
            validate: Additionally checks if the object also has validation enabled.

        Returns:
            True if the object is mutable, False otherwise.
        """

        return (
            self.__mutability__ is Mutability.MUTABLE
            if validate
            else self.__mutability__ is Mutability.MUTABLE_NO_VALIDATION
        )

    def mutability(self) -> Mutability:
        """
        Get the mutability type of the object.

        Returns:
            The mutability type of the object.
        """

        return self.__mutability__

    def set_mutability(self, mutability: Mutability) -> None:
        """
        Set the mutability of the object in-place.

        Args:
            mutability: The desired mutability type.
        """

        jax_dataclasses._copy_and_mutate._mark_mutable(
            self, mutable=mutability, visited=set()
        )

    def mutable(self: Self, mutable: bool = True, validate: bool = False) -> Self:
        """
        Return a mutable reference of the object.

        Args:
            mutable: Whether to make the object mutable.
            validate: Whether to enable validation on the object.

        Returns:
            A mutable reference of the object.
        """

        if mutable:
            mutability = (
                Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
            )
        else:
            mutability = Mutability.FROZEN

        self.set_mutability(mutability=mutability)
        return self

    def copy(self: Self) -> Self:
        """
        Return a copy of the object.

        Returns:
            A copy of the object.
        """

        # Make a copy calling tree_map.
        obj = jax.tree.map(lambda leaf: leaf, self)

        # Make sure that the copied object and all the copied leaves have the same
        # mutability of the original object.
        obj.set_mutability(mutability=self.mutability())

        return obj

    def replace(self: Self, validate: bool = True, **kwargs) -> Self:
        """
        Return a new object replacing in-place the specified fields with new values.

        Args:
            validate: Whether to validate that the new fields do not alter the PyTree.
            **kwargs: The fields to replace.

        Returns:
            A reference of the object with the specified fields replaced.
        """

        # Use the dataclasses replace method.
        obj = dataclasses.replace(self, **kwargs)

        if validate:
            JaxsimDataclass.check_compatibility(self, obj)

        # Make sure that all the new leaves have the same mutability of the object.
        obj.set_mutability(mutability=self.mutability())

        return obj

    def flatten(self) -> jtp.Vector:
        """
        Flatten the object into a 1D vector.

        Returns:
            A 1D vector containing the flattened object.
        """

        return self.flatten_fn()(self)

    @classmethod
    def flatten_fn(cls: type[Self]) -> Callable[[Self], jtp.Vector]:
        """
        Return a function to flatten the object into a 1D vector.

        Returns:
            A function to flatten the object into a 1D vector.
        """

        return lambda pytree: jax.flatten_util.ravel_pytree(pytree)[0]

    def unflatten_fn(self: Self) -> Callable[[jtp.Vector], Self]:
        """
        Return a function to unflatten a 1D vector into the object.

        Returns:
            A function to unflatten a 1D vector into the object.

        Notes:
            Due to JAX internals, the function to unflatten a PyTree needs to be
            created from an existing instance of the PyTree.
        """
        return jax.flatten_util.ravel_pytree(self)[1]
