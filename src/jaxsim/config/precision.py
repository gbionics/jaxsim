from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

import jaxsim.typing as jtp


@dataclasses.dataclass(frozen=True)
class PrecisionPolicy:
    """
    Selective precision policy for simulation and optimization paths.
    """

    simulation_dtype: jnp.dtype
    contact_dtype: jnp.dtype
    reduction_dtype: jnp.dtype
    matmul_precision: jax.lax.PrecisionLike

    @staticmethod
    def safe_default() -> "PrecisionPolicy":
        dtype = jnp.asarray(0.0, dtype=float).dtype
        return PrecisionPolicy(
            simulation_dtype=dtype,
            contact_dtype=dtype,
            reduction_dtype=dtype,
            matmul_precision=jax.lax.Precision.HIGHEST,
        )

    @staticmethod
    def tpu_fast() -> "PrecisionPolicy":
        return PrecisionPolicy(
            simulation_dtype=jnp.float32,
            contact_dtype=jnp.float32,
            reduction_dtype=jnp.float32,
            matmul_precision=jax.lax.Precision.DEFAULT,
        )

    @staticmethod
    def reference() -> "PrecisionPolicy":
        return PrecisionPolicy(
            simulation_dtype=jnp.float64,
            contact_dtype=jnp.float64,
            reduction_dtype=jnp.float64,
            matmul_precision=jax.lax.Precision.HIGHEST,
        )

    @staticmethod
    def resolve(policy: "PrecisionPolicy | None") -> "PrecisionPolicy":
        return policy if policy is not None else PrecisionPolicy.safe_default()

    def cast(self, array: jtp.ArrayLike, *, kind: str = "simulation") -> jax.Array:
        """
        Cast an array according to the selected precision role.
        """

        match kind:
            case "simulation":
                dtype = self.simulation_dtype
            case "contact":
                dtype = self.contact_dtype
            case "reduction":
                dtype = self.reduction_dtype
            case _:
                raise ValueError(kind)

        return jnp.asarray(array, dtype=dtype)
