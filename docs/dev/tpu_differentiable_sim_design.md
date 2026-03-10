# TPU-Friendly Differentiable Simulation Design Note

## Scope

This note documents a conservative extension of JaxSim for long-horizon differentiable
simulation and optimization on accelerator backends, with explicit attention to TPU
execution constraints.

The goal of this pass is not to rewrite the simulator. The goal is to add a clean,
reviewable path that keeps the existing APIs working while introducing:

- a lean simulation-state carry for `jax.lax.scan`
- a rollout API that is shape-stable by construction
- a masked static-shape contact path
- focused safe-math helpers for numerically fragile primitives
- an explicit precision-policy surface for selective casting

Low-level kernel work is intentionally deferred until profiling justifies it.

## Current Structure

### Model state and caches

`jaxsim.api.data.JaxSimModelData` currently stores:

- minimal simulation state
  - base position
  - base quaternion
  - base linear/angular velocity in inertial representation
  - joint positions/velocities
  - contact state
- eagerly materialized derived quantities
  - base transform
  - joint transforms
  - link transforms
  - link velocities

This makes the public object convenient, but it also couples scan carry size to
derived kinematics unless downstream code extracts the minimal state manually.

### Simulation step path

`jaxsim.api.model.step` delegates to the selected integrator. The current integrators
operate on `JaxSimModelData` and rebuild caches through `replace(model=...)`.

The current path is correct, but not ideal for optimization-oriented rollouts because:

- the carry is larger than necessary if the full data object is propagated
- cache rebuilds are coupled to state mutation
- Python loops are still the dominant pattern in tests/examples for multi-step execution

### Contact execution

Contact logic already uses JAX-friendly array code, but most public helpers operate on
"enabled collidable points" rather than the full collidable-point layout.

That is acceptable for regular simulation because the enabled set is static for a fixed
model, but it is not the clearest contract for TPU-oriented rollouts. A TPU-friendly
path should be explicit about using:

- fixed maximum collidable-point tensors
- masking for disabled or inactive points
- shape-stable contact Jacobians/forces across time

### Numerically fragile operations

The codebase already includes `safe_norm`, but several fragile patterns remain:

- quaternion normalization repeated ad hoc with `jnp.where`
- division by norms or other small values in multiple modules
- contact/friction expressions that rely on zero checks inside `jnp.where`
- global `float` casting without a selective precision abstraction

### Precision story

JaxSim currently defaults to global x64 when available and warns on 32-bit use. That is
reasonable for correctness, but it does not provide an auditable mixed-precision path.

For TPU execution, a global dtype switch is too blunt. Sensitive reductions or contact
solves may need different dtypes from the surrounding kinematics.

## Pain Points

The audit identified the following practical bottlenecks for TPU-friendly optimization:

1. `JaxSimModelData` mixes minimal state and cached quantities.
2. The default multi-step usage pattern is still Python-loop oriented.
3. Contact helper outputs are indexed by enabled points instead of an explicit static
   layout plus mask.
4. Quaternion normalization and guarded division are repeated in slightly different ways.
5. `RelaxedRigidContacts` uses an adaptive `while_loop` around the optimizer, which is
   less predictable than a fixed-trip-count path for optimization-oriented rollouts.
6. The codebase has no explicit precision-policy abstraction.

## Proposed Changes

### 1. Internal state/cache decomposition

Add a small internal state dataclass and cache helpers while keeping
`JaxSimModelData` as the public object.

The intended split is:

- `JaxSimModelState`
  - minimal simulation carry for `lax.scan`
- `JaxSimModelData`
  - public state-plus-cache object
- cache reconstruction helpers
  - used when a full `JaxSimModelData` is required

This preserves the public API while allowing rollout code to carry only the minimal
state tree.

### 2. Rollout API

Add `jaxsim.api.rollout.rollout_scan(...)` as the TPU/optimization-oriented public
entry point.

The rollout API should:

- use `jax.lax.scan`
- accept fixed-shape control sequences
- optionally return only the final state
- avoid materializing full trajectories unless requested
- expose checkpointing and precision-policy hooks explicitly

### 3. Static-shape masked contact mode

Add an explicit masked contact mode to collidable-point and contact helpers.

The new mode should:

- use the full collidable-point layout
- keep disabled points in the tensors
- apply masking instead of structural filtering
- preserve the existing enabled-only behavior as the default path

This keeps backward compatibility while making the TPU-friendly path explicit.

### 4. Safe math utilities

Add a focused safe-math module containing:

- guarded division
- safe normalization
- quaternion-safe normalization
- stable norm wrappers reused by existing quaternion/contact code

The intent is to centralize gradient-sensitive primitives instead of scattering
slightly different `jnp.where` patterns throughout the codebase.

### 5. Fixed-iteration relaxed-rigid solver mode

Extend `RelaxedRigidContacts` with a fixed-iteration solver mode that uses a
static-trip-count primitive for the optimization path while preserving the existing
adaptive/general mode.

This is useful because optimization-oriented rollouts benefit from:

- predictable control flow
- consistent differentiation behavior
- TPU/XLA-friendly compilation

### 6. Precision policy

Add an explicit precision policy with conservative presets:

- `safe_default`
- `tpu_fast`
- `reference`

The policy should remain additive and opt-in. Existing behavior should still work
without callers having to opt into the new surface.

## Backward Compatibility

The following should remain backward compatible:

- `JaxSimModelData` construction and property access
- existing `js.model.step(...)` semantics
- existing contact behavior when no new mode is requested
- current integrator selection on `JaxSimModel`
- existing tests and callers that operate on enabled-only contact helpers

The new rollout and precision surfaces are additive.

## What Is New

This pass introduces a new optimization-oriented layer rather than replacing the
existing simulation API:

- minimal scan-carry state
- rollout API
- masked contact layout
- safe math helpers
- precision-policy presets
- fixed-iteration relaxed-rigid solve mode

## Why The Added Complexity Is Worth It

Each addition is justified by a concrete pain point:

- State/cache split:
  reduces scan carry size without removing the public cached object.
- Rollout API:
  avoids ad hoc Python loops and makes differentiation through horizons explicit.
- Masked contact mode:
  clarifies the static-shape contract required by TPU-oriented rollouts.
- Safe math:
  reduces NaN-prone gradient paths at the source instead of hiding failures.
- Fixed-iteration solver:
  gives an optimization-focused alternative to adaptive control flow.
- Precision policy:
  enables selective precision changes that are local and reviewable.

## Deferred Work

The following are intentionally deferred from this pass:

- Pallas or other low-level kernel rewrites
- broad algorithmic rewrites of ABA/CRBA/RNEA
- new configuration surfaces that duplicate existing model settings

These should only be revisited after profiling the pure-JAX rollout path and after the
new tests/benchmarks identify true bottlenecks.
