TPU-Friendly Rollouts
=====================

JaxSim now exposes an additive rollout path designed for long-horizon differentiable
simulation and optimization:

.. code-block:: python

    import jax.numpy as jnp
    import jaxsim.api as js
    from jaxsim.config import PrecisionPolicy

    state_t0 = js.data.JaxSimModelData.build(model=model)
    controls = jnp.zeros((128, model.dofs()))

    state_tf, trajectory = js.rollout.rollout_scan(
        model=model,
        state=state_t0,
        controls=controls,
        link_forces=jnp.zeros((128, model.number_of_links(), 6)),
        contact_mode="masked",
        return_trajectory=True,
        checkpoint_policy="step",
        precision_policy=PrecisionPolicy.safe_default(),
    )

The rollout API keeps the scan carry as a minimal state tree and reconstructs cached
kinematics internally when a full :class:`jaxsim.api.data.JaxSimModelData` object is
needed by the existing simulation code.


Static-Shape Contact Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``contact_mode="masked"`` to request the TPU-oriented contact layout.

In this mode:

- collidable-point tensors use the full collidable-point layout
- disabled points stay in the arrays
- disabled or inactive points are masked instead of being structurally removed

The previous enabled-only behavior remains available as ``contact_mode="enabled"`` and
is still the default for the existing APIs.


Checkpointing
~~~~~~~~~~~~~

``rollout_scan`` currently supports the following checkpoint policies:

- ``None``: no rematerialization
- ``"step"``: rematerialize the per-step body

The step-level policy is the current default recommendation for long-horizon gradient
computation when activation memory becomes the main bottleneck.


Precision Policies
~~~~~~~~~~~~~~~~~~

JaxSim now provides explicit precision presets in :mod:`jaxsim.config.precision`:

- ``PrecisionPolicy.safe_default()``
- ``PrecisionPolicy.tpu_fast()``
- ``PrecisionPolicy.reference()``

The policy is additive and opt-in. Existing code continues to use the current default
behavior unless a precision policy is passed explicitly.

In this first pass, the policy is mainly intended for the rollout/contact path and for
keeping dtype choices explicit and auditable. It is not a wholesale replacement for the
current global x64 configuration.


Relaxed-Rigid Fixed Iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RelaxedRigidContacts`` now accepts a ``solver_mode`` argument:

- ``"adaptive"`` keeps the existing tolerance-based loop
- ``"fixed_iterations"`` uses a fixed-trip-count optimization loop

The fixed-iteration mode is intended for optimization-oriented rollouts where predictable
control flow and shape stability are more important than adaptive early stopping.


Research Workflow
~~~~~~~~~~~~~~~~~

The intended workflow for trajectory optimization, system identification, and other
long-horizon differentiable objectives is:

1. Build an initial :class:`jaxsim.api.data.JaxSimModelData`.
2. Call :func:`jaxsim.api.rollout.rollout_scan`.
3. Differentiate an objective through the rollout.
4. Switch to ``contact_mode="masked"`` and, where useful, a fixed-iteration relaxed
   rigid contact solver for more predictable compilation behavior.


Current Limitations
~~~~~~~~~~~~~~~~~~~

- The rollout API is additive and intentionally reuses the existing ``js.model.step``
  implementation instead of introducing a second simulation path.
- The precision policy currently targets the new rollout/contact path and is not yet
  threaded through every internal kernel.
- Low-level accelerator-specific kernels are intentionally deferred until profiling
  justifies them.
