from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import rod
import rod.builder.primitives
import rod.urdf.exporter

import jaxsim
import jaxsim.api as js
from jaxsim import VelRepr
from jaxsim.config import PrecisionPolicy


def build_box_model(*, contact: bool) -> js.model.JaxSimModel:
    rod_model = (
        rod.builder.primitives.BoxBuilder(x=0.3, y=0.2, z=0.1, mass=1.0, name="box")
        .build_model()
        .add_link(name="box_link")
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    urdf_string = rod.urdf.exporter.UrdfExporter(
        pretty=True, gazebo_preserve_fixed_joints=True
    ).to_urdf_string(sdf=rod_model)

    model = js.model.JaxSimModel.build_from_model_description(urdf_string)

    with model.editable(validate=False) as edited_model:
        edited_model.integrator = js.model.IntegratorType.SemiImplicitEuler

        if not contact:
            edited_model.terrain = jaxsim.terrain.FlatTerrain.build(height=-1e9)
            edited_model.gravity = 0.0

    return edited_model


def build_inputs(
    model: js.model.JaxSimModel, *, horizon: int
) -> tuple[js.data.JaxSimModelData, jax.Array, jax.Array]:
    data_t0 = js.data.JaxSimModelData.build(
        model=model,
        base_position=jnp.array([0.0, 0.0, 0.3]),
        base_linear_velocity=jnp.array([0.2, 0.0, -0.1]),
        velocity_representation=VelRepr.Inertial,
    )

    controls = jnp.zeros((horizon, model.dofs()))
    link_forces = jnp.zeros((horizon, model.number_of_links(), 6))

    return data_t0, controls, link_forces


def rollout_loop(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    controls: jax.Array,
    link_forces: jax.Array,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
) -> js.data.JaxSimModelState:
    data = data_t0

    for control_t, link_force_t in zip(controls, link_forces, strict=True):
        data = js.model.step(
            model=model,
            data=data,
            joint_force_references=control_t,
            link_forces=link_force_t,
            contact_mode=contact_mode,
            precision_policy=precision_policy,
        )

    return data.state


def rollout_scan(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    controls: jax.Array,
    link_forces: jax.Array,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
) -> js.data.JaxSimModelState:
    return js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=controls,
        link_forces=link_forces,
        contact_mode=contact_mode,
        precision_policy=precision_policy,
    )


def block_tree(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def measure(fn, *args):
    start = time.perf_counter()
    output = fn(*args)
    block_tree(output)
    return time.perf_counter() - start, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument(
        "--contact",
        action="store_true",
        help="Benchmark a contact-rich rollout instead of a contact-free one.",
    )
    parser.add_argument(
        "--precision",
        choices=("safe_default", "tpu_fast", "reference"),
        default="safe_default",
    )
    args = parser.parse_args()

    precision_policy = getattr(PrecisionPolicy, args.precision)()
    contact_mode = "masked"

    model = build_box_model(contact=args.contact)
    data_t0, controls, link_forces = build_inputs(model=model, horizon=args.horizon)

    loop_fn = jax.jit(
        lambda data_t0, controls, link_forces: rollout_loop(
            model,
            data_t0,
            controls,
            link_forces,
            precision_policy=precision_policy,
            contact_mode=contact_mode,
        )
    )
    scan_fn = jax.jit(
        lambda data_t0, controls, link_forces: rollout_scan(
            model,
            data_t0,
            controls,
            link_forces,
            precision_policy=precision_policy,
            contact_mode=contact_mode,
        )
    )

    objective_fn = jax.jit(
        jax.grad(
            lambda controls: jnp.sum(
                rollout_scan(
                    model,
                    data_t0,
                    controls,
                    link_forces,
                    precision_policy=precision_policy,
                    contact_mode=contact_mode,
                ).base_position
                ** 2
            )
        )
    )

    loop_compile_time, _ = measure(loop_fn, data_t0, controls, link_forces)
    loop_run_time, _ = measure(loop_fn, data_t0, controls, link_forces)

    scan_compile_time, _ = measure(scan_fn, data_t0, controls, link_forces)
    scan_run_time, final_state = measure(scan_fn, data_t0, controls, link_forces)

    grad_compile_time, _ = measure(objective_fn, controls)
    grad_run_time, grad_value = measure(objective_fn, controls)

    print(f"scenario={ 'contact' if args.contact else 'no_contact' }")
    print(f"horizon={args.horizon}")
    print(f"precision={args.precision}")
    print(f"loop_compile_s={loop_compile_time:.6f}")
    print(f"loop_run_s={loop_run_time:.6f}")
    print(f"scan_compile_s={scan_compile_time:.6f}")
    print(f"scan_run_s={scan_run_time:.6f}")
    print(f"grad_compile_s={grad_compile_time:.6f}")
    print(f"grad_run_s={grad_run_time:.6f}")
    print(f"final_base_position={jnp.asarray(final_state.base_position)}")
    print(f"grad_norm={jnp.linalg.norm(grad_value):.6f}")


if __name__ == "__main__":
    main()
