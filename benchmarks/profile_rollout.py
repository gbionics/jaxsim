from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

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


def block_tree(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


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
    checkpoint_policy: str | None,
    return_trajectory: bool,
) -> js.data.JaxSimModelState | tuple[js.data.JaxSimModelState, js.data.JaxSimModelState]:
    return js.rollout.rollout_scan(
        model=model,
        state=data_t0,
        controls=controls,
        link_forces=link_forces,
        contact_mode=contact_mode,
        precision_policy=precision_policy,
        checkpoint_policy=checkpoint_policy,
        return_trajectory=return_trajectory,
    )


def make_variant(
    name: str,
    model: js.model.JaxSimModel,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
) -> Callable[[js.data.JaxSimModelData, jax.Array, jax.Array], object]:
    match name:
        case "loop":
            return lambda data_t0, controls, link_forces: rollout_loop(
                model,
                data_t0,
                controls,
                link_forces,
                precision_policy=precision_policy,
                contact_mode=contact_mode,
            )
        case "scan":
            return lambda data_t0, controls, link_forces: rollout_scan(
                model,
                data_t0,
                controls,
                link_forces,
                precision_policy=precision_policy,
                contact_mode=contact_mode,
                checkpoint_policy=None,
                return_trajectory=False,
            )
        case "scan_checkpoint":
            return lambda data_t0, controls, link_forces: rollout_scan(
                model,
                data_t0,
                controls,
                link_forces,
                precision_policy=precision_policy,
                contact_mode=contact_mode,
                checkpoint_policy="step",
                return_trajectory=False,
            )
        case "scan_trajectory":
            return lambda data_t0, controls, link_forces: rollout_scan(
                model,
                data_t0,
                controls,
                link_forces,
                precision_policy=precision_policy,
                contact_mode=contact_mode,
                checkpoint_policy=None,
                return_trajectory=True,
            )
        case _:
            raise ValueError(name)


def compiled_cost_analysis(compiled) -> dict[str, float]:
    if not hasattr(compiled, "cost_analysis"):
        return {}

    result = compiled.cost_analysis()
    if isinstance(result, list):
        result = result[0]

    return {str(k): float(v) for k, v in result.items()}


def compiled_memory_analysis(compiled) -> dict[str, int | None]:
    if not hasattr(compiled, "memory_analysis"):
        return {}

    analysis = compiled.memory_analysis()
    if analysis is None:
        return {}

    fields = (
        "generated_code_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {field: getattr(analysis, field, None) for field in fields}


def measure_variant(
    variant_name: str,
    fn: Callable,
    args: tuple[object, ...],
    *,
    repeats: int,
) -> dict[str, object]:
    jitted_fn = jax.jit(fn)

    compile_start = time.perf_counter()
    lowered = jitted_fn.lower(*args)
    compiled = lowered.compile()
    compile_time_s = time.perf_counter() - compile_start

    steady_times = []
    output = None

    for _ in range(repeats):
        run_start = time.perf_counter()
        output = compiled(*args)
        block_tree(output)
        steady_times.append(time.perf_counter() - run_start)

    return {
        "variant": variant_name,
        "compile_time_s": compile_time_s,
        "run_mean_s": statistics.mean(steady_times),
        "run_min_s": min(steady_times),
        "run_max_s": max(steady_times),
        "run_std_s": statistics.stdev(steady_times) if len(steady_times) > 1 else 0.0,
        "cost_analysis": compiled_cost_analysis(compiled),
        "memory_analysis": compiled_memory_analysis(compiled),
        "output_summary": str(jax.tree.map(lambda x: getattr(x, "shape", None), output)),
    }


def measure_scan_gradient(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    controls: jax.Array,
    link_forces: jax.Array,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
    checkpoint_policy: str | None,
    repeats: int,
) -> dict[str, object]:
    def objective(link_forces):
        final_state = rollout_scan(
            model,
            data_t0,
            controls,
            link_forces,
            precision_policy=precision_policy,
            contact_mode=contact_mode,
            checkpoint_policy=checkpoint_policy,
            return_trajectory=False,
        )
        base_error = final_state.base_position - jnp.array([0.25, 0.0, 0.2])
        return (
            jnp.sum(base_error**2)
            + 0.1 * jnp.sum(final_state.base_velocity[:3] ** 2)
            + 1.0e-3 * jnp.sum(link_forces[..., :3] ** 2)
        )

    grad_fn = jax.jit(jax.grad(objective))

    compile_start = time.perf_counter()
    lowered = grad_fn.lower(link_forces)
    compiled = lowered.compile()
    compile_time_s = time.perf_counter() - compile_start

    steady_times = []
    grad_value = None

    for _ in range(repeats):
        run_start = time.perf_counter()
        grad_value = compiled(link_forces)
        block_tree(grad_value)
        steady_times.append(time.perf_counter() - run_start)

    return {
        "variant": f"scan_grad_{checkpoint_policy or 'none'}",
        "compile_time_s": compile_time_s,
        "run_mean_s": statistics.mean(steady_times),
        "run_min_s": min(steady_times),
        "run_max_s": max(steady_times),
        "run_std_s": statistics.stdev(steady_times) if len(steady_times) > 1 else 0.0,
        "cost_analysis": compiled_cost_analysis(compiled),
        "memory_analysis": compiled_memory_analysis(compiled),
        "grad_norm": float(jnp.linalg.norm(grad_value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--precision",
        choices=("safe_default", "tpu_fast", "reference"),
        default="safe_default",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("no_contact", "contact"),
        default=("no_contact", "contact"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("loop", "scan", "scan_checkpoint", "scan_trajectory"),
        default=("loop", "scan", "scan_checkpoint", "scan_trajectory"),
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    precision_policy = getattr(PrecisionPolicy, args.precision)()
    results: list[dict[str, object]] = []

    for scenario in args.scenarios:
        contact = scenario == "contact"
        model = build_box_model(contact=contact)
        contact_mode = "masked"

        for horizon in args.horizons:
            data_t0, controls, link_forces = build_inputs(model=model, horizon=horizon)
            call_args = (data_t0, controls, link_forces)

            for variant_name in args.variants:
                variant_fn = make_variant(
                    variant_name,
                    model,
                    precision_policy=precision_policy,
                    contact_mode=contact_mode,
                )
                metrics = measure_variant(
                    variant_name,
                    variant_fn,
                    call_args,
                    repeats=args.repeats,
                )
                metrics.update(
                    {
                        "scenario": scenario,
                        "horizon": horizon,
                        "precision": args.precision,
                    }
                )
                results.append(metrics)

            for checkpoint_policy in (None, "step"):
                grad_metrics = measure_scan_gradient(
                    model,
                    data_t0,
                    controls,
                    link_forces,
                    precision_policy=precision_policy,
                    contact_mode=contact_mode,
                    checkpoint_policy=checkpoint_policy,
                    repeats=args.repeats,
                )
                grad_metrics.update(
                    {
                        "scenario": scenario,
                        "horizon": horizon,
                        "precision": args.precision,
                    }
                )
                results.append(grad_metrics)

    payload = json.dumps(results, indent=2, sort_keys=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)

    print(payload)


if __name__ == "__main__":
    main()
