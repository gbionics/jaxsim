from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable

import jax
import jax.numpy as jnp

import jaxsim.api as js
from jaxsim.config import PrecisionPolicy


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_rollout import block_tree, build_box_model, build_inputs


def primitive_histogram(jaxpr) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def is_closed_jaxpr(value) -> bool:
        return hasattr(value, "jaxpr") and hasattr(value, "consts")

    def walk(closed_jaxpr) -> None:
        for eqn in closed_jaxpr.jaxpr.eqns:
            counts[eqn.primitive.name] += 1
            for value in eqn.params.values():
                if is_closed_jaxpr(value):
                    walk(value)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        if is_closed_jaxpr(item):
                            walk(item)

    walk(jaxpr)
    return dict(sorted(counts.items()))


def output_summary(tree) -> str:
    return str(jax.tree.map(lambda x: getattr(x, "shape", None), tree))


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


def make_forward_fn(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    controls: jax.Array,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
    checkpoint_policy: str | None,
) -> Callable[[jax.Array], js.data.JaxSimModelState]:
    def fn(link_forces: jax.Array) -> js.data.JaxSimModelState:
        return js.rollout.rollout_scan(
            model=model,
            state=data_t0,
            controls=controls,
            link_forces=link_forces,
            contact_mode=contact_mode,
            precision_policy=precision_policy,
            checkpoint_policy=checkpoint_policy,
            return_trajectory=False,
        )

    return fn


def make_gradient_fn(
    model: js.model.JaxSimModel,
    data_t0: js.data.JaxSimModelData,
    controls: jax.Array,
    *,
    precision_policy: PrecisionPolicy,
    contact_mode: str,
    checkpoint_policy: str | None,
) -> Callable[[jax.Array], jax.Array]:
    def objective(link_forces: jax.Array) -> jax.Array:
        final_state = js.rollout.rollout_scan(
            model=model,
            state=data_t0,
            controls=controls,
            link_forces=link_forces,
            contact_mode=contact_mode,
            precision_policy=precision_policy,
            checkpoint_policy=checkpoint_policy,
            return_trajectory=False,
        )
        base_error = final_state.base_position - jnp.array([0.25, 0.0, 0.2])
        return (
            jnp.sum(base_error**2)
            + 0.1 * jnp.sum(final_state.base_velocity[:3] ** 2)
            + 1.0e-3 * jnp.sum(link_forces[..., :3] ** 2)
        )

    return jax.grad(objective)


def save_text(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("no_contact", "contact"), default="no_contact")
    parser.add_argument("--mode", choices=("forward", "grad"), default="forward")
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=3)
    parser.add_argument("--checkpoint-policy", choices=("none", "step"), default="none")
    parser.add_argument(
        "--precision",
        choices=("safe_default", "tpu_fast", "reference"),
        default="safe_default",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "trace"
    trace_dir.mkdir(exist_ok=True)

    precision_policy = getattr(PrecisionPolicy, args.precision)()
    checkpoint_policy = None if args.checkpoint_policy == "none" else args.checkpoint_policy

    model = build_box_model(contact=args.scenario == "contact")
    data_t0, controls, link_forces = build_inputs(model=model, horizon=args.horizon)
    contact_mode = "masked"

    fn = (
        make_forward_fn(
            model,
            data_t0,
            controls,
            precision_policy=precision_policy,
            contact_mode=contact_mode,
            checkpoint_policy=checkpoint_policy,
        )
        if args.mode == "forward"
        else make_gradient_fn(
            model,
            data_t0,
            controls,
            precision_policy=precision_policy,
            contact_mode=contact_mode,
            checkpoint_policy=checkpoint_policy,
        )
    )

    jitted_fn = jax.jit(fn)
    lowered = jitted_fn.lower(link_forces)

    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_time_s = time.perf_counter() - compile_start

    stablehlo_path = output_dir / "stablehlo.mlir"
    lowered_path = output_dir / "lowered.txt"
    jaxpr_path = output_dir / "jaxpr.txt"
    summary_path = output_dir / "summary.json"
    memory_profile_path = output_dir / "device_memory.prof"

    save_text(stablehlo_path, str(lowered.compiler_ir(dialect="stablehlo")))
    save_text(lowered_path, lowered.as_text())

    closed_jaxpr = jax.make_jaxpr(fn)(link_forces)
    save_text(jaxpr_path, str(closed_jaxpr))

    run_times = []
    output = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = compiled(link_forces)
        block_tree(output)
        run_times.append(time.perf_counter() - start)

    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        for _ in range(args.trace_repeats):
            traced_output = compiled(link_forces)
            block_tree(traced_output)

    jax.profiler.save_device_memory_profile(str(memory_profile_path))

    summary = {
        "scenario": args.scenario,
        "mode": args.mode,
        "horizon": args.horizon,
        "precision": args.precision,
        "checkpoint_policy": args.checkpoint_policy,
        "compile_time_s": compile_time_s,
        "run_mean_s": statistics.mean(run_times),
        "run_min_s": min(run_times),
        "run_max_s": max(run_times),
        "run_std_s": statistics.stdev(run_times) if len(run_times) > 1 else 0.0,
        "cost_analysis": compiled_cost_analysis(compiled),
        "memory_analysis": compiled_memory_analysis(compiled),
        "primitive_histogram": primitive_histogram(closed_jaxpr),
        "output_summary": output_summary(output),
        "grad_norm": (
            float(jnp.linalg.norm(output)) if args.mode == "grad" else None
        ),
        "artifacts": {
            "stablehlo": str(stablehlo_path),
            "lowered": str(lowered_path),
            "jaxpr": str(jaxpr_path),
            "trace_dir": str(trace_dir),
            "device_memory_profile": str(memory_profile_path),
        },
    }

    save_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
