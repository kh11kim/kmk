#!/usr/bin/env python3
from __future__ import annotations

"""Parseable benchmark harness for HandKinematics variants."""

import argparse
import importlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from kmk import HandInfo, HandKinematics


@dataclass(frozen=True)
class VariantSpec:
    name: str
    target: str


def _parse_variant_spec(raw: str) -> VariantSpec:
    if "=" not in raw or ":" not in raw:
        raise argparse.ArgumentTypeError("variant must have the form name=module:attr")
    name, target = raw.split("=", 1)
    module, attr = target.split(":", 1)
    name = name.strip()
    module = module.strip()
    attr = attr.strip()
    if not name or not module or not attr:
        raise argparse.ArgumentTypeError("variant must have the form name=module:attr")
    return VariantSpec(name=name, target=f"{module}:{attr}")


def _resolve_object(spec: str) -> Any:
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not getattr(torch.backends, "mps", None).is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(samples_ns: Sequence[int]) -> dict[str, float]:
    samples_ms = [value / 1_000_000.0 for value in samples_ns]
    return {
        "mean_ms": float(statistics.fmean(samples_ms)),
        "median_ms": float(statistics.median(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "stdev_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def _measure_sequence(
    fn: Any,
    input_sequence: Sequence[tuple[Any, ...]],
    warmup: int,
    sync_device: torch.device,
) -> list[int]:
    with torch.inference_mode():
        for args in input_sequence[:warmup]:
            fn(*args)
            _maybe_sync(sync_device)

        samples_ns: list[int] = []
        for args in input_sequence[warmup:]:
            start = time.perf_counter_ns()
            fn(*args)
            _maybe_sync(sync_device)
            samples_ns.append(time.perf_counter_ns() - start)
    return samples_ns


def _build_q_batches(
    hand_info: HandInfo,
    batch_sizes: Sequence[int],
    repeats: int,
    warmup: int,
    device: torch.device,
    seed: int,
) -> dict[int, list[torch.Tensor]]:
    kin = HandKinematics(hand_info).to(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    lb = kin.chain.lb.to(device=device, dtype=torch.float32)
    ub = kin.chain.ub.to(device=device, dtype=torch.float32)

    batches: dict[int, list[torch.Tensor]] = {}
    for batch_size in batch_sizes:
        q_list: list[torch.Tensor] = []
        for _ in range(repeats + warmup):
            q = lb + (ub - lb) * torch.rand((batch_size, kin.dof), generator=generator, device=device, dtype=lb.dtype)
            q_list.append(q)
        batches[int(batch_size)] = q_list
    return batches


def _build_points_by_link(hand_info: HandInfo, device: torch.device) -> dict[str, torch.Tensor]:
    points_by_link: dict[str, torch.Tensor] = {}
    base_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, -0.05, 0.0],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.05],
        ],
        dtype=float,
    )
    points_by_link[hand_info.urdf.base_link] = torch.as_tensor(base_points, dtype=torch.float32, device=device)
    for link_name in hand_info.contact_anchor_links:
        try:
            anchor = hand_info.get_contact_anchor_by_link(link_name)
        except KeyError:
            continue
        points_by_link[link_name] = torch.as_tensor(np.asarray(anchor, dtype=float).reshape(1, 3), dtype=torch.float32, device=device)
    return points_by_link


def _instantiate_variant(target: Any, hand_info: HandInfo, device: torch.device) -> Any:
    attempts = (
        lambda: target(hand_info),
        lambda: target(Path(hand_info.config_path)),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt().to(device=device)
        except TypeError as exc:
            last_error = exc
    raise TypeError(f"Failed to instantiate benchmark target {target!r}") from last_error


def _maybe_compile_method(method: Any) -> tuple[Any, str | None]:
    if not hasattr(torch, "compile"):
        return method, "torch.compile is unavailable"
    try:
        compiled = torch.compile(method, mode="reduce-overhead")
    except Exception as exc:  # pragma: no cover - backend dependent
        return method, f"{type(exc).__name__}: {exc}"
    return compiled, None


def _benchmark_variant(
    model: Any,
    q_batches: dict[int, list[torch.Tensor]],
    points_by_link: dict[str, torch.Tensor],
    repeats: int,
    warmup: int,
    device: torch.device,
    compile_methods: bool,
) -> dict[str, Any]:
    fk_fn = model.forward_kinematics
    tp_fn = model.transform_link_points
    fk_compile_error = None
    tp_compile_error = None
    if compile_methods:
        fk_fn, fk_compile_error = _maybe_compile_method(fk_fn)
        tp_fn, tp_compile_error = _maybe_compile_method(tp_fn)

    fk_results: dict[str, Any] = {}
    tp_results: dict[str, Any] = {}
    for batch_size, batches in q_batches.items():
        fk_inputs = [(q,) for q in batches]
        tp_inputs = [(q, points_by_link) for q in batches]
        fk_samples = _measure_sequence(fk_fn, fk_inputs, warmup=warmup, sync_device=device)
        tp_samples = _measure_sequence(tp_fn, tp_inputs, warmup=warmup, sync_device=device)
        fk_results[str(batch_size)] = _stats(fk_samples)
        tp_results[str(batch_size)] = _stats(tp_samples)

    result: dict[str, Any] = {
        "forward_kinematics": fk_results,
        "transform_link_points": tp_results,
        "compiled": bool(compile_methods),
    }
    if fk_compile_error is not None:
        result["forward_kinematics_compile_error"] = fk_compile_error
    if tp_compile_error is not None:
        result["transform_link_points_compile_error"] = tp_compile_error
    return result


def benchmark(
    config_path: str | Path,
    batch_sizes: Sequence[int],
    repeats: int = 20,
    warmup: int = 5,
    device: str = "auto",
    seed: int = 0,
    variant: Sequence[VariantSpec] = (),
    compile_methods: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    batch_sizes = [int(v) for v in batch_sizes]
    if not batch_sizes:
        batch_sizes = [1, 8, 32]
    device_obj = _resolve_device(device)
    hand_info = HandInfo.from_config(config_path)
    q_batches = _build_q_batches(
        hand_info,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmup=warmup,
        device=device_obj,
        seed=seed,
    )
    points_by_link = _build_points_by_link(hand_info, device=device_obj)

    variants = [VariantSpec(name="baseline", target="kmk.kinematics:HandKinematics"), *variant]
    results: dict[str, Any] = {}
    for spec in variants:
        target = _resolve_object(spec.target)
        model = _instantiate_variant(target, hand_info, device_obj)
        results[spec.name] = {
            "target": spec.target,
            "benchmark": _benchmark_variant(
                model=model,
                q_batches=q_batches,
                points_by_link=points_by_link,
                repeats=repeats,
                warmup=warmup,
                device=device_obj,
                compile_methods=compile_methods,
            ),
        }

    comparison: dict[str, Any] = {}
    if "baseline" in results and len(results) > 1:
        baseline = results["baseline"]["benchmark"]
        for name, entry in results.items():
            if name == "baseline":
                continue
            variant_cmp: dict[str, Any] = {}
            for metric_name in ("forward_kinematics", "transform_link_points"):
                baseline_metric = baseline[metric_name]
                candidate_metric = entry["benchmark"][metric_name]
                metric_cmp: dict[str, float] = {}
                for batch_size, baseline_stats in baseline_metric.items():
                    candidate_stats = candidate_metric[batch_size]
                    baseline_ms = float(baseline_stats["mean_ms"])
                    candidate_ms = float(candidate_stats["mean_ms"])
                    metric_cmp[batch_size] = float(baseline_ms / candidate_ms) if candidate_ms > 0 else float("inf")
                variant_cmp[metric_name] = metric_cmp
            comparison[name] = variant_cmp

    headline_batch_size = str(batch_sizes[0])
    baseline_benchmark = results["baseline"]["benchmark"]
    headline_benchmark = {
        "batch_size": int(headline_batch_size),
        "forward_kinematics": baseline_benchmark["forward_kinematics"][headline_batch_size],
        "transform_link_points": baseline_benchmark["transform_link_points"][headline_batch_size],
    }
    if len(batch_sizes) > 1:
        headline_benchmark["by_batch_size"] = {
            str(batch_size): {
                "forward_kinematics": baseline_benchmark["forward_kinematics"][str(batch_size)],
                "transform_link_points": baseline_benchmark["transform_link_points"][str(batch_size)],
            }
            for batch_size in batch_sizes
        }

    summary = {
        "config_path": str(config_path),
        "device_requested": device,
        "device_used": str(device_obj),
        "device": str(device_obj),
        "benchmarks": headline_benchmark,
        "benchmarks_by_batch_size": {
            str(batch_size): {
                "forward_kinematics": baseline_benchmark["forward_kinematics"][str(batch_size)],
                "transform_link_points": baseline_benchmark["transform_link_points"][str(batch_size)],
            }
            for batch_size in batch_sizes
        },
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "batch_sizes": batch_sizes,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "seed": int(seed),
        "compile_methods": bool(compile_methods),
        "variants": results,
    }
    if comparison:
        summary["comparison"] = comparison
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--batch-size", "--batch-sizes", dest="batch_sizes", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", dest="compile_methods")
    parser.add_argument(
        "--variant",
        action="append",
        type=_parse_variant_spec,
        default=[],
        help="Additional variant spec of the form name=module:attr",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = benchmark(
        config_path=args.config_path,
        batch_sizes=args.batch_sizes or [1, 8, 32],
        repeats=args.repeats,
        warmup=args.warmup,
        device=args.device,
        seed=args.seed,
        variant=args.variant,
        compile_methods=args.compile_methods,
    )
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
