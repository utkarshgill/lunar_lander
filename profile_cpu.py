#!/usr/bin/env python3
"""Compare PyTorch and tinygrad CPU matrix execution with one timing protocol."""

import argparse
import base64
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings


SHAPES = ((4, 128, 128), (10_000, 128, 128), (128, 10_000, 128))
PPO_BATCH_SIZE = 10_000


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def fixed_matrix(m, k, n):
    import numpy as np

    random = np.random.default_rng(m * 1_000_003 + k * 1_009 + n)
    return (
        random.standard_normal((m, k), dtype=np.float32),
        random.standard_normal((k, n), dtype=np.float32),
    )


def error_metrics(actual, expected):
    import numpy as np

    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.abs(difference).max()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
    }


def matrix_reference(left, right):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return left.astype("float64") @ right.astype("float64")


def measure(operation, synchronize, warmup, samples):
    for _ in range(warmup):
        operation()
        synchronize()

    times_ms = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        operation()
        synchronize()
        times_ms.append((time.perf_counter_ns() - start) / 1e6)

    return {
        "median_ms": statistics.median(times_ms),
        "p10_ms": percentile(times_ms, 0.10),
        "p90_ms": percentile(times_ms, 0.90),
    }


def torch_worker(args):
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.test == "ppo":
        return torch_ppo_worker(args, torch)

    ordered_shapes = SHAPES[args.process_index % len(SHAPES):] + SHAPES[:args.process_index % len(SHAPES)]
    results = {}

    for m, k, n in ordered_shapes:
        left_array, right_array = fixed_matrix(m, k, n)
        left = torch.from_numpy(left_array)
        right = torch.from_numpy(right_array)
        output = torch.empty(m, n, dtype=torch.float32)
        key = f"{m}x{k}x{n}"
        results[key] = measure(
            lambda: torch.mm(left, right, out=output),
            lambda: None,
            args.warmup,
            args.samples,
        )
        results[key]["error"] = error_metrics(output.numpy(), matrix_reference(left_array, right_array))

    return {
        "backend": "torch",
        "version": torch.__version__,
        "device": "cpu",
        "threads": args.threads,
        "process_index": args.process_index,
        "results": results,
    }


def fixed_batch():
    import numpy as np

    random = np.random.default_rng(0)
    return {
        "states": random.standard_normal((PPO_BATCH_SIZE, 8), dtype=np.float32),
        "actions": random.standard_normal((PPO_BATCH_SIZE, 2), dtype=np.float32),
        "old_logprobs": random.standard_normal(PPO_BATCH_SIZE, dtype=np.float32),
        "advantages": random.standard_normal(PPO_BATCH_SIZE, dtype=np.float32),
        "returns": random.standard_normal(PPO_BATCH_SIZE, dtype=np.float32),
    }


def fixed_parameters():
    import numpy as np

    random = np.random.default_rng(1)

    def mlp(input_dim, output_dim, small_output=False):
        dimensions = (input_dim, 128, 128, 128, 128, output_dim)
        layers = []
        for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            weight = random.standard_normal((target, source), dtype=np.float32) * 0.05
            if small_output and index == len(dimensions) - 2:
                weight *= 0.01
            layers.append((weight.astype(np.float32), np.zeros(target, dtype=np.float32)))
        return layers

    return {"actor": mlp(8, 2, True), "critic": mlp(8, 1)}


def torch_parameter_metrics(parameters):
    values = [parameter.detach().double().reshape(-1) for parameter in parameters]
    return {
        "sum": float(sum(value.sum() for value in values)),
        "sumsq": float(sum((value * value).sum() for value in values)),
    }


def encode_parameters(arrays):
    import numpy as np

    flat = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in arrays])
    return {
        "count": int(flat.size),
        "names": parameter_names(),
        "sizes": [int(np.asarray(value).size) for value in arrays],
        "shapes": [list(np.asarray(value).shape) for value in arrays],
        "float32_base64": base64.b64encode(flat.tobytes()).decode("ascii"),
    }


def parameter_names():
    names = [f"actor.layer{index}.{kind}" for index in range(5) for kind in ("weight", "bias")]
    names.append("actor.log_std")
    names.extend(f"critic.layer{index}.{kind}" for index in range(5) for kind in ("weight", "bias"))
    return names


def torch_ppo_worker(args, torch):
    import torch.nn as nn
    import torch.nn.functional as functional

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = self.make_mlp(8, 2)
            self.critic = self.make_mlp(8, 1)
            self.actor[-1].weight.data.mul_(0.01)
            self.actor[-1].bias.data.zero_()
            self.log_std = nn.Parameter(torch.full((2,), -0.7))

        @staticmethod
        def make_mlp(input_dim, output_dim):
            layers = [nn.Linear(input_dim, 128), nn.ReLU()]
            for _ in range(3):
                layers.extend([nn.Linear(128, 128), nn.ReLU()])
            layers.append(nn.Linear(128, output_dim))
            return nn.Sequential(*layers)

        def forward(self, states):
            return self.actor(states), self.log_std.clamp(-5, 2), self.critic(states)

    model = ActorCritic()
    parameters = fixed_parameters()
    with torch.no_grad():
        for module, (weight, bias) in zip((layer for layer in model.actor if isinstance(layer, nn.Linear)), parameters["actor"]):
            module.weight.copy_(torch.from_numpy(weight))
            module.bias.copy_(torch.from_numpy(bias))
        for module, (weight, bias) in zip((layer for layer in model.critic if isinstance(layer, nn.Linear)), parameters["critic"]):
            module.weight.copy_(torch.from_numpy(weight))
            module.bias.copy_(torch.from_numpy(bias))
    actor_parameters = list(model.actor.parameters()) + [model.log_std]
    critic_parameters = list(model.critic.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=1e-3)
    batch = {name: torch.from_numpy(value) for name, value in fixed_batch().items()}
    log_2pi = float(__import__("math").log(2.0 * __import__("math").pi))
    entropy_constant = float(0.5 * __import__("math").log(2.0 * __import__("math").pi * __import__("math").e))

    def train_step():
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        means, log_std, values = model(batch["states"])
        normalized = (batch["actions"] - means) / log_std.exp()
        logprobs = (-0.5 * (normalized.square() + 2.0 * log_std + log_2pi)).sum(dim=-1)
        ratios = (logprobs - batch["old_logprobs"]).exp()
        unclipped = ratios * batch["advantages"]
        clipped = ratios.clamp(0.8, 1.2) * batch["advantages"]
        actor_loss = -torch.minimum(unclipped, clipped).mean()
        critic_loss = functional.mse_loss(values.squeeze(-1), batch["returns"])
        entropy = (log_std + entropy_constant).sum()
        loss = actor_loss + critic_loss - 0.001 * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(critic_parameters, max_norm=0.5)
        actor_optimizer.step()
        critic_optimizer.step()
        return loss

    initial_ms, initial_losses = [], []
    for _ in range(3):
        start = time.perf_counter_ns()
        loss = train_step()
        initial_ms.append((time.perf_counter_ns() - start) / 1e6)
        initial_losses.append(float(loss.detach()))
    validation = {
        "initial_losses": initial_losses,
        "parameters_after_initial": torch_parameter_metrics(actor_parameters + critic_parameters),
        "parameter_values_after_initial": encode_parameters(
            [parameter.detach().numpy() for parameter in actor_parameters + critic_parameters]
        ),
    }
    result = measure(train_step, lambda: None, args.warmup, args.samples)
    result["initial_ms"] = initial_ms
    validation["parameter_values_after_measurement"] = encode_parameters(
        [parameter.detach().numpy() for parameter in actor_parameters + critic_parameters]
    )
    return {
        "backend": "torch",
        "version": torch.__version__,
        "device": "cpu",
        "threads": args.threads,
        "process_index": args.process_index,
        "results": {"ppo_step": result},
        "validation": validation,
    }


def tiny_worker(args):
    os.environ["DEV"] = args.device
    os.environ["NUM_CPU_THREADS"] = str(args.threads)
    os.environ["HCQ2"] = "1" if args.profile_steady or args.hcq2 else "0"

    from tinygrad import Device, GlobalCounters, Tensor, TinyJit
    from tinygrad.engine.realize import run_linear

    if args.test == "ppo":
        return tiny_ppo_worker(args, Device, Tensor, TinyJit)

    ordered_shapes = SHAPES[args.process_index % len(SHAPES):] + SHAPES[:args.process_index % len(SHAPES)]
    results = {}

    for m, k, n in ordered_shapes:
        left_array, right_array = fixed_matrix(m, k, n)
        left = Tensor(left_array).realize()
        right = Tensor(right_array).realize()
        matmul = TinyJit(lambda x, y: (x @ y).realize())

        capture_ms = []
        for _ in range(3):
            start = time.perf_counter_ns()
            matmul(left, right)
            Device[Device.DEFAULT].synchronize()
            capture_ms.append((time.perf_counter_ns() - start) / 1e6)

        key = f"{m}x{k}x{n}"
        results[key] = measure(
            lambda: matmul(left, right),
            lambda: Device[Device.DEFAULT].synchronize(),
            args.warmup,
            args.samples,
        )
        GlobalCounters.reset()
        run_linear(
            matmul.captured.linear,
            input_uops=(left.uop, right.uop),
            update_stats=True,
            jit=True,
            wait=True,
        )
        results[key]["direct_kernel_ms"] = GlobalCounters.time_sum_s * 1e3
        results[key]["initial_ms"] = capture_ms
        actual = matmul(left, right).numpy()
        results[key]["error"] = error_metrics(actual, matrix_reference(left_array, right_array))

    return {
        "backend": "tiny",
        "device": str(Device.DEFAULT),
        "hcq2": args.hcq2,
        "threads": args.threads,
        "process_index": args.process_index,
        "results": results,
    }


def tiny_ppo_worker(args, Device, Tensor, TinyJit):
    from tinygrad import Context, nn
    from tinygrad.helpers import profile_marker

    class MLP:
        def __init__(self, input_dim, output_dim):
            self.layers = [nn.Linear(input_dim, 128)]
            self.layers.extend(nn.Linear(128, 128) for _ in range(3))
            self.layers.append(nn.Linear(128, output_dim))

        def __call__(self, value):
            for layer in self.layers[:-1]:
                value = layer(value).relu()
            return self.layers[-1](value)

    class ActorCritic:
        def __init__(self):
            self.actor = MLP(8, 2)
            self.critic = MLP(8, 1)
            self.actor.layers[-1].weight.assign(self.actor.layers[-1].weight * 0.01).realize()
            self.actor.layers[-1].bias.assign(Tensor.zeros_like(self.actor.layers[-1].bias)).realize()
            self.log_std = Tensor.full((2,), -0.7).realize()

        def __call__(self, states):
            return self.actor(states), self.log_std.clip(-5, 2), self.critic(states)

    def clip_grad_norm(parameters, max_norm):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        total_norm = sum((gradient.square().sum() for gradient in gradients)).sqrt()
        scale = (max_norm / (total_norm + 1e-6)).clip(max_=1.0)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * scale

    model = ActorCritic()
    parameters = fixed_parameters()
    for layer, (weight, bias) in zip(model.actor.layers, parameters["actor"]):
        layer.weight.assign(Tensor(weight)).realize()
        layer.bias.assign(Tensor(bias)).realize()
    for layer, (weight, bias) in zip(model.critic.layers, parameters["critic"]):
        layer.weight.assign(Tensor(weight)).realize()
        layer.bias.assign(Tensor(bias)).realize()
    actor_parameters = nn.state.get_parameters(model.actor) + [model.log_std]
    critic_parameters = nn.state.get_parameters(model.critic)
    optimizer = nn.optim.OptimizerGroup(
        nn.optim.Adam(actor_parameters, lr=1e-3),
        nn.optim.Adam(critic_parameters, lr=1e-3),
    )
    batch = {name: Tensor(value).realize() for name, value in fixed_batch().items()}
    log_2pi = float(__import__("math").log(2.0 * __import__("math").pi))
    entropy_constant = float(0.5 * __import__("math").log(2.0 * __import__("math").pi * __import__("math").e))

    @TinyJit
    @Context(TRAINING=1)
    def train_step(states, actions, old_logprobs, advantages, returns):
        optimizer.zero_grad()
        means, log_std, values = model(states)
        normalized = (actions - means) / log_std.exp()
        logprobs = (-0.5 * (normalized.square() + 2.0 * log_std + log_2pi)).sum(axis=-1)
        ratios = (logprobs - old_logprobs).exp()
        unclipped = ratios * advantages
        clipped = ratios.clip(0.8, 1.2) * advantages
        actor_loss = -unclipped.minimum(clipped).mean()
        critic_loss = (values.squeeze(-1) - returns).square().mean()
        entropy = (log_std + entropy_constant).sum()
        loss = actor_loss + critic_loss - 0.001 * entropy
        loss.backward()
        clip_grad_norm(actor_parameters, 0.5)
        clip_grad_norm(critic_parameters, 0.5)
        return loss.realize(*optimizer.schedule_step())

    def operation():
        return train_step(
            batch["states"], batch["actions"], batch["old_logprobs"],
            batch["advantages"], batch["returns"],
        )

    initial_ms, initial_losses = [], []
    for _ in range(3):
        start = time.perf_counter_ns()
        loss = operation()
        Device[Device.DEFAULT].synchronize()
        initial_ms.append((time.perf_counter_ns() - start) / 1e6)
        initial_losses.append(float(loss.item()))

    parameter_values = [parameter.numpy().astype("float64") for parameter in actor_parameters + critic_parameters]
    validation = {
        "initial_losses": initial_losses,
        "parameters_after_initial": {
            "sum": float(sum(value.sum() for value in parameter_values)),
            "sumsq": float(sum((value * value).sum() for value in parameter_values)),
        },
        "parameter_values_after_initial": encode_parameters(parameter_values),
    }

    if args.profile_steady:
        for _ in range(args.warmup):
            operation()
            Device[Device.DEFAULT].synchronize()
        profile_marker("steady_ppo_start")
        start = time.perf_counter_ns()
        operation()
        Device[Device.DEFAULT].synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        profile_marker("steady_ppo_end")
        result = {"median_ms": elapsed_ms, "p10_ms": elapsed_ms, "p90_ms": elapsed_ms}
    else:
        result = measure(
            operation,
            lambda: Device[Device.DEFAULT].synchronize(),
            args.warmup,
            args.samples,
        )
    result["initial_ms"] = initial_ms
    final_parameter_values = [parameter.numpy() for parameter in actor_parameters + critic_parameters]
    validation["parameter_values_after_measurement"] = encode_parameters(final_parameter_values)
    captured = train_step.captured
    result["graph_calls"] = len(captured.linear.src) if captured is not None else None
    return {
        "backend": "tiny",
        "device": str(Device.DEFAULT),
        "hcq2": args.hcq2,
        "threads": args.threads,
        "process_index": args.process_index,
        "results": {"ppo_step": result},
        "validation": validation,
    }


def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def run_worker(interpreter, script, backend, args, process_index, environment=None, hcq2=None):
    command = [
        str(interpreter), str(script), "--worker", backend,
        "--test", args.test,
        "--threads", str(args.threads),
        "--warmup", str(args.warmup),
        "--samples", str(args.samples),
        "--process-index", str(process_index),
        "--device", args.device,
    ]
    if args.hcq2 if hcq2 is None else hcq2:
        command.append("--hcq2")
    output = subprocess.check_output(command, text=True, env=environment)
    return json.loads(output)


def result_keys(test):
    return ("ppo_step",) if test == "ppo" else tuple(f"{m}x{k}x{n}" for m, k, n in SHAPES)


def summarize(runs, test):
    summary = {"ratio": {}}
    for backend in ("torch", "tiny"):
        backend_runs = [run for run in runs if run["backend"] == backend]
        summary[backend] = {}
        for key in result_keys(test):
            medians = [run["results"][key]["median_ms"] for run in backend_runs]
            summary[backend][key] = {
                "median_ms": statistics.median(medians),
                "process_medians_ms": medians,
            }
    for key in result_keys(test):
        ratios = []
        for process_index in sorted({run["process_index"] for run in runs}):
            torch_ms = next(run for run in runs if run["backend"] == "torch" and run["process_index"] == process_index)["results"][key]["median_ms"]
            tiny_ms = next(run for run in runs if run["backend"] == "tiny" and run["process_index"] == process_index)["results"][key]["median_ms"]
            ratios.append(tiny_ms / torch_ms)
        summary["ratio"][key] = {
            "median": statistics.median(ratios),
            "process_ratios": ratios,
        }
    return summary


def validate_runs(runs, test):
    if any(run["device"].upper().split(":")[0] != "CPU" for run in runs):
        raise RuntimeError("a worker did not use the CPU device")

    if test == "matrices":
        maximum_error = max(
            result["error"]["max_abs"]
            for run in runs
            for result in run["results"].values()
        )
        if maximum_error > 0.01:
            raise RuntimeError(f"matrix correctness failed: max_abs={maximum_error}")
        return {"matrix_max_abs": maximum_error, "passed": True}

    import numpy as np

    pairs = []
    for process_index in sorted({run["process_index"] for run in runs}):
        torch_run = next(run for run in runs if run["process_index"] == process_index and run["backend"] == "torch")
        tiny_run = next(run for run in runs if run["process_index"] == process_index and run["backend"] == "tiny")
        loss_delta = max(
            abs(torch_loss - tiny_loss)
            for torch_loss, tiny_loss in zip(
                torch_run["validation"]["initial_losses"],
                tiny_run["validation"]["initial_losses"],
            )
        )
        torch_sumsq = torch_run["validation"]["parameters_after_initial"]["sumsq"]
        tiny_sumsq = tiny_run["validation"]["parameters_after_initial"]["sumsq"]
        sumsq_relative_delta = abs(torch_sumsq - tiny_sumsq) / abs(torch_sumsq)
        tensor_checks = {}
        for point in ("after_initial", "after_measurement"):
            key = f"parameter_values_{point}"
            torch_encoded, tiny_encoded = torch_run["validation"][key], tiny_run["validation"][key]
            if torch_encoded["count"] != tiny_encoded["count"]:
                raise RuntimeError(f"parameter count differs at {point}")
            torch_values = np.frombuffer(base64.b64decode(torch_encoded["float32_base64"]), dtype=np.float32)
            tiny_values = np.frombuffer(base64.b64decode(tiny_encoded["float32_base64"]), dtype=np.float32)
            difference = torch_values.astype(np.float64) - tiny_values.astype(np.float64)
            offsets = np.cumsum([0] + torch_encoded["sizes"])
            tensor_errors = []
            for tensor_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                tensor_difference = difference[start:end]
                tensor_errors.append({
                    "index": tensor_index,
                    "name": torch_encoded["names"][tensor_index],
                    "shape": torch_encoded["shapes"][tensor_index],
                    "max_abs": float(np.abs(tensor_difference).max()),
                    "rms": float(np.sqrt(np.mean(tensor_difference * tensor_difference))),
                })
            tensor_checks[point] = {
                "count": int(torch_values.size),
                "max_abs": float(np.abs(difference).max()),
                "rms": float(np.sqrt(np.mean(difference * difference))),
                "worst_tensors": sorted(tensor_errors, key=lambda value: value["max_abs"], reverse=True)[:5],
            }
        pairs.append({
            "loss_max_abs": loss_delta,
            "parameter_sumsq_relative": sumsq_relative_delta,
            "full_parameters": tensor_checks,
        })

    if max(pair["loss_max_abs"] for pair in pairs) > 5e-5:
        raise RuntimeError(f"PPO loss parity failed: {pairs}")
    if max(pair["parameter_sumsq_relative"] for pair in pairs) > 1e-4:
        raise RuntimeError(f"PPO update parity failed: {pairs}")
    for run in runs:
        run["validation"].pop("parameter_values_after_initial")
        run["validation"].pop("parameter_values_after_measurement")
    return {"process_pairs": pairs, "passed": True}


def print_table(summary, test):
    print("test                  torch ms   tiny ms   ratio")
    for key in result_keys(test):
        torch_ms = summary["torch"][key]["median_ms"]
        tiny_ms = summary["tiny"][key]["median_ms"]
        print(f"{key:<21} {torch_ms:>8.4f} {tiny_ms:>9.4f} {summary['ratio'][key]['median']:>7.1f}x")


def controller(args):
    script = Path(__file__).resolve()
    tinygrad_path = Path(args.tinygrad_path).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tinygrad_path)
    environment.setdefault("XDG_CACHE_HOME", "/tmp/tinygrad-profile-cache")

    if args.compare_hcq2:
        runs = []
        for process_index in range(args.processes):
            order = (False, True) if process_index % 2 == 0 else (True, False)
            for hcq2 in order:
                if args.cooldown:
                    time.sleep(args.cooldown)
                result = run_worker(
                    args.tiny_python, script, "tiny", args, process_index, environment, hcq2=hcq2
                )
                result["validation"].pop("parameter_values_after_initial")
                result["validation"].pop("parameter_values_after_measurement")
                runs.append(result)
                print(
                    f"process {process_index + 1}/{args.processes} HCQ2={int(hcq2)} complete",
                    file=sys.stderr,
                )
        off = [run["results"]["ppo_step"]["median_ms"] for run in runs if not run["hcq2"]]
        on = [run["results"]["ppo_step"]["median_ms"] for run in runs if run["hcq2"]]
        paired = [
            next(run for run in runs if run["process_index"] == index and run["hcq2"])["results"]["ppo_step"]["median_ms"] /
            next(run for run in runs if run["process_index"] == index and not run["hcq2"])["results"]["ppo_step"]["median_ms"]
            for index in range(args.processes)
        ]
        report = {
            "tinygrad_commit": git_commit(tinygrad_path),
            "protocol": {
                "device": args.device,
                "test": "ppo",
                "alternating_order": True,
                "fresh_process_pairs": args.processes,
                "warmup": args.warmup,
                "samples_per_process": args.samples,
                "cooldown_seconds_before_each_process": args.cooldown,
            },
            "summary": {
                "hcq2_off_median_ms": statistics.median(off),
                "hcq2_on_median_ms": statistics.median(on),
                "paired_on_over_off_median": statistics.median(paired),
                "paired_on_over_off": paired,
            },
            "runs": runs,
        }
        print(json.dumps(report["summary"], indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {args.json}", file=sys.stderr)
        return

    runs = []
    for process_index in range(args.processes):
        order = ("torch", "tiny") if process_index % 2 == 0 else ("tiny", "torch")
        for backend in order:
            if backend == "torch":
                result = run_worker(args.torch_python, script, backend, args, process_index)
            else:
                result = run_worker(args.tiny_python, script, backend, args, process_index, environment)
            runs.append(result)
            print(f"process {process_index + 1}/{args.processes} {backend} complete", file=sys.stderr)

    validation = validate_runs(runs, args.test)
    summary = summarize(runs, args.test)
    report = {
        "protocol": {
            "device": args.device,
            "test": args.test,
            "dtype": "float32",
            "threads": args.threads,
            "warmup": args.warmup,
            "samples_per_process": args.samples,
            "fresh_processes": args.processes,
            "synchronize_each_sample": True,
            "stable_realized_inputs": True,
            "torch_matrix_output_preallocated": args.test == "matrices",
            "tiny_outputs_reused_by_tinyjit": True,
            "hcq2": args.hcq2,
        },
        "tinygrad_commit": git_commit(tinygrad_path),
        "validation": validation,
        "summary": summary,
        "runs": runs,
    }
    print_table(summary, args.test)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("torch", "tiny"))
    parser.add_argument("--test", choices=("matrices", "ppo"), default="matrices")
    parser.add_argument("--torch-python", default=".venv/bin/python")
    parser.add_argument("--tiny-python", default="/opt/homebrew/bin/python3.13")
    parser.add_argument("--tinygrad-path", default="../tinygrad")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--json")
    parser.add_argument("--profile-steady", action="store_true")
    parser.add_argument("--hcq2", action="store_true")
    parser.add_argument("--compare-hcq2", action="store_true")
    parser.add_argument("--cooldown", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker == "torch":
        print(json.dumps(torch_worker(arguments)))
    elif arguments.worker == "tiny":
        print(json.dumps(tiny_worker(arguments)))
    else:
        controller(arguments)
