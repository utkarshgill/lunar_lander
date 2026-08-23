#!/usr/bin/env python3
"""Compare tinygrad default and beam-selected CPU schedules."""

import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiling/profile_cpu.py"
TINYGRAD = Path(os.getenv("TINYGRAD_PATH", ROOT.parent / "tinygrad")).resolve()
PYTHON = os.getenv("TINY_PYTHON", "/opt/homebrew/bin/python3.13")
PROCESSES = 5


def run_worker(test, beam, process_index):
    environment = os.environ.copy()
    environment.update({
        "BEAM": str(beam),
        "DEV": "CPU",
        "PYTHONPATH": str(TINYGRAD),
        "XDG_CACHE_HOME": f"/tmp/tinygrad-cpu-beam{beam}-cache",
    })
    samples = 30 if test == "matrices" else 10
    command = [
        PYTHON, str(PROFILE), "--worker", "tiny", "--test", test,
        "--device", "CPU", "--threads", "1", "--warmup", "5",
        "--samples", str(samples), "--process-index", str(process_index),
    ]
    return json.loads(subprocess.check_output(command, text=True, env=environment))


def metric(run, test, key):
    result_key = key if test == "matrices" else "ppo_step"
    return run["result"]["results"][result_key]["median_ms"]


def summarize(runs, test, key):
    selected = [run for run in runs if run["test"] == test]
    default = [metric(run, test, key) for run in selected if run["beam"] == 0]
    beam = [metric(run, test, key) for run in selected if run["beam"] == 1]
    ratios = []
    for process_index in range(PROCESSES):
        default_run = next(run for run in selected if run["beam"] == 0 and run["process_index"] == process_index)
        beam_run = next(run for run in selected if run["beam"] == 1 and run["process_index"] == process_index)
        ratios.append(metric(default_run, test, key) / metric(beam_run, test, key))
    return {
        "default_median_ms": statistics.median(default),
        "beam1_median_ms": statistics.median(beam),
        "paired_speedup_median": statistics.median(ratios),
        "paired_speedups": ratios,
        "default_process_medians_ms": default,
        "beam1_process_medians_ms": beam,
    }


def main():
    runs = []
    for test in ("matrices", "ppo"):
        for process_index in range(PROCESSES):
            order = (0, 1) if process_index % 2 == 0 else (1, 0)
            for beam in order:
                result = run_worker(test, beam, process_index)
                if "validation" in result:
                    result["validation"].pop("parameter_values_after_initial", None)
                    result["validation"].pop("parameter_values_after_measurement", None)
                runs.append({"test": test, "beam": beam, "process_index": process_index, "result": result})
                print(f"{test} process {process_index + 1}/{PROCESSES} beam={beam}", flush=True)
    report = {
        "protocol": {
            "device": "CPU",
            "tinygrad_commit": "5b60a09ab0a7f20c3426e505251c48cac020d1fa",
            "fresh_process_pairs": PROCESSES,
            "alternating_schedule_order": True,
            "matrix_samples": 30,
            "ppo_samples": 10,
            "warmup": 5,
        },
        "summary": {
            "forward_matrix": summarize(runs, "matrices", "10000x128x128"),
            "weight_gradient_matrix": summarize(runs, "matrices", "128x10000x128"),
            "ppo_step": summarize(runs, "ppo", "ppo_step"),
        },
        "runs": runs,
    }
    destination = ROOT / "profiling/cpu_schedule_profile.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
