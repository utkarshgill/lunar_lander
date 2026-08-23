import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = Path('/tmp/lunar-direct-kernels')
SOURCE = ROOT / 'profiling/cpu_kernel_harness.c'
VARIANTS = {
    'tiny_o2': ('-O2',),
    'tiny_o3': ('-O3',),
    'tiny_o3_fast': ('-O3', '-ffast-math'),
}


def compile_variants():
    BUILD.mkdir(parents=True, exist_ok=True)
    binaries = {}
    common = [
        'clang', '-std=c11', '-mcpu=native', '-ffixed-x18', '-fno-math-errno',
        '-I', str(ROOT / 'profiling'), '-framework', 'Accelerate', '-lm',
    ]
    for name, flags in VARIANTS.items():
        binary = BUILD / name
        subprocess.run(common + list(flags) + [str(SOURCE), '-o', str(binary)], check=True)
        binaries[name] = binary
    return binaries


def execute(binary, mode, single_thread):
    environment = os.environ.copy()
    if single_thread:
        environment['VECLIB_MAXIMUM_THREADS'] = '1'
    return json.loads(subprocess.check_output([str(binary), mode], text=True, env=environment))


def summarize(runs):
    summary = {}
    for key in sorted({run['key'] for run in runs}):
        selected = [run for run in runs if run['key'] == key]
        summary[key] = {
            field: statistics.median(run[field] for run in selected)
            for field in ('median_ms', 'gflops', 'max_abs', 'rms')
        }
        summary[key]['process_medians_ms'] = [run['median_ms'] for run in selected]
    return summary


def main():
    binaries = compile_variants()
    runs = []
    cases = [
        ('cblas_default', binaries['tiny_o2'], 'cblas', False),
        ('cblas_single', binaries['tiny_o2'], 'cblas', True),
        ('cblas_column', binaries['tiny_o2'], 'cblas_column', False),
        *[(name, binary, 'tiny', False) for name, binary in binaries.items()],
    ]
    for process_index in range(5):
        order = cases if process_index % 2 == 0 else tuple(reversed(cases))
        for key, binary, mode, single_thread in order:
            result = execute(binary, mode, single_thread)
            result.update(key=key, process_index=process_index)
            runs.append(result)
            print(f'{key} process {process_index + 1}/5', flush=True)
    report = {
        'protocol': {
            'shape': [[10000, 128], [128, 128]],
            'dtype': 'float32',
            'alignment_bytes': 64,
            'warmup': 10,
            'samples_per_process': 50,
            'fresh_processes': 5,
            'alternating_case_order': True,
            'compiler': 'Apple clang 21',
            'tiny_flags': {name: list(flags) for name, flags in VARIANTS.items()},
        },
        'summary': summarize(runs),
        'runs': runs,
    }
    destination = ROOT / 'profiling/direct_kernel_profile.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
