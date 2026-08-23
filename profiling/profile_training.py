import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


CASES = {
    'torch_cpu': ('beautiful_lander.py', '0'),
    'tiny_cpu': ('beautiful_lander_tiny.py', '0'),
    'torch_metal': ('beautiful_lander.py', '1'),
    'tiny_metal': ('beautiful_lander_tiny.py', '1'),
}


def run_case(name, output_dir, torch_python, tiny_python, tinygrad_path):
    script, metal = CASES[name]
    interpreter = tiny_python if name.startswith('tiny_') else torch_python
    environment = os.environ.copy()
    environment.update(NUM_ENVS='4', PLOT='0', RENDER='0', METAL=metal)
    if name.startswith('tiny_'):
        environment['PYTHONPATH'] = tinygrad_path
    start = time.perf_counter()
    process = subprocess.run(
        [interpreter, script],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{name}.log').write_text(process.stdout)
    match = re.search(r'SOLVED at epoch (\d+)! eval_stoch=([\d.-]+)', process.stdout)
    result = {
        'case': name,
        'seconds': elapsed,
        'exit_code': process.returncode,
        'solved': match is not None,
        'solved_epoch': int(match.group(1)) if match else None,
        'eval_stoch': float(match.group(2)) if match else None,
    }
    (output_dir / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if process.returncode:
        raise RuntimeError(f'{name} failed; see {output_dir / f"{name}.log"}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=tuple(CASES) + ('all',), default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('profiling/full_training'))
    parser.add_argument('--torch-python', default='.venv/bin/python')
    parser.add_argument('--tiny-python', default='/opt/homebrew/bin/python3.13')
    parser.add_argument('--tinygrad-path', default='../tinygrad')
    args = parser.parse_args()
    names = CASES if args.case == 'all' else (args.case,)
    for name in names:
        run_case(
            name, args.output_dir, args.torch_python,
            args.tiny_python, args.tinygrad_path,
        )
    results = [
        json.loads((args.output_dir / f'{name}.json').read_text())
        for name in CASES
        if (args.output_dir / f'{name}.json').exists()
    ]
    (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
