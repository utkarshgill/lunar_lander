import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BATCHES = (4, 16, 64, 256, 1024)


def measure(call, warmup, samples):
    for _ in range(warmup):
        call()
    times = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        call()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(times)


def torch_worker(args):
    import torch

    os.environ['METAL'] = '1' if args.device == 'metal' else '0'
    import beautiful_lander as lander

    actor_critic = lander.ActorCritic(
        lander.state_dim, lander.action_dim, lander.hidden_dim,
        lander.actor_layers, lander.critic_layers,
    ).to(lander.device)
    results = {}
    for batch in BATCHES:
        states = np.zeros((batch, lander.state_dim), dtype=np.float32)
        state_tensor = torch.from_numpy(states).to(lander.device) * lander.OBS_SCALE_T

        @torch.inference_mode()
        def matched_act():
            tensor = torch.from_numpy(states).to(lander.device) * lander.OBS_SCALE_T
            mean = actor_critic.actor(tensor).cpu().numpy()
            std = actor_critic.log_std.clamp(-5, 2).exp().detach().cpu().numpy()
            sample = np.random.normal(mean, std).astype(np.float32)
            np.clip(sample, -1, 1)

        @torch.inference_mode()
        def device_actor():
            actor_critic.actor(state_tensor)
            if lander.device.type == 'mps':
                torch.mps.synchronize()

        results[str(batch)] = {
            'rollout_act_ms': measure(lambda: actor_critic.act(states), args.warmup, args.samples),
            'matched_act_ms': measure(matched_act, args.warmup, args.samples),
            'device_actor_ms': measure(device_actor, args.warmup, args.samples),
        }
    return {'framework': 'torch', 'device': str(lander.device), 'results': results}


def tiny_worker(args):
    os.environ['METAL'] = '1' if args.device == 'metal' else '0'
    import beautiful_lander_tiny as lander
    from tinygrad import Device, Tensor

    actor_critic = lander.ActorCritic(
        lander.state_dim, lander.action_dim, lander.hidden_dim,
        lander.actor_layers, lander.critic_layers,
    )
    results = {}
    for batch in BATCHES:
        states = np.zeros((batch, lander.state_dim), dtype=np.float32)
        state_tensor = Tensor(states * lander.OBS_SCALE, device=lander.device).realize()

        def device_actor():
            actor_critic.actor_jit(state_tensor).realize()
            Device[lander.device].synchronize()

        results[str(batch)] = {
            'rollout_act_ms': measure(lambda: actor_critic.act(states), args.warmup, args.samples),
            'matched_act_ms': measure(lambda: actor_critic.act(states), args.warmup, args.samples),
            'device_actor_ms': measure(device_actor, args.warmup, args.samples),
        }
    return {'framework': 'tinygrad', 'device': lander.device, 'results': results}


def run_worker(args, framework, device, process_index):
    interpreter = args.torch_python if framework == 'torch' else args.tiny_python
    command = [
        interpreter, __file__, '--worker', framework, '--device', device,
        '--warmup', str(args.warmup), '--samples', str(args.samples),
    ]
    environment = os.environ.copy()
    environment.update(METAL='1' if device == 'metal' else '0', PROCESS_INDEX=str(process_index))
    if framework == 'tiny':
        environment['PYTHONPATH'] = args.tinygrad_path
    output = subprocess.check_output(command, text=True, env=environment)
    result = json.loads(output)
    result['process_index'] = process_index
    return result


def summarize(runs):
    summary = {}
    for framework in ('torch', 'tiny'):
        for device in ('cpu', 'metal'):
            key = f'{framework}_{device}'
            selected = [run for run in runs if run['framework'].startswith(framework) and run['device'].lower() in (device, 'mps')]
            summary[key] = {}
            for batch in BATCHES:
                summary[key][str(batch)] = {
                    metric: statistics.median(run['results'][str(batch)][metric] for run in selected)
                    for metric in ('rollout_act_ms', 'matched_act_ms', 'device_actor_ms')
                }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('torch', 'tiny'))
    parser.add_argument('--device', choices=('cpu', 'metal'), default='cpu')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--torch-python', default='.venv/bin/python')
    parser.add_argument('--tiny-python', default='/opt/homebrew/bin/python3.13')
    parser.add_argument('--tinygrad-path', default='../tinygrad')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.worker == 'torch':
        print(json.dumps(torch_worker(args)))
        return
    if args.worker == 'tiny':
        print(json.dumps(tiny_worker(args)))
        return

    runs = []
    for process_index in range(args.processes):
        for framework, device in (('torch', 'cpu'), ('tiny', 'cpu'), ('torch', 'metal'), ('tiny', 'metal')):
            runs.append(run_worker(args, framework, device, process_index))
            print(f'{framework} {device} process {process_index + 1}/{args.processes}', flush=True)
    report = {
        'protocol': {
            'batches': BATCHES,
            'warmup': args.warmup,
            'samples_per_process': args.samples,
            'fresh_processes': args.processes,
            'rollout_act': 'exact stochastic act path including host-device transfer',
            'matched_act': 'actor-only stochastic act path including host-device transfer',
            'device_actor': 'preloaded input and synchronized actor network only',
        },
        'summary': summarize(runs),
        'runs': runs,
    }
    print(json.dumps(report['summary'], indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
