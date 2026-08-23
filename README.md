sub 500 line PPO agent lands on the moon in < 500 episodes, reliably.

## setup

Use Python 3.11 on macOS or Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python beautiful_lander.py
```

For Tinygrad, install `clang` and the tested commit:

```bash
python -m pip install "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@969df866a3ca21b6e6efb739486d094c2604d31b"
NUM_ENVS=4 python beautiful_lander_tiny.py
```

## performance

Both four-environment trainers solved Lunar Lander with stochastic evaluation.

Tests used float32 on an Apple M4. Each result is the median of five paired fresh processes.

| Test | PyTorch | tinygrad | Ratio |
|---|---:|---:|---:|
| Complete CPU PPO update | 10.582 ms | 234.562 ms | 22.1× |
| CPU `(10000×128) @ (128×128)` | 0.2260 ms | 6.2470 ms | 27.7× |
| Metal `(10000×128) @ (128×128)` | 0.4273 ms | 0.9002 ms | 1.87× |

The CPU gap is inside or near tinygrad's generated dense matrix kernels. The Metal gap is much smaller.

One complete four-environment training run gave these wall times. Each run stopped when stochastic evaluation reached 200.

| Framework | Device | Training time | Solved epoch | Stochastic score | Mean per epoch |
|---|---|---:|---:|---:|---:|
| PyTorch | CPU | 1m 26.5s | 25 | 247.9 | 3.33s |
| tinygrad | CPU | 11m 48.2s | 25 | 244.0 | 27.24s |
| PyTorch | Metal | 19m 16.1s | 25 | 216.2 | 44.47s |
| tinygrad | Metal | 5m 41.6s | 20 | 204.9 | 16.27s |

These are single stochastic training runs, not paired medians. The four runs used Gymnasium 1.1.1 and the same stopping condition.

The PyTorch rollout also computes the unused critic. See [`profiling/inference_profile.json`](profiling/inference_profile.json) for the matched actor-only comparison.

[Profiling notes and generated kernels](https://utkarshgill.github.io/blog/tinygrad_lander.html)

<img width="1550" height="1126" alt="image" src="https://github.com/user-attachments/assets/0f0bb9ff-f2b3-4ff2-ba2e-f3d56fc4ca32" />
