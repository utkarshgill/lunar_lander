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

<img width="1550" height="1126" alt="image" src="https://github.com/user-attachments/assets/0f0bb9ff-f2b3-4ff2-ba2e-f3d56fc4ca32" />
