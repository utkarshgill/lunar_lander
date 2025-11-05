sub 500 line PPO agent lands on the moon in < 500 episodes, reliably. read [how](https://utkarshgill.github.io/blog/lander.html)

## prerequisites

pygame needs the SDL2 headers to build when using the `gymnasium[box2d]` extra. Install them once before creating the virtual environment:

- macOS (Homebrew)
  ```bash
  brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
  ```
- Ubuntu / Debian
  ```bash
  sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  ```
## make env, install deps and run

```python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python beautiful_lander.py
```

<img width="1550" height="1126" alt="image" src="https://github.com/user-attachments/assets/0f0bb9ff-f2b3-4ff2-ba2e-f3d56fc4ca32" />
