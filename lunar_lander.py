#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import random
import time

mps_device = (
    torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
)

plt.ion()
max_ret = 0
max_rets, ep_rets, run_avg = [], [], []
window = 100

# hyperparameters
N_HIDDEN = 256
EPISODES = 10000
DISCOUNT = 0.995
# EPOCHS = 10
BUFFER_SIZE = 50
# BATCH_SIZE = 1000000
TRAIN_STEPS = 1
# EPS_START = 0.95
# EPS_END = 0.05
# EPS_DECAY = 200


class DQN(nn.Module):
    def __init__(self, fan_in, fan_out, n_hidden):
        self.fan_in = fan_in
        self.fan_out = fan_out
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Linear(fan_in, n_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.ReLU(),
                nn.Linear(n_hidden, fan_out, bias=True),
            ]
        )

        self.optim = torch.optim.AdamW(self.parameters())

    def forward(self, x):
        if self.fan_out == 1:
            return torch.exp(self.block(x))
        else:
            logits = self.block(x)
            return Categorical(logits=logits)


class Agent:
    def __init__(self):
        super().__init__()
        self.policy = DQN(8, 4, N_HIDDEN)
        self.policy.to(mps_device)
        # self.value = DQN(8, 1, N_HIDDEN)
        # self.value.to(mps_device)
        self.history = []
        self.max_rew = 0

    def act(self, obs):
        return self.policy(obs).sample().item()  # , self.value(obs).item()

    def record(self, episode):
        self.history.append(episode)

    def compute_loss(self, obs, act, wei):
        logp = self.policy(obs).log_prob(act)
        return (-logp * wei).mean()

    def update(self, obs, act, wei, iters, rtg):
        for _ in range(iters):
            # self.update_value(val_preds, rtg)
            loss = self.compute_loss(obs, act, wei)
            self.policy.optim.zero_grad()
            loss.backward()
            self.policy.optim.step()
        # self.history.clear()

    # def update_value(self, val_preds, rtg):
    #     val_loss = F.mse_loss(val_preds, rtg)
    #     val_loss.requires_grad = True
    #     self.value.optim.zero_grad()
    #     val_loss.backward()
    #     self.value.optim.step()
    #     return val_loss


lander = Agent()


# plt.figure(figsize=(16, 4))
# fig, ax = plt.subplots()

# ax.plot(0, 0, label="returns")
# ax.plot(0, 0, label="avg returns")
# ax.plot(0, 0, label="max")

# ax.set_xlabel("episodes")
# ax.set_ylabel("rewards")
# ax.legend()


def plot_results(x, y):
    # clear_output(wait=True)

    # max = [np.max(y)] * len(x)
    cnt = len(x)

    if cnt > window:
        run_avg.append(np.mean(x[-window:]))

    # Update the plot with the new data
    ax.lines[0].set_ydata(x)
    ax.lines[1].set_ydata(run_avg)
    ax.lines[2].set_ydata(y)
    plt.draw()

    # Wait for a short period of time before updating again
    plt.pause(0.01)
    # print(f'ep_ret {x[-1]:6.2f} | max {max[-1]:6.2f}')

    # plt.text(-10, max[-1], f'{max[-1]:.2f}', color='orange', fontsize=12, ha='center', va='center')
    # plt.text(-10, x[-1], f'{x[-1]:.2f}', color='blue', fontsize=12, ha='center', va='center')

    # plt.show()
    # plt.draw()
    # plt.pause(1)
    # plt.clf()
    # time.sleep(1)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + DISCOUNT * (rtgs[i + 1] if i + 1 < n else 0)
    return list(rtgs)


def train():
    env = gym.make("LunarLander-v2")
    observation, info = env.reset()
    global max_ret
    global ep_rets

    batch_obs = []
    batch_act = []
    batch_wei = []
    val_loss_sum = 0

    # rollouts
    for ep in (t := trange(EPISODES)):
        ep_rews = []
        # ep_vals = []

        while True:
            batch_obs.append(observation.copy())

            action = lander.act(torch.tensor(observation, device=mps_device))
            observation, reward, terminated, truncated, info = env.step(action)

            batch_act.append(action)
            # ep_vals.append(value)
            ep_rews.append(reward)

            if terminated or truncated:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                ep_rets.append(ep_ret)
                max_rets.append(max(ep_rets))

                max_ret = max(ep_rets)
                # plot_results(ep_rets, max_rets)

                rtg = torch.tensor(
                    reward_to_go(ep_rews),
                    dtype=torch.float32,
                )
                # val_preds = torch.tensor(ep_vals, device=mps_device)
                # val_loss = lander.update_value(val_preds, rtg)
                t.set_description(f"ep {ep:8} return {ep_ret:8.2f} max {max_ret:8.2f}")

                batch_wei += rtg
                # print(len(batch_wei), len(batch_obs), len(batch_act))
                # batch_wei += [ep_ret] * ep_len
                # print(len(batch_wei), len(batch_obs), len(batch_act))
                # ep_vals = []
                ep_rews = []

                observation, info = env.reset()

                break

        if ep % BUFFER_SIZE == 0:
            lander.update(
                torch.tensor(np.array(batch_obs), device=mps_device),
                torch.tensor(np.array(batch_act), dtype=torch.int32, device=mps_device),
                torch.tensor(
                    np.array(batch_wei), dtype=torch.float32, device=mps_device
                ),
                TRAIN_STEPS,
                # val_preds,
                rtg,
            )
            batch_obs = []
            batch_act = []
            batch_wei = []

    env.close()


def eval():
    wins = 0
    eval_eps = 100
    env = gym.make("LunarLander-v2")
    obs, info = env.reset()
    for _ in (t := trange(eval_eps)):
        ep_rew = 0
        while True:
            act = lander.act(torch.tensor(obs))
            obs, rew, done, trunc, info = env.step(act)
            ep_rew += rew
            if done:
                if ep_rew > 200:
                    wins += 1
                ep_rew = 0
                obs, info = env.reset()
                break
        t.set_description(f"accuracy {wins} / {eval_eps}")
    env.close()


train()
eval()

plt.plot(ep_rets)
plt.plot(max_rets)
plt.draw()
