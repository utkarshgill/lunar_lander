import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

device = (
    torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
)

train_env = gym.make("LunarLander-v2")
test_env = gym.make("LunarLander-v2")


class DQN(nn.Module):
    def __init__(self, fan_in, n_hidden, fan_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(fan_in, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, fan_out, bias=True),
        )

        self.optim = torch.optim.AdamW(self.parameters())

    def forward(self, x):
        return self.block(x)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n

actor = DQN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = DQN(INPUT_DIM, HIDDEN_DIM, 1)

lander = ActorCritic(actor, critic)
lander.to(device)

# LEARNING_RATE = 0.0005
optim = torch.optim.AdamW(lander.parameters())


def train(env, policy, optim, discount, ppo_steps, ppo_clip, batch_size):
    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    ep_reward = 0

    state, info = train_env.reset()

    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        states.append(state)

        action_pred, value_pred = policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        state, reward, done, trunc, info = env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        ep_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calc_returns(rewards, discount)
    advantages = calc_advantages(returns, values)

    policy_loss, value_loss = update_policy(
        policy,
        states,
        actions,
        log_prob_actions,
        advantages,
        returns,
        optim,
        ppo_steps,
        ppo_clip,
        batch_size,
    )

    return policy_loss, value_loss, ep_reward


def calc_advantages(returns, values, normalize=True):
    advs = returns - values
    if normalize:
        advs = (advs - advs.mean()) / advs.std()
    return advs.float()


def calc_returns(rewards, discount, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def update_policy(
    policy,
    states,
    actions,
    log_prob_actions,
    advantages,
    returns,
    optim,
    ppo_steps,
    ppo_clip,
    batch_size,
):
    total_policy_loss = 0
    total_value_loss = 0

    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    # print(advantages.dtype)

    for _ in range(ppo_steps):
        # print(states.shape[0])
        # samp = np.random.randint(0, states.shape[0], (batch_size,))

        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = torch.distributions.Categorical(action_prob)

        # new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)

        # print(new_log_prob_actions.dtype)

        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = (
            torch.clip(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip)
            + advantages
        )

        # print(value_pred.dtype, returns.dtype)

        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = F.mse_loss(returns, value_pred).mean().float()

        # print(policy_loss.dtype, value_loss.dtype)

        optim.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optim.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def eval(env, policy):
    policy.eval()

    rewards = []
    done = False
    ep_reward = 0

    state, info = env.reset()

    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_pred, value_pred = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)

        state, reward, done, trunc, info = env.step(action.item())

        ep_reward += reward

    return ep_reward


MAX_EPS = 1000
BATCH_SIZE = 32
DISCOUNT = 0.99
N_TRIALS = 25
REWARD_TARGET = 200
PRINT_EVERY = 10
PPO_STEPS = 3
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []

for episode in (t := trange(1, MAX_EPS + 1)):
    policy_loss, value_loss, train_reward = train(
        train_env, lander, optim, DISCOUNT, PPO_STEPS, PPO_CLIP, BATCH_SIZE
    )
    test_reward = eval(test_env, lander)

    # train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    # mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

    t.set_description(f"ep {episode:3} test {mean_test_rewards:7.1f} |")

    if mean_test_rewards >= REWARD_TARGET:
        print(f"Reached reward threshold in {episode} episodes")
        break

plt.figure(figsize=(12, 8))
plt.plot(test_rewards, label="test")
plt.plot(train_rewards, label="train")
plt.xlabel("episode", fontsize=20)
plt.ylabel("reward", fontsize=20)
plt.hlines(REWARD_TARGET, 0, len(test_rewards), color="r")
plt.legend(loc="lower right")
plt.grid()
