import torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as distributions
import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")#, render_mode="human")

# hyperparams
INPUTS = env.observation_space.shape[0]
OUTPUTS = env.action_space.n
CHANNELS = 128
BUFFER_SIZE = 100
EPOCHS = 10000
LEARN_RATE = 0.0005
DISCOUNT = 0.995
WINDOW = 50

plt.ion()

def rtg(rewards):
    N = len(rewards)
    rtg = np.zeros_like(rewards)
    for i in reversed(range(N)):
        rtg[i] = rewards[i] + DISCOUNT * (rtg[i+1] if i+1<N else 0)
    return list(rtg)

def plot():

    print(f'return {all_rets[-1]:6.2f}          avg {mean_rets[-1]:6.2f}          max {max_rets[-1]:6.2f}')
    # plt.subplot(2, 1, 1)
    plt.plot(all_rets)
    plt.plot(max_rets)
    plt.plot(mean_rets)
    plt.title('PERFORMANCE')
    plt.xlabel('episodes')
    plt.ylabel('returns')

    # plt.subplot(2, 1, 2)
    # plt.plot(critic_losses)
    # plt.title('VALUE LOSS')
    # plt.xlabel('episodes')
    # plt.ylabel('loss')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.2)

    plt.clf()

class DQN(nn.Module):
    def __init__(self, in_dim, channels, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, out_dim)
        )

    def forward(self, state):
        out = self.block(state)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        return distributions.Categorical(logits= self.actor(state)), self.critic(state)
        
    def act(self, state):
        policy, value = self(state)
        action = policy.sample().item()
        return action, value

   

   
actor = DQN(INPUTS, CHANNELS, OUTPUTS)
critic = DQN(INPUTS, CHANNELS, 1)
agent = ActorCritic(actor, critic)
optim = torch.optim.Adam(agent.parameters(), LEARN_RATE)

#plot variables
max_rets = []
all_rets = []
mean_rets = []
critic_losses = []

def compute_loss(states, actions, weights):
    global critic_losses
    policy, values = agent(states)
    logp = policy.log_prob(actions)
    actor_loss =  - (logp * (weights-values)).mean()
    critic_loss = F.mse_loss(values, weights)

    # critic_losses.append(critic_loss.detach())
    # plot()

    loss = actor_loss + critic_loss
    return loss

def train_one_epoch():
    global all_rets
    global max_rets
    global mean_rets

    batch_states = []
    batch_actions = []
    batch_values = []
    batch_weights = []

    ep_rews = []
    state, _ = env.reset()
    while 1:
        batch_states.append(state)

        action, value = agent.act(torch.as_tensor(state, dtype=torch.float32))
        state, reward, done, trunc, _ = env.step(action)

        batch_values.append(value)
        batch_actions.append(action)
        ep_rews.append(reward)

        if done or trunc:

            ep_len, ep_ret = len(ep_rews), sum(ep_rews)
            # agent.record('lengths', ep_len)
            # agent.record('returns', ep_ret)
            all_rets.append(ep_ret)
            mean_rets.append(np.mean(all_rets.copy()) if len(all_rets) < WINDOW else np.mean(all_rets.copy()[-WINDOW:]))
            max_rets.append(max_rets[-1] if len(max_rets) > 0 and ep_ret < max_rets[-1] else ep_ret)

            plot()

            batch_weights += rtg(ep_rews)
            state, _ = env.reset()
            ep_rews = []
            if len(batch_states) > BUFFER_SIZE:
                break

    loss = compute_loss(torch.tensor(np.array(batch_states), dtype=torch.float32), 
                        torch.tensor(np.array(batch_actions), dtype=torch.int32), 
                        torch.tensor(np.array(batch_weights), dtype=torch.float32),)
        
    optim.zero_grad(set_to_none=1)
    loss.backward()
    optim.step() 

for ep in range(EPOCHS):
    train_one_epoch()

plt.ioff()
plt.plot(all_rets)
plt.plot(max_rets)
plt.plot(mean_rets)
plt.show()