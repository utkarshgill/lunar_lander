import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

env_name = 'LunarLanderContinuous-v3'
state_dim, action_dim = 8, 2
num_envs = int(os.getenv('NUM_ENVS', 10))

# https://gymnasium.farama.org/environments/box2d/lunar_lander/#:~:text=For%20the%20default%20values%20of%20VIEWPORT_W%2C%20VIEWPORT_H%2C%20SCALE%2C%20and%20FPS%2C%20the%20scale%20factors%20equal%3A%20%E2%80%98x%E2%80%99%3A%2010%2C%20%E2%80%98y%E2%80%99%3A%206.666%2C%20%E2%80%98vx%E2%80%99%3A%205%2C%20%E2%80%98vy%E2%80%99%3A%207.5%2C%20%E2%80%98angle%E2%80%99%3A%201%2C%20%E2%80%98angular%20velocity%E2%80%99%3A%202.5
OBS_SCALE = np.array([10, 6.666, 5, 7.5, 1, 2.5, 1, 1], dtype=np.float32)

max_epochs, max_timesteps, steps_per_epoch = 100, 1000, 50_000
log_interval, eval_interval = 5, 10
batch_size, K_epochs = 128, 10
trunk_dim, hidden_dim = 32, 64
num_hidden_trunk, num_hidden_layers = 1, 3
lr = 3e-4
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
entropy_coef = 0.001
solved_threshold = 250

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
METAL = bool(int(os.getenv('METAL', '0')))

device = torch.device('mps' if METAL and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

if PLOT: import matplotlib.pyplot as plt

def make_env(n, render=False):
    render_mode = 'human' if render else None
    return gym.vector.AsyncVectorEnv([lambda: gym.make(env_name, render_mode=render_mode) for _ in range(n)])

def update_plot(ax, returns, threshold):
    ax.clear()
    ax.plot(returns, alpha=0.3, label='Episode Returns')
    if len(returns) >= 100:
        ma = np.convolve(returns, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(returns)), ma, label='100-ep MA', linewidth=2)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({threshold})')
    ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)

def tanh_log_prob(raw_action, dist):
    # Change of variables for tanh squashing
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, trunk_dim, hidden_dim, num_hidden_trunk, num_hidden_layers):
        super(ActorCritic, self).__init__()
        
        # Shared trunk (configurable depth and width)
        trunk_layers = [nn.LayerNorm(state_dim), nn.Linear(state_dim, trunk_dim), nn.ReLU()]
        for _ in range(num_hidden_trunk - 1):
            trunk_layers.extend([nn.LayerNorm(trunk_dim), nn.Linear(trunk_dim, trunk_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Projection from trunk to heads (if dimensions differ)
        self.trunk_to_actor = nn.Linear(trunk_dim, hidden_dim) if trunk_dim != hidden_dim else nn.Identity()
        self.trunk_to_critic = nn.Linear(trunk_dim, hidden_dim) if trunk_dim != hidden_dim else nn.Identity()
        
        # Actor head (Pre-LN: normalize before compute)
        actor_layers = []
        for _ in range(num_hidden_layers):
            actor_layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),

                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        self.actor_layers = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (Pre-LN: normalize before compute)
        critic_layers = []
        for _ in range(num_hidden_layers):
            critic_layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        trunk_features = self.trunk(state)
        
        # Actor: task-specific processing
        actor_feat = self.actor_layers(self.trunk_to_actor(trunk_features))
        action_mean = self.actor_mean(actor_feat)
        action_std = self.log_std.exp()
        
        # Critic: task-specific processing
        critic_feat = self.critic_layers(self.trunk_to_critic(trunk_features))
        value = self.critic_out(critic_feat)
        
        return action_mean, action_std, value
    
    def preprocess(self, state):
        return torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
    
    @torch.no_grad()
    def act(self, state, deterministic=False, return_internals=False):
        state_tensor = self.preprocess(state)
        action_mean, action_std, _ = self(state_tensor)
        
        if deterministic:
            raw_action = action_mean
            action = torch.tanh(action_mean)
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
        
        if return_internals:
            return action.cpu().numpy(), state_tensor, raw_action
        return action.cpu().numpy()

class PPO:
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, batch_size, entropy_coef):
        self.actor_critic = actor_critic
        self.states, self.actions = [], []
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

    def __call__(self, state):
        action_np, state_tensor, raw_action = self.actor_critic.act(
            state, deterministic=False, return_internals=True
        )
        self.states.append(state_tensor)
        self.actions.append(raw_action)
        return action_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        gae = torch.zeros(N, device=rewards.device)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            advantages[t] = gae
        
        returns = advantages + state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        dist = torch.distributions.Normal(action_means, action_stds)
        action_logprobs = tanh_log_prob(batch_actions, dist)
        state_values = state_values.squeeze(-1)
        
        # PPO actor loss with clipping
        ratios = torch.exp(action_logprobs - batch_logprobs)
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, batch_returns)
        
        # Entropy bonus (maximize entropy = minimize -entropy)
        entropy = dist.entropy().sum(-1).mean()
        
        return actor_loss + critic_loss - self.entropy_coef * entropy
    
    def update(self, rewards, dones):
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(dones), dtype=torch.float32).to(device)

            old_states = torch.cat(self.states)
            old_actions = torch.cat(self.actions)

            # Compute old log probs and values with policy BEFORE updates
            action_means, action_stds, old_state_values = self.actor_critic(old_states)
            dist = torch.distributions.Normal(action_means, action_stds)
            old_logprobs = tanh_log_prob(old_actions, dist)
            old_state_values = old_state_values.squeeze(-1)
            
            # Reshape for GAE computation: (T, N) where T=timesteps, N=num_envs
            N = rewards.size(1)
            old_state_values = old_state_values.view(-1, N)
            
            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns in dataloader:
                loss = self.compute_losses(
                    batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.states, self.actions = [], []

def rollout(env, policy, num_steps):
    states, _ = env.reset()
    traj_rewards, traj_dones, ep_returns = [], [], []
    n = env.num_envs
    ep_rets = np.zeros(n)
    
    for _ in range(num_steps // n):
        actions = policy(states)
        states, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        traj_rewards.append(rewards)
        traj_dones.append(dones)
        ep_rets += rewards
        
        if np.any(dones):
            for idx in np.where(dones)[0]:
                ep_returns.append(ep_rets[idx])
                ep_rets[idx] = 0.0
    
    return traj_rewards, traj_dones, ep_returns

def train_one_epoch(env, ppo):
    rewards, dones, ep_returns = rollout(env, ppo, steps_per_epoch)
    ppo.update(rewards, dones)
    return ep_returns

def train():
    actor_critic = ActorCritic(state_dim, action_dim, trunk_dim, hidden_dim, num_hidden_trunk, num_hidden_layers).to(device)
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, batch_size, entropy_coef)
    env = make_env(num_envs)
    
    all_episode_returns = []
    last_eval = float('-inf')
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    pbar = trange(max_epochs, desc="Training", unit='epoch')
    
    for epoch in range(max_epochs):
        ep_rets = train_one_epoch(env, ppo)
        all_episode_returns.extend(ep_rets)
        pbar.update(1)
        
        train_100 = np.mean(all_episode_returns[-100:]) if all_episode_returns else 0.0
        
        if (epoch + 1) % eval_interval == 0:
            last_eval = evaluate_policy(actor_critic)
            if RENDER:
                evaluate_policy(actor_critic, render=True)
        
        if (epoch + 1) % log_interval == 0:
            s = actor_critic.log_std.exp().detach().cpu().numpy()
            pbar.write(f"Epoch {epoch+1}: n_ep={len(ep_rets)}, ret={np.mean(ep_rets):.1f}±{np.std(ep_rets):.1f}, train_100={train_100:.1f}, eval={last_eval:.1f}, σ=[{s[0]:.2f},{s[1]:.2f}]")
        
        if train_100 >= solved_threshold:  # success = rolling 100-episode average crosses threshold
            pbar.write(f"\n{'='*60}\nSOLVED at epoch {epoch+1}! train_100={train_100:.1f} ≥ {solved_threshold}\n{'='*60}")
            break
        
        if PLOT and (epoch + 1) % (log_interval * 2) == 0:
            update_plot(ax, all_episode_returns, solved_threshold)
    
    env.close()
    pbar.close()
    if PLOT:
        plt.ioff()
        plt.show()

def evaluate_policy(actor_critic, n=16, render=False):
    n = 1 if render else n
    env = make_env(n, render)
    actor_critic.eval()
    
    def policy(state):
        return actor_critic.act(state, deterministic=True)
    
    _, _, ep_rets = rollout(env, policy, max_timesteps * n)
    
    env.close()
    actor_critic.train()
    
    return float(np.mean(ep_rets)) if ep_rets else 0.0

if __name__ == '__main__':
    train()