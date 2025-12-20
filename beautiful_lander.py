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

max_epochs, max_timesteps, steps_per_epoch = 200, 1000, 10_000
log_interval, eval_interval = 10, 20
batch_size, K_epochs = 128, 10
hidden_dim, num_hidden_layers = 256, 2
lr_actor, lr_critic = 1e-4, 5e-4
gamma, gae_lambda, eps_clip = 0.99, 0.95, 0.2
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
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layers):
        super(ActorCritic, self).__init__()
        
        # Actor
        actor_layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            actor_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.actor = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        critic_layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            critic_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.critic = nn.Sequential(*critic_layers)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        action_mean = self.actor_mean(self.actor(state))
        action_std = self.log_std.exp()
        value = self.critic_out(self.critic(state))
        return action_mean, action_std, value
    
    @torch.no_grad()
    def act_deterministic(self, state):
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
        mean, _, _ = self(state_tensor)
        return torch.tanh(mean).cpu().numpy()

class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, batch_size):
        self.actor_critic = actor_critic
        self.states, self.actions = [], []
        
        # Separate optimizers for actor and critic
        actor_params = [*actor_critic.actor.parameters(), *actor_critic.actor_mean.parameters(), actor_critic.log_std]
        critic_params = [*actor_critic.critic.parameters(), *actor_critic.critic_out.parameters()]
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, state):
        state_tensor = torch.as_tensor(state * OBS_SCALE, dtype=torch.float32).to(device)
        action_mean, action_std, _ = self.actor_critic(state_tensor)
        dist = torch.distributions.Normal(action_mean, action_std)
        raw_action = dist.sample()
        
        self.states.append(state_tensor)  # logprobs computed later in update()
        self.actions.append(raw_action)
        
        return torch.tanh(raw_action).cpu().numpy()

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
        state_values = state_values.squeeze()
        
        # PPO actor loss with clipping
        ratios = torch.exp(action_logprobs - batch_logprobs.detach())
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, batch_returns)
        
        return actor_loss, critic_loss
    
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
                actor_loss, critic_loss = self.compute_losses(
                    batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
                )
                
                # Single backward pass, separate optimizer steps
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_optimizer.param_groups[0]['params'], max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_optimizer.param_groups[0]['params'], max_norm=1.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.states, self.actions = [], []

def rollout(env, policy, num_steps):
    states, _ = env.reset()
    traj_rewards, traj_dones, ep_returns = [], [], []
    n = env.num_envs
    ep_rets = np.zeros(n)
    
    for _ in range(0, num_steps, n):
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
    
    return (traj_rewards, traj_dones), ep_returns

def train_one_epoch(env, ppo):
    (rewards, dones), ep_returns = rollout(env, ppo, steps_per_epoch)
    ppo.update(rewards, dones)
    return ep_returns

def train():
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, num_hidden_layers).to(device)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, batch_size)
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
    
    _, ep_rets = rollout(env, actor_critic.act_deterministic, max_timesteps * n)
    
    env.close()
    actor_critic.train()
    
    return float(np.mean(ep_rets)) if ep_rets else 0.0

if __name__ == '__main__':
    train()