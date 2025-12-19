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
state_dim = 8
action_dim = 2
num_envs = int(os.getenv('NUM_ENVS', 10))

max_epochs = 500
max_timesteps = 1000
steps_per_epoch = 10_000        # timesteps to collect per epoch before update
log_interval = 1                # log every N epochs
batch_size = 128               
K_epochs = 10                   # squeeze more learning from each rollout                   
hidden_dim = 256
num_hidden_layers = 2
lr_actor = 1e-4               
lr_critic = 5e-4              
gamma = 0.99                  
gae_lambda = 0.95
eps_clip = 0.2
eval_interval = 10              # evaluate every N epochs
solved_threshold = 250

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
METAL = bool(int(os.getenv('METAL', '0')))

# device selection
if METAL and torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using {device} device")

if PLOT: import matplotlib.pyplot as plt

def tanh_log_prob(raw_action, dist):
    """Compute log probability with tanh squashing (change of variables)"""
    action = torch.tanh(raw_action)
    logp_gaussian = dist.log_prob(raw_action).sum(-1)
    return logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layers):
        super(ActorCritic, self).__init__()
        
        # Actor network
        actor_layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            actor_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.actor = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network
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
        """Deterministic action selection (for evaluation)"""
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        mean, _, _ = self(state_tensor)
        return torch.tanh(mean).cpu().numpy()

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, batch_size):
        self.actor_critic = actor_critic
        self.memory = Memory()
        
        # Separate optimizers for actor and critic
        actor_params = list(actor_critic.actor.parameters()) + list(actor_critic.actor_mean.parameters()) + [actor_critic.log_std]
        critic_params = list(actor_critic.critic.parameters()) + list(actor_critic.critic_out.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, state):
        """Sample action from policy and store in memory (for training)"""
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
        action_mean, action_std, _ = self.actor_critic(state_tensor)
        dist = torch.distributions.Normal(action_mean, action_std)
        raw_action = dist.sample()
        
        # store in memory
        self.memory.states.append(state_tensor)
        self.memory.actions.append(raw_action)
        self.memory.logprobs.append(tanh_log_prob(raw_action, dist))
        
        return torch.tanh(raw_action).cpu().numpy()

    def compute_advantages(self, rewards, state_values, is_terminals):
        """Compute GAE advantages and returns"""
        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        gae = torch.zeros(N, device=rewards.device)
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            returns[t] = gae + state_values_pad[t]

        advantages = returns - state_values_pad[:-1]
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
    
    def store_rollout(self, rewards, dones):
        """Store rewards and dones from rollout"""
        self.memory.rewards.extend(rewards)
        self.memory.is_terminals.extend(dones)
    
    def update(self):
        """Update policy from collected experience, then clear memory"""
        with torch.no_grad():
            rewards = torch.as_tensor(np.stack(self.memory.rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(self.memory.is_terminals), dtype=torch.float32).to(device)

            old_states = torch.cat(self.memory.states)
            old_actions = torch.cat(self.memory.actions)
            old_logprobs = torch.cat(self.memory.logprobs)

            _, _, old_state_values = self.actor_critic(old_states)
            old_state_values = old_state_values.squeeze(-1)

            if rewards.dim() > 1:
                N = rewards.size(1)
                old_state_values_reshaped = old_state_values.view(-1, N)
            else:
                old_state_values_reshaped = old_state_values

            advantages, returns = self.compute_advantages(rewards, old_state_values_reshaped, is_terms)

        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for _ in range(self.K_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns = batch
                
                actor_loss, critic_loss = self.compute_losses(
                    batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns
                )
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor_optimizer.param_groups[0]['params'], max_norm=0.5)
                self.actor_optimizer.step()

                # Update critic  
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_optimizer.param_groups[0]['params'], max_norm=1.0)
                self.critic_optimizer.step()
        
        # clear memory after update
        self.memory.clear()

def rollout(env, policy, num_steps, num_envs):
    """Collect trajectories from vectorized envs (like tiny_reinforce but vectorized)"""
    states, _ = env.reset()
    traj_rewards, traj_dones = [], []
    ep_returns = []
    ep_rets = np.zeros(num_envs)
    timestep = 0
    
    while timestep < num_steps:
        timestep += num_envs
        actions = policy(states)
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        traj_rewards.append(rewards)
        traj_dones.append(dones)
        ep_rets += rewards
        
        if np.any(dones):
            for idx in np.where(dones)[0]:
                ep_returns.append(ep_rets[idx])
                ep_rets[idx] = 0.0
            
            if hasattr(env, 'reset_done'):
                next_states[dones] = env.reset_done()[0]
            else:
                next_states, _ = env.reset()
                ep_rets[:] = 0.0
        
        states = next_states
    
    return (traj_rewards, traj_dones), ep_returns

def train_one_epoch(env, ppo, steps_per_epoch, num_envs):
    """Run one epoch: collect experience → update policy"""
    (rewards, dones), ep_returns = rollout(env, ppo, steps_per_epoch, num_envs)
    ppo.store_rollout(rewards, dones)
    ppo.update()
    return ep_returns

def train():
    # setup
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, num_hidden_layers).to(device)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, batch_size)
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    
    all_episode_returns = []
    last_eval = float('-inf')
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    pbar = trange(max_epochs, desc="Training", unit='epoch')
    pbar.write("epoch | n_ep | ep_ret_mean | ep_ret_std | train_100 |    eval |  std")
    pbar.write("-" * 75)
    
    # main training loop
    for epoch in range(max_epochs):
        epoch_returns = train_one_epoch(env, ppo, steps_per_epoch, num_envs)
        all_episode_returns.extend(epoch_returns)
        pbar.update(1)
        
        # logging
        n_ep = len(epoch_returns)
        ep_mean = np.mean(epoch_returns) if epoch_returns else 0.0
        ep_std = np.std(epoch_returns) if epoch_returns else 0.0
        train_100 = np.mean(all_episode_returns[-100:]) if all_episode_returns else 0.0
        
        if (epoch + 1) % eval_interval == 0:
            eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, num_envs=num_envs)
            last_eval = eval_ret
            std = actor_critic.log_std.exp().detach().cpu().numpy()
            pbar.write(f"{epoch+1:5d} | {n_ep:4d} | {ep_mean:11.2f} | {ep_std:10.2f} | {train_100:9.2f} | {eval_ret:7.2f} | [{std[0]:.2f},{std[1]:.2f}]")
            
            if RENDER:
                evaluate_policy(env_name, actor_critic, max_timesteps, num_envs=1, render=True)
            
            if train_100 >= solved_threshold:
                pbar.write(f"{'='*75}\nSOLVED at epoch {epoch+1}! train_100={train_100:.2f}, eval={eval_ret:.2f}\n{'='*75}")
                break
        
        elif (epoch + 1) % log_interval == 0:
            pbar.write(f"{epoch+1:5d} | {n_ep:4d} | {ep_mean:11.2f} | {ep_std:10.2f} | {train_100:9.2f} | {last_eval:7.2f}")
        
        if PLOT and (epoch + 1) % (log_interval * 2) == 0:
            ax.clear()
            ax.plot(all_episode_returns, alpha=0.3, label='Episode Returns')
            if len(all_episode_returns) >= 100:
                ma = np.convolve(all_episode_returns, np.ones(100)/100, mode='valid')
                ax.plot(range(99, len(all_episode_returns)), ma, label='100-ep MA', linewidth=2)
            ax.axhline(solved_threshold, color='red', linestyle='--', alpha=0.5, label=f'Solved ({solved_threshold})')
            ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
    
    env.close()
    pbar.close()
    if PLOT:
        plt.ioff()
        plt.show()

def evaluate_policy(env_name, actor_critic, max_timesteps, num_envs=16, render=False):
    """Evaluate policy deterministically using rollout abstraction"""
    render_mode = 'human' if render else None
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name, render_mode=render_mode) for _ in range(num_envs)])
    actor_critic.eval()
    
    _, ep_rets = rollout(env, actor_critic.act_deterministic, max_timesteps * num_envs, num_envs)
    
    env.close()
    actor_critic.train()
    
    if not ep_rets:
        return 0.0
    # average first num_envs episodes (one per env)
    return float(np.mean(ep_rets[:num_envs]))

if __name__ == '__main__':
    train()