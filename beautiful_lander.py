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

max_episodes = 1000
max_timesteps = 1000
update_timestep = 10_000          # more data per update
log_interval = 10               
batch_size = 128               
K_epochs = 10                   # squeeze more learning from each rollout                   
hidden_dim = 256
num_hidden_layers = 2
lr_actor = 1e-4               
lr_critic = 5e-4              
gamma = 0.99                  
gae_lambda = 0.95
eps_clip = 0.2
eval_interval = 100             
solved_threshold = 250

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))
METAL = bool(int(os.getenv('METAL', '0')))

if METAL and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Metal) device")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

if PLOT: import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layers):
        super(ActorCritic, self).__init__()
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()

        if num_hidden_layers < 1: raise ValueError("num_hidden_layers must be at least 1")

        # Actor network
        self.actor_layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1): self.actor_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable std

        # Critic network
        self.critic_layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1): self.critic_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = state
        for layer in self.actor_layers: x = F.relu(layer(x))
        action_mean = self.actor_out(x)  # unbounded mean
        action_std = self.log_std.exp()  # learnable std
        
        v = state
        for layer in self.critic_layers: v = F.relu(layer(v))
        value = self.critic_out(v)
        
        return action_mean, action_std, value

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, batch_size):
        self.actor_critic = actor_critic
        # Use separate optimizers for actor and critic to decouple updates
        actor_params = [p for n, p in actor_critic.named_parameters() if 'actor_' in n or n == 'log_std']
        critic_params = [p for n, p in actor_critic.named_parameters() if 'critic_' in n]
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size

    def select_action(self, state, memory):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            action_mean, action_std, _ = self.actor_critic(state_tensor)
            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.sample()  # unbounded sample
            action = torch.tanh(raw_action)  # squash to [-1, 1]
            
            # change of variables correction
            logp_gaussian = dist.log_prob(raw_action).sum(-1)
            action_logprob = logp_gaussian - torch.log(1 - action**2 + 1e-6).sum(-1)

        memory.states.append(state_tensor)
        memory.actions.append(raw_action)  # store raw action for gradient computation
        memory.logprobs.append(action_logprob)

        return action.detach().cpu().numpy()

    def compute_advantages(self, rewards, state_values, is_terminals, normalize=True):
        T, N = rewards.shape
        device = rewards.device
        returns = torch.zeros_like(rewards)
        
        state_values_pad = torch.cat([state_values, state_values[-1:]], dim=0)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            returns[t] = gae + state_values_pad[t]

        advantages = returns - state_values_pad[:-1]
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.reshape(-1), returns.reshape(-1)
    
    def compute_losses(self, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        action_means, action_stds, state_values = self.actor_critic(batch_states)
        
        dist = torch.distributions.Normal(action_means, action_stds)
        
        # batch_actions are raw (unbounded) actions
        env_actions = torch.tanh(batch_actions)
        logp_gaussian = dist.log_prob(batch_actions).sum(-1)
        action_logprobs = logp_gaussian - torch.log(1 - env_actions**2 + 1e-6).sum(-1)
        
        state_values = torch.squeeze(state_values)
        
        # PPO actor loss with clipping
        ratios = torch.exp(action_logprobs - batch_logprobs.detach())
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(state_values, batch_returns)
        
        return actor_loss, critic_loss
    
    def update(self, memory):
        with torch.no_grad():
            # Use raw rewards - normalization adds noise
            rewards = torch.as_tensor(np.stack(memory.rewards), dtype=torch.float32).to(device)
            is_terms = torch.as_tensor(np.stack(memory.is_terminals), dtype=torch.float32).to(device)

            old_states = torch.cat(memory.states)
            old_actions = torch.cat(memory.actions)
            old_logprobs = torch.cat(memory.logprobs)

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
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in self.actor_critic.named_parameters() if 'actor_' in n or n == 'log_std'],
                    max_norm=0.5
                )
                self.actor_optimizer.step()

                # Update critic  
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in self.actor_critic.named_parameters() if 'critic_' in n],
                    max_norm=1.0
                )
                self.critic_optimizer.step()

def train_one_epoch(env, ppo, memory, states, update_timestep, num_envs):
    """
    Run one epoch of PPO:
    1. Collect experience until update_timestep steps
    2. Update policy and value function
    
    Returns episode returns and current state for continuing rollout.
    """
    timestep = 0
    episode_returns = []
    per_env_returns = np.zeros(num_envs)
    
    while timestep < update_timestep:
        timestep += num_envs
        
        actions = ppo.select_action(states, memory)
        next_states, rewards, terminated, truncated, _ = env.step(actions)
        
        memory.rewards.append(rewards)
        memory.is_terminals.append(np.logical_or(terminated, truncated).astype(float))
        
        per_env_returns += rewards
        
        done_mask = np.logical_or(terminated, truncated)
        if np.any(done_mask):
            for idx in np.where(done_mask)[0]:
                episode_returns.append(per_env_returns[idx])
                per_env_returns[idx] = 0.0
            
            if hasattr(env, 'reset_done'):
                states_reset, _ = env.reset_done()
                next_states[done_mask] = states_reset
            else:
                next_states, _ = env.reset()
                per_env_returns[:] = 0.0
        
        states = next_states
    
    # update policy
    ppo.update(memory)
    memory.clear_memory()
    
    return episode_returns, states

def train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, num_hidden_layers, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda, batch_size, num_envs=1):
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim, num_hidden_layers).to(device)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, batch_size)
    memory = Memory()
    
    all_episode_returns = []
    last_eval = float('-inf')
    recent_train_avg = float('-inf')
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    states, _ = env.reset()
    
    pbar = trange(max_episodes, unit='ep')
    pbar.write("  ep |     ret | train_avg |    eval")
    pbar.write("-" * 42)
    completed_episodes = 0
    
    while completed_episodes < max_episodes:
        # run one epoch: collect experience and update policy
        epoch_returns, states = train_one_epoch(env, ppo, memory, states, update_timestep, num_envs)
        all_episode_returns.extend(epoch_returns)
        
        for _ in epoch_returns:
            pbar.update(1)
            completed_episodes += 1
            
            if completed_episodes > 0 and completed_episodes % eval_interval == 0:
                eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, num_envs=num_envs)
                last_eval = eval_ret
                
                # Compute rolling average of last 100 training episodes
                recent_train_avg = np.mean(all_episode_returns[-100:]) if len(all_episode_returns) >= 100 else np.mean(all_episode_returns)
                
                # Log learned std for diagnostics
                current_std = actor_critic.log_std.exp().detach().cpu().numpy()
                std_info = f"std=[{current_std[0]:.3f}, {current_std[1]:.3f}]"
                
                desc = f"ep {completed_episodes:4d} | ret {all_episode_returns[-1]:7.2f} | train_avg {recent_train_avg:7.2f} | eval {last_eval:7.2f} | {std_info}"
                pbar.set_description(desc)
                pbar.write(f"ep {completed_episodes:4d} | ret {all_episode_returns[-1]:7.2f} | train_avg {recent_train_avg:7.2f} | eval {last_eval:7.2f} | {std_info}")
                
                # Render one episode after each eval (if RENDER=1)
                if RENDER:
                    render_policy(env_name, actor_critic, max_timesteps)
                
                # Check if SOLVED based on training average (more honest!)
                if recent_train_avg >= solved_threshold:
                    pbar.write(f"SOLVED with training avg: {recent_train_avg:.2f} (eval: {eval_ret:.2f})")
                    pbar.close()
                    env.close()
                    if PLOT:
                        plt.ioff()
                        plt.show()
                    return
            
            elif completed_episodes % log_interval == 0:
                # Compute rolling average for logging
                recent_train_avg = np.mean(all_episode_returns[-100:]) if len(all_episode_returns) >= 100 else np.mean(all_episode_returns)
                
                desc = f"ep {completed_episodes:4d} | ret {all_episode_returns[-1]:7.2f} | train_avg {recent_train_avg:7.2f} | eval {last_eval:7.2f}"
                pbar.set_description(desc)
                pbar.write(f"ep {completed_episodes:4d} | ret {all_episode_returns[-1]:7.2f} | train_avg {recent_train_avg:7.2f}")
            
            if PLOT and completed_episodes % (log_interval * 2) == 0:
                ax.clear()
                ax.plot(all_episode_returns, alpha=0.3)
                if len(all_episode_returns) >= 10:
                    ma = np.convolve(all_episode_returns, np.ones(10)/10, mode='valid')
                    ax.plot(range(9, len(all_episode_returns)), ma, label='10-ep MA')
                else:
                    ax.plot(all_episode_returns, label='Returns')
                ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
    
    env.close()
    if PLOT:
        plt.ioff()
        plt.show()
    
    pbar.close()

def evaluate_policy(env_name, actor_critic, max_timesteps, num_envs: int = 16):
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    states, _ = env.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    finished_rewards = []
    finished_mask = np.zeros(num_envs, dtype=bool)
    actor_critic.eval()

    with torch.no_grad():
        for _ in range(max_timesteps):
            state_tensor = torch.as_tensor(states, dtype=torch.float32).to(device)
            action_mean, _, _ = actor_critic(state_tensor)
            actions_np = torch.tanh(action_mean).cpu().numpy()  # squash for environment
            states, rewards, terminated, truncated, _ = env.step(actions_np)
            episode_rewards += rewards
            
            # Save reward when env finishes (first time only)
            done_mask = np.logical_or(terminated, truncated)
            for idx in np.where(done_mask)[0]:
                if not finished_mask[idx]:
                    finished_rewards.append(episode_rewards[idx])
                    finished_mask[idx] = True
            
            # Stop when all envs finished at least once
            if finished_mask.all():
                break

    env.close()
    actor_critic.train()
    return float(np.mean(finished_rewards)) if finished_rewards else 0.0

def render_policy(env_name, actor_critic, max_timesteps):
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    total = 0.0
    actor_critic.eval()
    with torch.no_grad():
        for _ in range(max_timesteps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action_mean, _, _ = actor_critic(s)
            action = torch.tanh(action_mean).squeeze(0).cpu().numpy()  # squash for environment
            state, reward, done, trunc, _ = env.step(action)
            total += reward
            if done or trunc:
                break
    env.close()
    actor_critic.train()
    return total

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, num_hidden_layers, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda, batch_size, num_envs=num_envs)