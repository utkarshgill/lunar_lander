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
num_envs = int(os.getenv('NUM_ENVS', 8))

max_episodes = 500
max_timesteps = 1000

update_timestep = 4096          # more data per update
log_interval = 10               
batch_size = 512               
K_epochs = 6                   # squeeze more learning from each rollout                   

hidden_dim = 512

lr_actor = 1e-4                # much slower - prevent policy collapse
lr_critic = 5e-4               # moderate critic learning

gamma = 0.998                  # longer horizon for landing task
gae_lambda = 0.98              # less bias in advantage estimates
eps_clip = 0.15                # moderate clipping

action_std = 0.6               # aggressive exploration to find landing zone

eval_interval = 40             
solved_threshold = 240

PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))

if PLOT: import matplotlib.pyplot as plt

# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_mean = torch.tanh(self.actor_out(x))
        
        v = F.relu(self.critic_fc1(state))
        v = F.relu(self.critic_fc2(v))
        value = self.critic_out(v)
        
        return action_mean, value

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
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, action_std, batch_size):
        self.actor_critic = actor_critic
        # Use separate optimizers for actor and critic to decouple updates
        actor_params = [p for n, p in actor_critic.named_parameters() if 'actor_' in n]
        critic_params = [p for n, p in actor_critic.named_parameters() if 'critic_' in n]
        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.batch_size = batch_size

    def select_action(self, state, memory):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_mean, _ = self.actor_critic(state_tensor)
            std_tensor = torch.ones_like(action_mean) * self.action_std
            dist = torch.distributions.Normal(action_mean, std_tensor)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(-1)

        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach().cpu().numpy()

    def compute_advantages(self, rewards, state_values, is_terminals):
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
        return advantages.reshape(-1), returns.reshape(-1)
    
    def update(self, memory):
        with torch.no_grad():
            # Use raw rewards - normalization adds noise
            rewards = torch.as_tensor(np.stack(memory.rewards), dtype=torch.float32)
            is_terms = torch.as_tensor(np.stack(memory.is_terminals), dtype=torch.float32)

            old_states = torch.cat(memory.states)
            old_actions = torch.cat(memory.actions)
            old_logprobs = torch.cat(memory.logprobs)

            _, old_state_values = self.actor_critic(old_states)
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
                
                action_means, state_values = self.actor_critic(batch_states)
                
                std_tensor = torch.ones_like(action_means) * self.action_std
                dist = torch.distributions.Normal(action_means, std_tensor)
                
                action_logprobs = dist.log_prob(batch_actions).sum(-1)
                state_values = torch.squeeze(state_values)
                
                # PPO losses
                ratios = torch.exp(action_logprobs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(state_values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in self.actor_critic.named_parameters() if 'actor_' in n],
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

def train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, batch_size, num_envs=1):
    timestep = 0
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, action_std, batch_size)
    memory = Memory()
    
    episode_returns = []
    last_eval = float('-inf')
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    states, _ = env.reset()

    pbar = trange(max_episodes, unit='ep')
    per_env_returns = np.zeros(num_envs)
    completed_episodes = 0

    while completed_episodes < max_episodes:
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
                pbar.update(1)
                completed_episodes += 1
                per_env_returns[idx] = 0.0

                if completed_episodes % log_interval == 0:
                    pbar.set_description(f"ep {completed_episodes} ret {episode_returns[-1]:.2f} eval {last_eval:.1f}")

                    if PLOT and completed_episodes % (log_interval * 2) == 0:
                        ax.clear()
                        # Raw returns (faded)
                        ax.plot(episode_returns, alpha=0.3)
                        # Overlay 10-episode moving average for stability
                        if len(episode_returns) >= 10:
                            ma = np.convolve(episode_returns, np.ones(10)/10, mode='valid')
                            ax.plot(range(9, len(episode_returns)), ma, label='10-ep MA')
                        else:
                            ax.plot(episode_returns, label='Returns')
                        ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)

                if completed_episodes > 0 and completed_episodes % eval_interval == 0:
                    eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, num_envs=num_envs)
                    last_eval = eval_ret
                    pbar.set_description(f"ep {completed_episodes} ret {episode_returns[-1]:.2f} eval {last_eval:.1f}")

                    if eval_ret >= solved_threshold:
                        pbar.write(f"SOLVED with eval score: {eval_ret:.2f}")
                        render_policy(env_name, actor_critic, max_timesteps)
                        pbar.close()
                        return

            if hasattr(env, 'reset_done'):
                states_reset, _ = env.reset_done()
                next_states[done_mask] = states_reset
            else:
                next_states, _ = env.reset()
                per_env_returns[:] = 0.0

        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        states = next_states

    if PLOT:
        plt.ioff()
        plt.show()

    pbar.close()

def evaluate_policy(env_name, actor_critic, max_timesteps, num_envs: int = 16):
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    states, _ = env.reset()
    total_rewards = np.zeros(num_envs, dtype=np.float32)
    actor_critic.eval()

    with torch.no_grad():
        for _ in range(max_timesteps):
            state_tensor = torch.as_tensor(states, dtype=torch.float32)
            action_mean, _ = actor_critic(state_tensor)
            actions_np = action_mean.cpu().numpy()
            states, rewards, terminated, truncated, _ = env.step(actions_np)
            total_rewards += rewards
            if np.all(np.logical_or(terminated, truncated)):
                break

    env.close()
    actor_critic.train()
    return float(total_rewards.mean())

def render_policy(env_name, actor_critic, max_timesteps):
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    total = 0.0
    actor_critic.eval()
    with torch.no_grad():
        for _ in range(max_timesteps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mean, _ = actor_critic(s)
            action = action_mean.squeeze(0).cpu().numpy()
            state, reward, done, trunc, _ = env.step(action)
            total += reward
            if done or trunc:
                break
    env.close()
    actor_critic.train()
    return total

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, batch_size, num_envs=num_envs)
