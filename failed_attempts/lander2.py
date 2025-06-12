import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import os
from torch.utils.data import DataLoader, TensorDataset
import random


# Hyperparameters
env_name = 'LunarLanderContinuous-v2'
state_dim = 8
action_dim = 2
max_episodes = 5000
max_timesteps = 1000
update_timestep = 4096
log_interval = 20
hidden_dim = 256
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
K_epochs = 6
eps_clip = 0.2
action_std = 1.0  # initial std, will decay linearly to min_action_std
gae_lambda = 0.95
ppo_loss_coef = 1.0
critic_loss_coef = 1.0
entropy_coef = 0.01
batch_size = 1024

# -----------------------------------------------------------------------------
# Configuration flags controlled via environment variables
#   PLOT=0   -> disable matplotlib plotting (useful in headless environments)
#   RENDER=0 -> disable Gym rendering
# -----------------------------------------------------------------------------
PLOT = os.getenv('PLOT', 0)
RENDER = os.getenv('RENDER', 0)

# Conditional import to avoid issues on machines without display back-end
if PLOT:
    import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialisation (recommended for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Last layer smaller init
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)

    def forward(self, state):
        # Actor network forward pass
        x = F.relu(self.actor_ln1(self.actor_fc1(state)))
        x = F.relu(self.actor_ln2(self.actor_fc2(x)))
        action_mean = torch.tanh(self.actor_out(x))  # Continuous action space
        
        # Critic network forward pass
        v = F.relu(self.critic_ln1(self.critic_fc1(state)))
        v = F.relu(self.critic_ln2(self.critic_fc2(v)))
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
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size):
        self.actor_critic = actor_critic

        # Separate actor/critic parameter groups
        actor_params = [p for n, p in actor_critic.named_parameters() if 'actor_' in n]
        critic_params = [p for n, p in actor_critic.named_parameters() if 'critic_' in n]

        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std_init = action_std  # initial std
        self.min_action_std = 0.1          # final std after decay
        self.action_std_decay_steps = 500_000  # total env steps over which to decay
        self.action_std = action_std       # current std (mutable)
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.total_steps = 0

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Increment global step counter and linearly decay exploration noise
        self.total_steps += 1
        if self.total_steps < self.action_std_decay_steps:
            frac = self.total_steps / self.action_std_decay_steps
            self.action_std = self.action_std_init - (self.action_std_init - self.min_action_std) * frac
        else:
            self.action_std = self.min_action_std

        with torch.no_grad():
            action_mean, _ = self.actor_critic(state)
            action_var = torch.full((action_mean.size(-1),), self.action_std**2)
            cov_mat = torch.diag(action_var).unsqueeze(0)
            
            dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def rtg(self, rewards, is_terms):
        out = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terms)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            out.insert(0, discounted_reward)
        return out

    def compute_advantages(self, rewards, state_values, is_terminals):
        """Compute GAE advantages and bootstrap returns.

        1. Build returns = GAE + V(s).
        2. Advantages = returns - V(s).
        3. *Then* normalise advantages only (returns stay on the original reward scale).
        """
        returns = []
        gae = 0
        state_values = torch.cat((state_values, torch.tensor([0.0])))  # pad for t+1 when step == last

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * state_values[step + 1] * (1 - is_terminals[step]) - state_values[step]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[step]) * gae
            returns.insert(0, gae + state_values[step])

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - state_values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns
    
    def update(self, memory):
        with torch.no_grad():
            rewards = torch.tensor(memory.rewards, dtype=torch.float32)
            is_terms = torch.tensor(memory.is_terminals, dtype=torch.float32)
            
            old_states = torch.cat(memory.states).detach()
            old_actions = torch.cat(memory.actions).detach()
            old_logprobs = torch.cat(memory.logprobs).detach()
            _, old_state_values = self.actor_critic(old_states)
            old_state_values = torch.squeeze(old_state_values)

            advantages, returns = self.compute_advantages(rewards, old_state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns, old_state_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.K_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns, batch_old_values = batch
                
                # Forward pass
                action_means, state_values = self.actor_critic(batch_states)
                action_var = torch.full((action_means.size(-1),), self.action_std**2)
                cov_mat = torch.diag(action_var).unsqueeze(0)
                
                dist = MultivariateNormal(action_means, covariance_matrix=cov_mat)
                action_logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy()
                state_values = torch.squeeze(state_values)
                
                # Value function clipping (per PPO paper)
                value_clipped = batch_old_values + torch.clamp(state_values - batch_old_values, -self.eps_clip, self.eps_clip)
                critic_loss_unclipped = (state_values - batch_returns) ** 2
                critic_loss_clipped = (value_clipped - batch_returns) ** 2
                critic_loss = 0.5 * torch.mean(torch.max(critic_loss_unclipped, critic_loss_clipped))

                # Compute ratios
                ratios = torch.exp(action_logprobs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # PPO loss components
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = dist_entropy.mean()

                # (Optional) KL early stopping disabled for now to allow larger updates
                # approx_kl = torch.mean(batch_logprobs - action_logprobs).abs()
                # if approx_kl > 0.03:
                #     early_stop = True
                # else:
                #     early_stop = False
                early_stop = False

                # ---------------------------------------------
                # Optimise actor
                # ---------------------------------------------
                self.actor_optimizer.zero_grad(set_to_none=True)
                (self.ppo_loss_coef * actor_loss - self.entropy_coef * entropy_loss).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_( [p for group in self.actor_optimizer.param_groups for p in group['params']], max_norm=0.5)
                self.actor_optimizer.step()

                # ---------------------------------------------
                # Optimise critic
                # ---------------------------------------------
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.critic_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_( [p for group in self.critic_optimizer.param_groups for p in group['params']], max_norm=0.5)
                self.critic_optimizer.step()

                if early_stop:
                    break
            if early_stop:
                break

def train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size):
    timestep = 0
    # Seeding for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size)
    memory = Memory()
    
    episode_returns = []
    running_avg_returns = []
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    # Environment (single instance) with observation normalisation
    render_mode = 'human' if RENDER else None
    env = NormalizeObservation(gym.make(env_name, render_mode=render_mode))

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1

            action = ppo.select_action(state, memory)
            next_state, reward, done, trunc, _ = env.step(action)

            memory.rewards.append(reward)
            total_reward += reward
            memory.is_terminals.append(done or trunc)

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            state = next_state
            if done or trunc:
                break

        episode_returns.append(total_reward)
        running_avg = np.mean(episode_returns[-log_interval:]) if episode >= log_interval else np.mean(episode_returns)
        running_avg_returns.append(running_avg)
        
        if episode % log_interval == 0:
            print(f'ep {episode:6} return {total_reward:.2f}')

            if PLOT:
                # Dynamic plotting
                ax.clear()
                ax.plot(episode_returns, label='Returns')
                ax.plot(running_avg_returns, label='Running Average Returns')
                ax.axhline(y=max(episode_returns), color='r', linestyle='--', label='Max Return')
                ax.legend()
                ax.set_xlabel('Episode')
                ax.set_ylabel('Return')
                plt.pause(0.01)
    
    if PLOT:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size)
