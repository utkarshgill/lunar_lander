import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# Hyperparameters
env_name = 'LunarLanderContinuous-v2'
state_dim = 8
action_dim = 2
max_episodes = 5000
max_timesteps = 1000
update_timestep = 4000
log_interval = 20
hidden_dim = 128
lr = 3e-4
gamma = 0.99
K_epochs = 3
eps_clip = 0.2
action_std = 1.0
gae_lambda = 0.99
ppo_loss_coef = 1.0
critic_loss_coef = 0.5
entropy_coef = 0.1
batch_size = 32

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dropout_rate=0.2):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        self.actor_dropout = nn.Dropout(dropout_rate)
        
        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        self.critic_dropout = nn.Dropout(dropout_rate)

    def forward(self, state):
        # Actor network forward pass
        x = F.relu(self.actor_ln1(self.actor_fc1(state)))
        x = self.actor_dropout(x)
        x = F.relu(self.actor_ln2(self.actor_fc2(x)))
        x = self.actor_dropout(x)
        action_mean = torch.tanh(self.actor_out(x))  # Continuous action space
        
        # Critic network forward pass
        v = F.relu(self.critic_ln1(self.critic_fc1(state)))
        v = self.critic_dropout(v)
        v = F.relu(self.critic_ln2(self.critic_fc2(v)))
        v = self.critic_dropout(v)
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
    def __init__(self, actor_critic, lr, gamma, lamda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.mse_loss = nn.MSELoss()
        self.batch_size = batch_size

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        
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
        advantages = []
        gae = 0
        state_values = torch.cat((state_values, torch.tensor([0.0])))  # Add a zero to handle the last state value

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * state_values[step + 1] * (1 - is_terminals[step]) - state_values[step]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + state_values[:-1]

        return advantages, returns
    
    def update(self, memory):
        with torch.no_grad():
            rewards = torch.tensor(memory.rewards, dtype=torch.float32)
            is_terms = torch.tensor(memory.is_terminals, dtype=torch.float32)
            
            old_states = torch.cat(memory.states).detach()
            old_actions = torch.cat(memory.actions).detach()
            old_logprobs = torch.cat(memory.logprobs).detach()
            _, state_values = self.actor_critic(old_states)
            state_values = torch.squeeze(state_values)

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            advantages, returns = self.compute_advantages(rewards, state_values, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.K_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns = batch
                
                # Forward pass
                action_means, state_values = self.actor_critic(batch_states)
                action_var = torch.full((action_means.size(-1),), self.action_std**2)
                cov_mat = torch.diag(action_var).unsqueeze(0)
                
                dist = MultivariateNormal(action_means, covariance_matrix=cov_mat)
                action_logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy()
                state_values = torch.squeeze(state_values)
                
                # Compute ratios
                ratios = torch.exp(action_logprobs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # PPO loss components
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(state_values, batch_returns)
                entropy_loss = dist_entropy.mean()

                # Total loss
                loss = self.ppo_loss_coef * actor_loss + self.critic_loss_coef * critic_loss - self.entropy_coef * entropy_loss

                # Backward pass and optimizer step with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

def train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size):
    timestep = 0
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    ppo = PPO(actor_critic, lr, gamma, gae_lambda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size)
    memory = Memory()
    
    episode_returns = []
    running_avg_returns = []
    
    plt.ion()
    fig, ax = plt.subplots()
    
    for episode in range(1, max_episodes + 1):
        render_mode = 'human' if episode % 200 == 0 else None
        env = gym.make(env_name, render_mode=render_mode)
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

            # Dynamic plotting
            ax.clear()
            ax.plot(episode_returns, label='Returns')
            ax.plot(running_avg_returns, label='Running Average Returns')
            ax.axhline(y=max(episode_returns), color='r', linestyle='--', label='Max Return')
            ax.legend()
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size)
