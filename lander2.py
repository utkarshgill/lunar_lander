import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal

# Hyperparameters
env_name = 'LunarLanderContinuous-v2'
state_dim = 8
action_dim = 2
max_episodes = 2000
max_timesteps = 3000
update_timestep = 5000
log_interval = 20
hidden_dim = 128
lr = 3e-4
gamma = 0.99
K_epochs = 100
eps_clip = 0.2
action_std = 0.5
gae_lambda = 0.95
ppo_loss_coef = 1.0
critic_loss_coef = 0.8
entropy_coef = 0.01

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
        # Actor network forward pass
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_mean = torch.tanh(self.actor_out(x))  # Continuous action space
        
        # Critic network forward pass
        v = F.relu(self.critic_fc1(state))
        v = F.relu(self.critic_fc2(v))
        value = self.critic_out(v)
        
        return action_mean, value
    
class PPO:
    def __init__(self, actor_critic, lr, gamma, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
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

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.cat(memory.states).detach()
        old_actions = torch.cat(memory.actions).detach()
        old_logprobs = torch.cat(memory.logprobs).detach()

        for _ in range(self.K_epochs):
            action_means, state_values = self.actor_critic(old_states)
            action_var = torch.full((action_means.size(-1),), self.action_std**2)
            cov_mat = torch.diag(action_var).unsqueeze(0)

            dist = MultivariateNormal(action_means, covariance_matrix=cov_mat)
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)

            advantages = rewards - state_values.detach()

            ratios = torch.exp(action_logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = self.ppo_loss_coef * -torch.min(surr1, surr2) + self.critic_loss_coef * self.mse_loss(state_values, rewards) - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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

class GAE:
    def __init__(self, gamma, lamda):
        self.gamma = gamma
        self.lamda = lamda

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def compute_returns(self, rewards, dones, values, next_values):
        advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns
    
def train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef):
    timestep = 0
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    ppo = PPO(actor_critic, lr, gamma, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef)
    gae = GAE(gamma, gae_lambda)
    memory = Memory()
    
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
                _, next_value = actor_critic(torch.FloatTensor(next_state).unsqueeze(0))
                values = [actor_critic(torch.FloatTensor(s).unsqueeze(0))[1].item() for s in memory.states]
                next_values = values[1:] + [next_value.item()]
                returns = gae.compute_returns(memory.rewards, memory.is_terminals, values, next_values)
                memory.returns = torch.tensor(returns)
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            state = next_state
            if done or trunc:
                break

        if episode % log_interval == 0:
            print(f'ep {episode:6} return {total_reward:.2f}')

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef)
