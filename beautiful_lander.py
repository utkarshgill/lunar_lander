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

update_timestep = 2048          # steps before PPO update (across all envs)
log_interval = 10              # episodes between progress updates  
batch_size = 512               # minibatch size for PPO updates
K_epochs = 5                   # PPO epochs per update (increased for stability)

hidden_dim = 512

lr_actor = 2e-4                # slower actor learning for stability
lr_critic = 8e-4               # slower critic learning

gamma = 0.995                   # discount factor
gae_lambda = 0.99              # GAE parameter
eps_clip = 0.2                 # PPO clip parameter (tighter clipping)

action_std_init = 0.5          # initial exploration noise (reduced)
action_std_min = 0.02          # minimum exploration (small residual noise)
action_std_decay_steps = 2_000_000  # slower exploration decay for robustness
entropy_coef_init = 0.02       # initial entropy bonus (reduced)
entropy_coef_final = 0.001     # final entropy floor
entropy_decay_steps = 400_000  # entropy decay schedule

ppo_loss_coef = 1.0
critic_loss_coef = 0.5

eval_interval = 40             # evaluate every N episodes (more frequent)
solved_threshold = 240         # stricter solved threshold for more precise landings

# Consistent penalty when an episode times out without a proper landing (matches evaluation logic)
timeout_penalty = -100.0

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

        # ------------------------------------------------------------------
        # Weight initialisation – orthogonal is recommended for PPO because
        # it yields well-scaled outputs at the first forward pass and improves
        # learning stability across random seeds.
        # ------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Output layers use smaller gains (per PPO/TD3 best-practices)
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)

    def forward(self, state):
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_mean = torch.tanh(self.actor_out(x))  # Continuous action space
        
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
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, action_std_init, ppo_loss_coef, critic_loss_coef, entropy_coef_init, batch_size, entropy_coef_final, entropy_decay_steps, action_std_min, action_std_decay_steps):
        self.actor_critic = actor_critic

        actor_params = [p for n, p in actor_critic.named_parameters() if 'actor_' in n]
        critic_params = [p for n, p in actor_critic.named_parameters() if 'critic_' in n]

        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
        self.gamma = gamma
        self.lamda = lamda
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std_init = action_std_init
        self.min_action_std = action_std_min
        self.action_std_decay_steps = action_std_decay_steps
        self.action_std = action_std_init
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef

        self.entropy_start = entropy_coef_init
        self.entropy_final = entropy_coef_final
        self.entropy_decay_steps = entropy_decay_steps
        self.current_entropy_coef = entropy_coef_init
        self.batch_size = batch_size
        self.total_steps = 0
        self.actor_lr_init = lr_actor

    def select_action(self, state, memory):
        """Select an action for a *batch* of states.

        The input ``state`` is assumed to follow the Gymnasium vector API:
        Always a 2-D array of shape ``(N, state_dim)``.
        The function returns a numpy array of the same batch size with shape
        ``(batch, action_dim)``.
        """

        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        batch_size = state_tensor.size(0)

        # Increment global step counter and linearly decay exploration noise
        self.total_steps += batch_size
        if self.total_steps < self.action_std_decay_steps:
            frac = self.total_steps / self.action_std_decay_steps
            self.action_std = self.action_std_init - (self.action_std_init - self.min_action_std) * frac
        else:
            self.action_std = self.min_action_std

        # Update entropy coefficient linearly
        frac_ent = min(1.0, self.total_steps / self.entropy_decay_steps)
        self.current_entropy_coef = self.entropy_start - (self.entropy_start - self.entropy_final) * frac_ent

        with torch.no_grad():
            action_mean, _ = self.actor_critic(state_tensor)
            if self.action_std > 0:
                std_tensor = torch.ones_like(action_mean) * self.action_std
                dist = torch.distributions.Normal(action_mean, std_tensor)
                action = dist.sample()
                action_logprob = dist.log_prob(action).sum(-1)
            else:
                action = action_mean
                action_logprob = torch.zeros(state_tensor.size(0))

        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        actions_np = action.detach().cpu().numpy()
        return actions_np

    def compute_advantages(self, rewards, state_values, is_terminals):
        """Compute Generalised Advantage Estimation (supports batched envs).

        Args:
            rewards: Tensor of shape ``(T,)`` for single env or ``(T, N)`` for N envs.
            state_values: Same shape as ``rewards``.
            is_terminals: Bool tensor indicating episode termination (same shape).
        Returns:
            advantages_flat: 1-D tensor ``(T*N,)``
            returns_flat:     1-D tensor ``(T*N,)``
        """

        if rewards.dim() == 1:  # ----- original single-env path -----
            returns = []
            gae = 0.0
            state_values_pad = torch.cat((state_values, torch.tensor([0.0])))
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
                gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
                returns.insert(0, gae + state_values_pad[t])

            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = returns - state_values_pad[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return advantages, returns

        # ---------- batched vector-env path ----------
        T, N = rewards.shape
        device = rewards.device
        returns = torch.zeros((T, N), dtype=torch.float32, device=device)
        advantages = torch.zeros_like(returns)

        state_values_pad = torch.cat([state_values, torch.zeros((1, N), device=device)], dim=0)  # (T+1, N)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * state_values_pad[t + 1] * (1 - is_terminals[t]) - state_values_pad[t]
            gae = delta + self.gamma * self.lamda * (1 - is_terminals[t]) * gae
            returns[t] = gae + state_values_pad[t]

        advantages = returns - state_values_pad[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten time and env dimensions for the optimiser
        return advantages.reshape(-1), returns.reshape(-1)
    
    def update(self, memory):
        with torch.no_grad():
            import numpy as _np  # local alias to avoid top-level clutter
            rewards = torch.as_tensor(_np.stack(memory.rewards), dtype=torch.float32)
            is_terms = torch.as_tensor(_np.stack(memory.is_terminals), dtype=torch.float32)

            # Stack & flatten
            old_states = torch.cat(memory.states)       # (T*N, state_dim)
            old_actions = torch.cat(memory.actions)     # (T*N, action_dim)
            old_logprobs = torch.cat(memory.logprobs)   # (T*N,)

            _, old_state_values = self.actor_critic(old_states)
            old_state_values = old_state_values.squeeze(-1)  # (T*N,)

            # Reshape to (T, N) for advantage computation if needed
            if rewards.dim() > 1:
                N = rewards.size(1)
                old_state_values_reshaped = old_state_values.view(-1, N)
            else:
                old_state_values_reshaped = old_state_values

            advantages, returns = self.compute_advantages(rewards, old_state_values_reshaped, is_terms)
        
        dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns, old_state_values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.K_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns, batch_old_values = batch
                
                # Forward pass - optimized distribution computation
                action_means, state_values = self.actor_critic(batch_states)
                
                # Use simpler Normal distribution instead of MultivariateNormal for speed
                std_tensor = torch.ones_like(action_means) * self.action_std
                std_tensor = torch.clamp(std_tensor, min=self.min_action_std)
                dist = torch.distributions.Normal(action_means, std_tensor)
                
                action_logprobs = dist.log_prob(batch_actions).sum(-1)  # sum over action dims
                dist_entropy = dist.entropy().sum(-1)  # sum over action dims
                state_values = torch.squeeze(state_values)
                
                # Value function clipping (per PPO paper) with Huber (smooth L1) loss
                value_clipped = batch_old_values + torch.clamp(state_values - batch_old_values, -self.eps_clip, self.eps_clip)
                critic_loss_unclipped = F.smooth_l1_loss(state_values, batch_returns, reduction='none')
                critic_loss_clipped = F.smooth_l1_loss(value_clipped, batch_returns, reduction='none')
                critic_loss = 0.5 * torch.mean(torch.max(critic_loss_unclipped, critic_loss_clipped))

                # Compute ratios
                ratios = torch.exp(action_logprobs - batch_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # PPO loss components
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = dist_entropy.mean()

                # ---------------------------------------------
                # Optimise actor
                # ---------------------------------------------
                self.actor_optimizer.zero_grad(set_to_none=True)
                (self.ppo_loss_coef * actor_loss - self.current_entropy_coef * entropy_loss).backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_( [p for group in self.actor_optimizer.param_groups for p in group['params']], max_norm=0.2)
                self.actor_optimizer.step()

                # ---------------------------------------------
                # Optimise critic
                # ---------------------------------------------
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.critic_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_( [p for group in self.critic_optimizer.param_groups for p in group['params']], max_norm=0.2)
                self.critic_optimizer.step()

                # Approximate KL divergence for early stopping & adaptive LR
                approx_kl = (batch_logprobs - action_logprobs).mean().abs()

                # Adapt actor learning-rate based on KL (simple heuristic)
                if approx_kl > 0.02:  # tighter KL threshold for stability
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.9, 1e-5)
                elif approx_kl < 0.015:
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = min(g["lr"] * 1.1, self.actor_lr_init)

                if self.total_steps > 500_000: self.eps_clip = 0.15   # Anneal clip range after enough training steps
                if approx_kl > 0.03: break                            # If policy changed too much, break out early

    def freeze_actor(self, lr: float = 1e-5):
        """Freeze/finetune actor: drop LR and set deterministic actions for evaluation."""
        for group in self.actor_optimizer.param_groups: group["lr"] = lr
        # Make subsequent actions deterministic
        self.action_std = 0.0
        self.min_action_std = 0.0

def train(env_name, max_episodes, max_timesteps, _update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef_init, batch_size, num_envs=1):
    """Train with single or parallel environments (vectorised when num_envs>1)."""
    timestep = 0

    # Use constant rollout size independent of num_envs
    update_timestep = _update_timestep  # e.g., 2048 steps (aggregated across envs)
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, action_std_init, ppo_loss_coef, critic_loss_coef, entropy_coef_init, batch_size, entropy_coef_final, entropy_decay_steps, action_std_min, action_std_decay_steps)
    memory = Memory()
    
    episode_returns = []
    last_eval = float('-inf')
    eval_scores = []  # deterministic evaluation scores
    
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

        actions = ppo.select_action(states, memory)  # (num_envs, action_dim)
        next_states, rewards, terminated, truncated, _ = env.step(actions)

        # ------------------------------------------------------------------
        # Align training with evaluation: penalise episodes that reach the
        # time-limit without a successful landing. This discourages hovering
        # behaviour that sometimes fools the shaped reward but fails eval.
        # ------------------------------------------------------------------
        timed_out = np.logical_and(truncated, np.logical_not(terminated))
        if np.any(timed_out):
            rewards = rewards.copy()              # avoid mutating original
            rewards[timed_out] += timeout_penalty  # add negative penalty

        # Record transitions
        memory.rewards.append(rewards)
        memory.is_terminals.append(np.logical_or(terminated, truncated).astype(float))

        # Accumulate returns only for ongoing episodes (with penalty applied)
        per_env_returns += rewards

        # When any env finishes an episode
        done_mask = np.logical_or(terminated, truncated)
        if np.any(done_mask):
            # Log and bookkeeping per finished env
            for idx in np.where(done_mask)[0]:
                episode_returns.append(per_env_returns[idx])
                pbar.update(1)
                completed_episodes += 1

                # Reset this environment's accumulator IMMEDIATELY after logging
                per_env_returns[idx] = 0.0

                if completed_episodes % log_interval == 0:
                    pbar.set_description(f"ep {completed_episodes} ret {episode_returns[-1]:.2f} eval {last_eval:.1f}")

                    if PLOT and completed_episodes % (log_interval * 2) == 0:  # less frequent plotting
                        ax.clear()
                        ax.plot(episode_returns, label='Returns')
                        ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)

                if completed_episodes > 0 and completed_episodes % eval_interval == 0:
                    eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps, 0.0, num_envs=num_envs)
                    eval_scores.append(eval_ret); last_eval = eval_ret
                    pbar.set_description(f"ep {completed_episodes} ret {episode_returns[-1]:.2f} eval {last_eval:.1f}")

                    # Require single evaluation above threshold for declaring solved
                    if eval_ret >= solved_threshold:
                        # extra confirmation on a single environment (hard case)
                        single_score = evaluate_policy(env_name, actor_critic, max_timesteps, 0.0, num_envs=1)
                        if single_score >= solved_threshold:
                            ppo.freeze_actor()
                            ppo.actor_frozen = True
                            pbar.write(f"Task SOLVED – 16-env mean {eval_ret:.1f}, single-env {single_score:.1f} ≥ {solved_threshold}")
                            final_eval = evaluate_policy(env_name, actor_critic, max_timesteps, 0.0, num_envs=num_envs)
                            pbar.write(f"Final evaluation score: {final_eval:.2f}")
                            pbar.write("Rendering one episode for human verification...")
                            render_policy(env_name, actor_critic, max_timesteps)
                            pbar.close()
                            return

            # Reset finished environments
            if hasattr(env, 'reset_done'):
                # Gymnasium ≥0.28 provides this convenience method
                states_reset, _ = env.reset_done()
                next_states[done_mask] = states_reset
            else:
                # Fallback – reset *all* envs (simpler, but slightly wasteful)
                next_states, _ = env.reset()
                # All environments were reset, so zero the return accumulator
                per_env_returns[:] = 0.0

        # Time to update PPO?
        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        states = next_states

    if PLOT:
        plt.ioff()
        plt.show()

    pbar.close()

def evaluate_policy(env_name, actor_critic, max_timesteps, action_std, num_envs: int = 16):
    """Evaluate policy on `num_envs` parallel environments (no rendering). Returns mean episodic return."""
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    states, _ = env.reset()
    total_rewards = np.zeros(num_envs, dtype=np.float32)
    landed = np.zeros(num_envs, dtype=bool)
    actor_critic.eval()

    with torch.no_grad():
        for _ in range(max_timesteps):
            state_tensor = torch.as_tensor(states, dtype=torch.float32)

            action_mean, _ = actor_critic(state_tensor)
            if action_std > 0:
                std_tensor = torch.ones_like(action_mean) * action_std
                actions = torch.distributions.Normal(action_mean, std_tensor).sample()
            else: actions = action_mean

            actions_np = actions.cpu().numpy()
            states, rewards, terminated, truncated, _ = env.step(actions_np)
            total_rewards += rewards
            landed |= np.logical_or(terminated, truncated)
            if np.all(landed):
                break

    # Penalise any env that never landed (was still flying when time expired)
    if not np.all(landed): total_rewards[~landed] -= 100.0  # strong penalty to disqualify hover-only policies
    env.close()
    return float(total_rewards.mean())

def render_policy(env_name, actor_critic, max_timesteps):
    """Run one deterministic episode with human rendering."""
    import gymnasium as gym, torch
    env = gym.make(env_name, render_mode='human')
    state,_ = env.reset(); total=0.0
    actor_critic.eval()
    with torch.no_grad():
        for _ in range(max_timesteps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mean,_ = actor_critic(s)
            action = action_mean.squeeze(0).cpu().numpy()
            state,reward,done,trunc,_ = env.step(action); total+=reward
            if done or trunc:
                break
    env.close()
    return total

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef_init, batch_size, num_envs=num_envs)
