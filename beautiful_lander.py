import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import os
from torch.utils.data import DataLoader, TensorDataset
import random
import sys
import multiprocessing as _mp


# Hyperparameters
env_name = 'LunarLanderContinuous-v2'
state_dim = 8
action_dim = 2
# number of parallel environments (set to 1 for previous behaviour)
num_envs = int(os.getenv('NUM_ENVS', 8))
max_episodes = 5000
max_timesteps = 1000
update_timestep = 8192  # default (actual value recomputed in train)
log_interval = 20
hidden_dim = 256
lr_actor = 1e-4  # tuned actor learning rate
lr_critic = 4e-4  # slightly lower critic learning rate for stability
gamma = 0.99
K_epochs = 2  # fewer optimisation epochs per update (larger batch)
eps_clip = 0.2  # wider clip range for two-epoch regime
action_std = 0.8  # initial std for Gaussian policy
gae_lambda = 0.97
ppo_loss_coef = 1.0
critic_loss_coef = 1.0
entropy_coef = 0.03  # starting entropy coefficient (will decay)
batch_size = 1024

# -----------------------------------------------------------------------------
# Configuration flags controlled via environment variables
#   PLOT=0   -> disable matplotlib plotting (useful in headless environments)
#   RENDER=0 -> disable Gym rendering
# -----------------------------------------------------------------------------
PLOT = bool(int(os.getenv('PLOT', '0')))
RENDER = bool(int(os.getenv('RENDER', '0')))

# Conditional import to avoid issues on machines without display back-end
if PLOT:
    import matplotlib.pyplot as plt

# How often (episodes) to run a human-rendered evaluation when RENDER=1
eval_interval = int(os.getenv('EVAL_INTERVAL', 200))

# Entropy decay schedule
entropy_coef_final = 0.005  # keep small entropy floor to prevent premature convergence
entropy_decay_steps = 500_000  # faster entropy annealing

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
        self.critic_ln1 = nn.Identity()
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_ln2 = nn.Identity()
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

class RunningNorm:
    """Track running mean & variance to normalise rewards."""

    def __init__(self, eps: float = 1e-6):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        import numpy as _np
        x_arr = _np.asarray(x)
        batch_mean = float(x_arr.mean())
        batch_var = float(x_arr.var())
        batch_count = x_arr.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta * delta * self.count * batch_count / tot_count) / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        import numpy as _np
        return (x - self.mean) / (_np.sqrt(self.var) + 1e-8)

class PPO:
    def __init__(self, actor_critic, lr_actor, lr_critic, gamma, lamda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, entropy_coef_final=0.005, entropy_decay_steps=1_000_000):
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
        self.min_action_std = 0.1          # exploration floor to avoid collapse
        self.action_std_decay_steps = 1_500_000  # slower annealing to retain exploration
        self.action_std = action_std       # current std (mutable)
        self.ppo_loss_coef = ppo_loss_coef
        self.critic_loss_coef = critic_loss_coef
        # Entropy coefficient schedule
        self.entropy_start = entropy_coef
        self.entropy_final = entropy_coef_final
        self.entropy_decay_steps = entropy_decay_steps
        self.current_entropy_coef = entropy_coef  # will be updated over time
        self.batch_size = batch_size
        self.total_steps = 0
        # Store initial actor LR for adaptive schedule
        self.actor_lr_init = lr_actor

    def select_action(self, state, memory):
        """Select an action for a *batch* of states.

        The input ``state`` is assumed to follow the Gymnasium vector API:
        - For a single environment it is a 1-D array of shape ``(state_dim,)``.
        - For *N* parallel envs it is a 2-D array of shape ``(N, state_dim)``.
        The function returns a numpy array of the same batch size with shape
        ``(batch, action_dim)``.
        """

        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.dim() == 1:  # single env -> add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
            single_env = True
        else:
            single_env = False
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

            # Diagonal Gaussian with identical std across all dimensions/envs
            std_tensor = torch.ones_like(action_mean) * self.action_std
            # Ensure a minimum exploration noise
            std_tensor = torch.clamp(std_tensor, min=self.min_action_std)
            dist = torch.distributions.Normal(action_mean, std_tensor)

            action = dist.sample()                          # (batch, action_dim)
            action_logprob = dist.log_prob(action).sum(-1)  # (batch,)

        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        actions_np = action.detach().cpu().numpy()
        if single_env:
            return actions_np[0]  # legacy behaviour: 1-D array
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
                
                # Forward pass
                action_means, state_values = self.actor_critic(batch_states)
                action_var = torch.full((action_means.size(-1),), self.action_std**2)
                cov_mat = torch.diag(action_var).unsqueeze(0)
                
                dist = MultivariateNormal(action_means, covariance_matrix=cov_mat)
                action_logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy()
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
                torch.nn.utils.clip_grad_norm_( [p for group in self.actor_optimizer.param_groups for p in group['params']], max_norm=0.25)
                self.actor_optimizer.step()

                # ---------------------------------------------
                # Optimise critic
                # ---------------------------------------------
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.critic_loss_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_( [p for group in self.critic_optimizer.param_groups for p in group['params']], max_norm=0.25)
                self.critic_optimizer.step()

                # Approximate KL divergence for early stopping & adaptive LR
                approx_kl = (batch_logprobs - action_logprobs).mean().abs()

                # Adapt actor learning-rate based on KL (simple heuristic)
                if approx_kl > 0.03:
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.9, 1e-5)
                elif approx_kl < 0.015:
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = min(g["lr"] * 1.1, self.actor_lr_init)

                # Anneal clip range after enough training steps
                if self.total_steps > 300_000:
                    self.eps_clip = 0.15

                # If policy changed too much, break out early
                if approx_kl > 0.04:
                    break

    # -------------------------------------------------
    # Utility to freeze actor once task is solved
    # -------------------------------------------------
    def freeze_actor(self, lr: float = 1e-6):
        """Freeze actor by dropping its learning-rate and turning off exploration noise."""
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr
        # Make subsequent actions deterministic
        self.action_std = 0.0
        self.min_action_std = 0.0

def train(env_name, max_episodes, max_timesteps, _update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, num_envs=1):
    """Train with single or parallel environments (vectorised when num_envs>1)."""
    timestep = 0
    # Seeding for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Dynamic update_timestep proportional to number of envs
    update_timestep = num_envs * 1024  # larger batch per PPO update

    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)

    ppo = PPO(actor_critic, lr_actor, lr_critic, gamma, gae_lambda, K_epochs, eps_clip, action_std, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, entropy_coef_final, entropy_decay_steps)

    # Running reward normaliser
    reward_norm = RunningNorm()

    memory = Memory()
    
    episode_returns = []
    running_avg_returns = []
    eval_scores = []  # deterministic evaluation scores after solving
    
    if PLOT:
        plt.ion()
        fig, ax = plt.subplots()
    
    # ------------------------------------------------------------------
    # Environment setup (vectorised if num_envs > 1)
    # ------------------------------------------------------------------
    if num_envs == 1:
        env = gym.make(env_name)  # training environment without rendering
        states, _ = env.reset()
    else:
        # AsyncVectorEnv expects a list of callables that build environments
        env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    if num_envs == 1:
        # Retain the original (simpler) single-environment logic for clarity
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            for _ in range(max_timesteps):
                timestep += 1

                action = ppo.select_action(state, memory)
                next_state, reward, done, trunc, _ = env.step(action)

                # Update reward normaliser and store normalised reward
                reward_norm.update(reward)
                memory.rewards.append(reward_norm.normalize(reward))
                memory.is_terminals.append(float(done or trunc))
                total_reward += reward

                if timestep % update_timestep == 0:
                    ppo.update(memory); memory.clear_memory(); timestep = 0

                state = next_state
                if done or trunc:
                    break

            episode_returns.append(total_reward)
            running_avg_returns.append(np.mean(episode_returns[-log_interval:]))

            if episode % log_interval == 0:
                print(f'ep {episode:6} return {total_reward:.2f}')
                if PLOT:
                    ax.clear(); ax.plot(episode_returns, label='Returns');
                    ax.plot(running_avg_returns, label='Running Avg');
                    if eval_scores:
                        ax.plot(range(len(eval_scores)), eval_scores, label='Eval', color='green')
                    ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
                if RENDER and episode % eval_interval == 0:
                    eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps)
                    print(f'            eval return {eval_ret:.2f}')
                    eval_scores.append(eval_ret)

            # Freeze actor once solved (>200 average over last 100 episodes)
            if len(episode_returns) >= 100 and not hasattr(ppo, "actor_frozen"):
                if np.mean(episode_returns[-100:]) > 200:
                    ppo.freeze_actor()
                    ppo.actor_frozen = True
                    print("Actor frozen – task SOLVED; running one human-rendered evaluation...")
                    evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1)
                    return

        if PLOT:
            plt.ioff(); plt.show(); return

    # ---------- Rollout collection loop (vectorised) -----------
    states, _ = env.reset()
    per_env_returns = np.zeros(num_envs)
    completed_episodes = 0

    while completed_episodes < max_episodes:
        timestep += 1

        actions = ppo.select_action(states, memory)  # (num_envs, action_dim)
        next_states, rewards, terminated, truncated, _ = env.step(actions)

        # Update normaliser with batch rewards and store normalised
        # Reward clipping for stability (vectorised)
        rewards = np.clip(rewards, -10.0, 10.0)
        reward_norm.update(rewards)
        memory.rewards.append(reward_norm.normalize(rewards))
        memory.is_terminals.append(np.logical_or(terminated, truncated).astype(float))

        per_env_returns += rewards

        # When any env finishes an episode
        done_mask = np.logical_or(terminated, truncated)
        if np.any(done_mask):
            # Log and bookkeeping per finished env
            for idx in np.where(done_mask)[0]:
                episode_returns.append(per_env_returns[idx])
                per_env_returns[idx] = 0.0
                completed_episodes += 1

                if completed_episodes % log_interval == 0:
                    print(f'ep {completed_episodes:6} return {episode_returns[-1]:.2f}')

                    if PLOT:
                        ax.clear()
                        ax.plot(episode_returns, label='Returns')
                        running_avg = np.convolve(episode_returns, np.ones(log_interval)/log_interval, mode='valid')
                        ax.plot(range(log_interval - 1, len(episode_returns)), running_avg, label='Running Avg')
                        if eval_scores:
                            ax.plot(range(len(eval_scores)), eval_scores, label='Eval', color='green')
                        ax.legend(); ax.set_xlabel('Episode'); ax.set_ylabel('Return'); plt.pause(0.01)
                    if RENDER and completed_episodes % eval_interval == 0:
                        eval_ret = evaluate_policy(env_name, actor_critic, max_timesteps)
                        print(f'            eval return {eval_ret:.2f}')
                        eval_scores.append(eval_ret)

                    # Check solved condition in vectorised path
                    if len(episode_returns) >= 100 and not hasattr(ppo, "actor_frozen"):
                        if np.mean(episode_returns[-100:]) > 200:
                            ppo.freeze_actor()
                            ppo.actor_frozen = True
                            print("Actor frozen – task SOLVED; running one human-rendered evaluation...")
                            evaluate_policy(env_name, actor_critic, max_timesteps, episodes=1)
                            return

            # Reset finished environments
            if hasattr(env, 'reset_done'):
                # Gymnasium ≥0.28 provides this convenience method
                states_reset, _ = env.reset_done()
                next_states[done_mask] = states_reset
            else:
                # Fallback – reset *all* envs (simpler, but slightly wasteful)
                next_states, _ = env.reset()

        # Time to update PPO?
        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        states = next_states

    if PLOT:
        plt.ioff()
        plt.show()

    # Also ensure pyglet event loop exits (Box2D viewer)
    if 'pyglet' in sys.modules:
        try:
            import pyglet
            pyglet.app.exit()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Utility for occasional evaluation with human-rendering
# -----------------------------------------------------------------------------

def _eval_worker(conn, state_dict, env_name, max_timesteps):
    """Runs *one* deterministic evaluation episode in a subprocess."""
    import torch, gymnasium as gym
    from math import sqrt

    # Reconstruct model
    state_dim = 8
    action_dim = 2
    hidden_dim = 256
    model = ActorCritic(state_dim, action_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(max_timesteps):
            action_mean, _ = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            state, reward, done, trunc, _ = env.step(action_mean.squeeze(0).cpu().numpy())
            total_reward += reward
            if done or trunc:
                break

    env.close()
    conn.send(total_reward)
    conn.close()


def evaluate_policy(env_name, actor_critic, max_timesteps, episodes: int = 5):
    """Average return over a few deterministic episodes (subprocess, human render)."""

    state_dict = {k: v.cpu() for k, v in actor_critic.state_dict().items()}
    returns = []
    for _ in range(episodes):
        parent_conn, child_conn = _mp.Pipe()
        p = _mp.Process(target=_eval_worker, args=(child_conn, state_dict, env_name, max_timesteps))
        p.start()
        ret = parent_conn.recv(); p.join()
        returns.append(ret)
    return float(np.mean(returns))

if __name__ == '__main__':
    train(env_name, max_episodes, max_timesteps, update_timestep, log_interval, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std, gae_lambda, ppo_loss_coef, critic_loss_coef, entropy_coef, batch_size, num_envs=num_envs)
