import torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as distributions
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import time

st = time.time()

if __name__ == '__main__':
    
    DEVICE = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    ENV_NAME = "CartPole-v1"
    ENV_COUNT = 64
    # env = gym.make(ENV_NAME)
    env = gym.make_vec(ENV_NAME, num_envs=ENV_COUNT)
    # hyperparams
    # print(env.observation_space.shape[-1],)
    INPUTS = env.observation_space.shape[-1]
    OUTPUTS = env.action_space[0].n
    CHANNELS = 128 
    BUFFER_SIZE = 1024
    BATCH_SIZE = 8
    UPDATE_STEPS = 10
    EPOCHS = 1000
    LEARN_RATE = 1e-5
    DISCOUNT = 0.995
    WINDOW = 50
    TARGET = 475
    RENDER = False
    PPO_CLIP = 0.5

    plt.ion()

    def rtg(rewards):
        N = len(rewards)
        rtg = np.zeros_like(rewards)
        for i in reversed(range(N)):
            rtg[i] = rewards[i] + DISCOUNT * (rtg[i+1] if i+1<N else 0)
        return list(rtg)

    def parallel_rtg(rewards, dones):
        N = len(rewards)
        mask = (~np.array(dones)).astype(np.float32)
        rtg = np.zeros_like(rewards)
        for i in reversed(range(N)):
            rtg[i] = (rewards[i] + DISCOUNT*(rtg[i+1] if i+1 < N else 0)) * mask[i]
        return list(rtg)

    def plot(t, ep_ret):

        all_rets.append(ep_ret)
        mean_rets.append(np.mean(all_rets.copy()) if len(all_rets) < WINDOW else np.mean(all_rets.copy()[-WINDOW:]))
        max_rets.append(max_rets[-1] if len(max_rets) > 0 and ep_ret < max_rets[-1] else ep_ret)

        # if ep_ret > TARGET:
        #     print(f'WIN: scored {ep_ret:8.2f} in {steps} steps')

        t.set_description(f'return {all_rets[-1]:6.2f}       avg {mean_rets[-1]:6.2f}       max {max_rets[-1]:6.2f}')
        # plt.subplot(2, 1, 1)
        plt.plot(all_rets)
        plt.plot(max_rets)
        plt.plot(mean_rets)
        plt.title('PERFORMANCE')
        plt.xlabel('episodes')
        plt.ylabel('returns')

        # plt.subplot(2, 1, 2)
        # plt.plot(critic_losses)
        # plt.title('VALUE LOSS')
        # plt.xlabel('episodes')
        # plt.ylabel('loss')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.2)

        plt.clf()

    class Memory(Dataset): 
        def __init__(self, states, actions, weights, logprobs): 
            self.states, self.actions, self.weights, self.logprobs = states, actions, weights, logprobs

        def __len__(self): 
            return len(self.states)
    
        def __getitem__(self, index): 
            return self.states[index], self.actions[index], self.weights[index], self.logprobs[index]

    class DQN(nn.Module):
        def __init__(self, in_dim, channels, out_dim):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, channels),
                nn.Tanh(),
                nn.Linear(channels, channels // 2),
                nn.Tanh(),
                nn.Linear(channels // 2, channels // 3),
                nn.Tanh(),
                nn.Linear(channels // 3, out_dim),
            )

        def forward(self, state):
            out = self.block(state)
            return out.squeeze(-1) if out.shape[-1] == 1 else out

    class ActorCritic(nn.Module):
        def __init__(self, actor, critic):
            super().__init__()
            self.actor = actor
            self.critic = critic
            
        def forward(self, state):
            return distributions.Categorical(logits= self.actor(state)), self.critic(state)
            
        def act(self, state):
            policy, value = self(state)
            action = policy.sample()
            logprob = policy.log_prob(action)
            return action.detach().cpu().numpy(), logprob.detach().cpu().numpy(), value.detach().cpu().numpy()
    
    actor = DQN(INPUTS, CHANNELS, OUTPUTS)
    critic = DQN(INPUTS, CHANNELS, 1)
    agent = ActorCritic(actor, critic)
    optim = torch.optim.AdamW(agent.parameters(), lr=LEARN_RATE)
    agent.to(DEVICE)
    #plot variables
    max_rets = []
    all_rets = []
    mean_rets = []
    critic_losses = []

    # def GAE(dones, rewards, values):

    #     return

    def compute_loss(states, actions, weights, logprobs):
        global critic_losses

        # samp = np.random.randint(0, len(states), (BATCH_SIZE,))
        # batch_states = torch.tensor(np.array(states)).float().to(DEVICE)[samp] 
        # batch_actions = torch.tensor(np.array(actions)).float().to(DEVICE)[samp]
        # batch_weights = torch.from_numpy(np.array(weights)).float().to(DEVICE)[samp]
        # batch_logprobs = torch.from_numpy(np.array(logprobs)).float().to(DEVICE)[samp]

        policy, values = agent(states)
        
        new_logp = policy.log_prob(actions)
        old_logp = logprobs
        advantage = weights - values
        advantage = (advantage - advantage.mean())/ advantage.std()

        # print(states.shape, weights.shape, actions.shape, values.shape, advantage.shape)
        actor_loss =  - (new_logp * advantage).mean()
        # e = PPO_CLIP
        # ratio = torch.exp(new_logp - old_logp) * advantage
        # clipped = torch.clamp(ratio, min=1-e, max=1+e) * advantage
        # (ratio.shape, clipped.shape, torch.min(ratio, clipped).shape)
        # actor_loss = - torch.min(ratio, clipped).mean(-1).mean()

        # print(actor_loss.shape)

        critic_loss = F.mse_loss(values, weights)

        # critic_losses.append(critic_loss.detach())
        # plot()

        loss = actor_loss + critic_loss
        return loss

    steps = 0
    epoch_render_done = False
    def train_one_epoch(t):
        global all_rets
        global max_rets
        global mean_rets
        global epoch_render_done 
        global steps

        batch_states = []
        batch_actions = []
        batch_values = []
        batch_weights = []
        batch_logprobs = []

        ep_rews = []
        ep_dones = []
        acc = [0] * ENV_COUNT

        state, _ = env.reset()

        while 1:
            steps += 1
            batch_states.append(state)

            action, value, logprob = agent.act(torch.tensor(state, dtype=torch.float32, device=DEVICE))
            state, reward, done, trunc, _ = env.step(action)

            batch_logprobs.append(logprob)
            batch_values.append(value)
            batch_actions.append(action)
            ep_rews.append(reward)
            ep_dones.append(done)

            acc += reward

            if done[0]:
                ep_ret = acc[0]
                acc[0] = 0
                plot(t, ep_ret)
                state, _ = env.reset()

            if len(batch_states) >= BUFFER_SIZE:
                batch_weights += parallel_rtg(ep_rews, ep_dones)
                state, _ = env.reset()
                break

        rollouts = Memory(torch.tensor(np.array(batch_states), dtype=torch.float32, device=DEVICE).view(-1,INPUTS),
                          torch.tensor(np.array(batch_actions), dtype=torch.int32, device=DEVICE).view(-1),
                          torch.tensor(np.array(batch_weights), dtype=torch.float32, device=DEVICE).view(-1),
                          torch.tensor(np.array(batch_logprobs), dtype=torch.float32, device=DEVICE).view(-1))

        dataloader = DataLoader(rollouts, batch_size=BATCH_SIZE, shuffle=1, num_workers=0) 

        for states, actions, weights, logprobs in dataloader: 
            # print(batch[3].shape)
            loss = compute_loss(states, actions, weights, logprobs)
            optim.zero_grad(set_to_none=1)
            loss.backward()
            optim.step() 

   


    def eval():
        
        wins = 0
        total_ep = 100
        env = gym.make(ENV_NAME)
        state, _ = env.reset()
        for i in range(total_ep):
            ep_ret = 0
            while 1:
                action, _, _ = agent.act(torch.tensor(state, dtype=torch.float32, device=DEVICE))
                state, reward, done, trunc, _ = env.step(action.cpu().item())
                ep_ret += reward
                if done or trunc:
                    
                    if ep_ret > TARGET:
                        print("eval WIN")
                        wins += 1
                    else:
                        print("eval LOST")

                    state, _ = env.reset()
                    break

        print(f'accuracy {wins/total_ep:8.2f}')
        env.close()
        
        env = gym.make(ENV_NAME, render_mode="human")
        state, _ = env.reset()
        for i in range(3):
            while 1:
                action, _, _ = agent.act(torch.tensor(state, dtype=torch.float32, device=DEVICE))
                state, reward, done, trunc, _ = env.step(action.cpu().item())
                if done or trunc:
                    state, _ = env.reset()
                    break

        env.close()

    def train():
        for ep in (t:=trange(EPOCHS)):
            # st = time.time()
            # print(f'EPOCH {ep}')
            train_one_epoch(t)
            # et = time.time()
            # print(f'\n{(et-st)}')

    train()

    env.close()

    plt.ioff()
    plt.plot(all_rets)
    plt.plot(max_rets)
    plt.plot(mean_rets)
    plt.show()


    eval()



print