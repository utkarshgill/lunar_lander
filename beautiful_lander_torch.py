import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import matplotlib.pyplot as plt


DEBUG = False


def train(
    env, policy, optim, discount, lambda_, ppo_steps, ppo_clip, entropy, batch_size
):
    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done, trunc = False, False
    ep_return = 0.0

    state, info = train_env.reset()

    # with record_function("run_train_env"):
    while not done and not trunc:
        print(f"state {state.shape, state.dtype}") if DEBUG else None
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"state {state.shape, state.dtype}") if DEBUG else None
        states.append(state)

        actor_logits, value_pred = policy(state)
        print(
            f"action_pred{action_pred.shape, action_pred.dtype} value_pred {value_pred, value_pred.dtype}"
        ) if DEBUG else None
        action_prob = F.softmax(actor_logits, dim=-1)
        dist = torch.distributions.Categorical(action_prob)
        # print(f"cat dist{dist.shape, dist.dtype}")
        action = dist.sample()

        print(f"action {action.shape, action.dtype}") if DEBUG else None
        log_prob_action = dist.log_prob(action)

        print(
            f"log_prob_action {log_prob_action.shape, log_prob_action.dtype}"
        ) if DEBUG else None
        state, reward, done, trunc, info = env.step(action.item())

        actions.append(action)
        # print(log_prob_actions.shape)
        log_prob_actions.append(log_prob_action)
        # print(value_pred.shape, value_pred.unsqueeze(0).shape)
        values.append(value_pred.squeeze(-1))
        rewards.append(reward)

        ep_return += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values)
    # rewards = torch.cat(rewards)
    # print(states.shape, values.shape)

    returns = calc_returns(rewards, discount)

    print(f"ret {returns.shape, returns.dtype}") if DEBUG else None
    value_loss = update_critic(values, returns)

    advantages = calc_advantages(rewards, values, discount, lambda_)
    print(f"adv {advantages.shape, advantages.dtype}") if DEBUG else None

    # with record_function("update_policy"):

    return (
        states,
        actions,
        log_prob_actions,
        values,
        returns,
        advantages,
        ep_return,
        value_loss,
    )


def calc_advantages(rewards, values, discount, trace_decay, normalize=True):
    # print(returns.shape, values.squeeze(-1).shape)

    # GAE advantages
    rewards = torch.tensor(rewards)
    # advs = torch.zeros_like(rewards, dtype=torch.float32)
    # last_adv = 0
    # last_val = values[-1]
    # for t in reversed(range(rewards.shape[0])):
    #     delta = rewards[t] + gamma * last_val - values[t]
    #     last_adv = delta + gamma * lambda_ * last_adv
    #     advs[t] = last_adv
    #     last_val = values[t]
    # return advs

    advantages = []
    advantage = 0
    next_value = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount - v
        advantage = td_error + advantage * discount * trace_decay
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages)

    # simple advantages
    # advs = rewards + discount * values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def calc_returns(rewards, discount, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def update_critic(values, returns):
    optim.zero_grad(set_to_none=True)
    loss = F.mse_loss(values, returns)
    loss.backward()
    optim.step()
    return loss.item()


def update_policy(
    policy,
    states,
    actions,
    log_prob_actions,
    advantages,
    returns,
    optim,
    ppo_steps,
    ppo_clip,
    entropy,
    batch_size,
):
    total_policy_loss = 0
    total_value_loss = 0

    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    # print(
    #     states.shape,
    #     actions.shape,
    #     log_prob_actions.shape,
    #     returns.shape,
    #     advantages.shape,
    # )
    memory = TrainDataset(states, actions, log_prob_actions, advantages, returns)
    # print(advantages.dtype)

    train_dataloader = DataLoader(
        memory,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    for _ in range(ppo_steps):
        for ste, act, lpa, ret, adv in train_dataloader:
            # print(states.shape[0])
            # samp = np.random.randint(0, states.shape[0], (batch_size,))
            ste, act, lpa, ret, adv = (
                ste.to(device),
                act.to(device),
                lpa.to(device),
                ret.to(device),
                adv.to(device),
            )
            print("ppo step start") if DEBUG else None
            actor_logits, value_pred = policy(ste)
            # print(ste.shape, ret.shape)
            # value_pred = value_pred.squeeze(-1)
            log_softmax = F.log_softmax(actor_logits, dim=-1)
            action_prob = F.softmax(actor_logits, dim=-1)
            dist = torch.distributions.Categorical(action_prob)

            print("ppo_action_pred") if DEBUG else None
            # new log prob using old actions
            new_log_prob_actions = dist.log_prob(act)

            # print(new_log_prob_actions.dtype)

            # ppo-clip
            policy_ratio = (new_log_prob_actions - lpa).exp()
            policy_loss_1 = policy_ratio * ret
            policy_loss_2 = (
                torch.clip(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip) * ret
            )
            entropy_loss = (log_softmax.exp() * log_softmax).sum(-1).mean()
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean(-1).mean()
            # policy_loss = -(lpa * adv).mean(-1).mean()
            # policy_loss.requires_grad = True
            # print(ret.shape, value_pred.squeeze(-1).shape)
            # value_loss = F.mse_loss(ret, value_pred.squeeze(-1)).float()
            # print(value_loss.shape)

            print("ppo_policy_loss") if DEBUG else None
            # print(policy_loss.dtype, value_loss.dtype)

            optim.zero_grad(set_to_none=True)

            (policy_loss + entropy_loss * entropy).backward()

            optim.step()

            print("backward and step") if DEBUG else None
            total_policy_loss += policy_loss.item()
            # total_value_loss += value_loss.item()

            print("ppo step end") if DEBUG else None
    return total_policy_loss / ppo_steps  # , total_value_loss / ppo_steps


def eval(env, policy):
    policy.eval()

    print("eval start") if DEBUG else None
    rewards = []
    done, trunc = False, False
    ep_reward = 0

    state, info = env.reset()

    print("step start") if DEBUG else None
    while not done and not trunc:
        print("state") if DEBUG else None
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_pred, value_pred = policy(state)

            print("action_pred") if DEBUG else None
            action_prob = F.softmax(action_pred, dim=-1)

            print("action_prob") if DEBUG else None
        action = torch.argmax(action_prob, dim=-1)

        print("action") if DEBUG else None
        state, reward, done, trunc, info = env.step(action.item())

        print("step") if DEBUG else None
        ep_reward += reward

    print("eval done") if DEBUG else None
    env.close()
    return ep_reward


class DQN(nn.Module):
    def __init__(self, fan_in, n_hidden, fan_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(fan_in, n_hidden, bias=True),
            nn.PReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.PReLU(),
            nn.Linear(n_hidden, fan_out, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class Head(nn.Module):
    def __init__(self, block_size, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embd, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embd, head_size, bias=False, device=device)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, device=device))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = k @ q.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embd, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, device=device),
            nn.PReLU(),
            nn.Linear(4 * n_embd, n_embd, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, n_embd, n_head, head_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, obs_space, act_space, block_size, n_embd, n_head, n_layer, dropout
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(obs_space, n_embd, device=device)
        self.pos_embedding = nn.Embedding(obs_space, n_embd, device=device)
        self.blocks = nn.Sequential(
            *[Block(block_size, n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd, device=device)
        self.head = nn.Linear(n_embd, act_space, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        # if targets is None:
        #     loss = None
        # else
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     loss = F.cross_entropy(logits, targets)

        return logits  # , loss


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


class TrainDataset(Dataset):
    def __init__(self, states, actions, log_prob_actions, returns, advantages):
        self.states = states
        self.actions = actions
        self.log_prob_actions = log_prob_actions
        self.returns = returns
        self.advantages = advantages

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        ste = self.states[idx]
        act = self.actions[idx]
        lpa = self.log_prob_actions[idx]
        ret = self.returns[idx]
        adv = self.advantages[idx]
        return ste, act, lpa, ret, adv


def update_plot(train_rewards, graph):
    graph.remove()
    graph = plt.plot(train_rewards, color="green")[0]
    plt.hlines(REWARD_TARGET, 0, len(train_rewards), color="orange")
    plt.ylim(-300, 300)
    plt.draw()
    plt.pause(0.1)


if __name__ == "__main__":
    device = (
        torch.device("cpu")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    train_env = gym.make("LunarLander-v2")
    test_env = gym.make("LunarLander-v2", render_mode="human")

    N_EMBD = 64
    N_HEAD = 6
    N_LAYER = 6
    BLOCK_SIZE = 50
    DROP = 0.2
    INPUT_DIM = train_env.observation_space.shape[0]
    # HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    # actor = Transformer(
    #     INPUT_DIM, OUTPUT_DIM, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROP
    # )
    actor = DQN(INPUT_DIM, N_EMBD, OUTPUT_DIM)
    critic = DQN(INPUT_DIM, N_EMBD, 1)

    lander = ActorCritic(actor, critic)
    lander.to(device)

    # LEARN_RATE = 0.0005
    optim = torch.optim.AdamW(lander.parameters())

    MAX_EPS = 2000
    BUFFER_SIZE = 250
    BATCH_SIZE = 32
    DISCOUNT = 0.995
    LAMBDA = 0.95
    N_TRIALS = 100
    REWARD_TARGET = 200
    PPO_STEPS = 5
    PPO_CLIP = 0.2
    ENTROPY = 0.0001

    ep_returns = []
    value_loss_i = []

    batch_states = []
    batch_actions = []
    batch_log_prob_actions = []
    batch_values = []
    batch_returns = []
    batch_advantages = []
    # update_plot(train_rewards)
    # graph = plt.plot
    # plt.plot()
    # plt.ion()
    # plt.plot()
    # test_rewards = []

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:

    policy_loss = 0.0
    value_loss = 0.0
    for episode in (t := trange(1, MAX_EPS + 1)):
        print("ep start") if DEBUG else None

        print("train") if DEBUG else None

        (
            states,
            actions,
            log_prob_actions,
            values,
            returns,
            advantages,
            ep_return,
            value_loss,
        ) = train(
            train_env,
            lander,
            optim,
            DISCOUNT,
            LAMBDA,
            PPO_STEPS,
            PPO_CLIP,
            ENTROPY,
            BATCH_SIZE,
        )

        # print(
        #     states.shape,
        #     actions.shape,
        #     log_prob_actions.shape,
        #     values.shape,
        #     returns.shape,
        #     advantages.shape,
        #     ep_return.shape,
        # )

        ep_returns.append(ep_return)
        value_loss_i.append(value_loss)

        batch_states.append(states)
        batch_actions.append(actions)
        batch_log_prob_actions.append(log_prob_actions)
        batch_values.append(values)
        batch_returns.append(returns)
        batch_advantages.append(advantages)

        if episode % BUFFER_SIZE == 0:
            states = torch.cat(batch_states)
            actions = torch.cat(batch_actions)
            log_prob_actions = torch.cat(batch_log_prob_actions)
            advantages = torch.cat(batch_advantages)
            returns = torch.cat(batch_returns)

            policy_loss = update_policy(
                lander,
                states,
                actions,
                log_prob_actions,
                advantages,
                returns,
                optim,
                PPO_STEPS,
                PPO_CLIP,
                ENTROPY,
                BATCH_SIZE,
            )
            batch_states = []
            batch_actions = []
            batch_log_prob_actions = []
            batch_values = []
            batch_returns = []
            batch_advantages = []

        print("mean") if DEBUG else None
        mean_ep_returns = np.mean(ep_returns[-N_TRIALS:])

        if not DEBUG:
            t.set_description(
                f"episode {episode:3} mean_ep_return {mean_ep_returns:7.1f} value_loss {value_loss:7.2f}"
            )

        if mean_ep_returns >= REWARD_TARGET:
            print(f"Reached reward threshold in {episode} episodes")
            break

    plt.subplot(2, 1, 1)
    plt.plot(ep_returns, label="returns")
    plt.subplot(2, 1, 2)
    plt.plot(value_loss_i, label="value loss")
    plt.show()
    eval(test_env, lander)
