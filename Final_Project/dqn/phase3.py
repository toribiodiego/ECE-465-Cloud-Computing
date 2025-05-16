#!/usr/bin/env python3
"""
phase3.py: Distributed DQN Prototype with Checkpointing & Evaluation

- DQN: MLP (19→128→9), Adam + Huber
- RolloutActors collect transitions only
- Head: replay buffer, minibatch updates, target-sync
- Silent checkpointing every N episodes
- After training: evaluate each checkpoint vs random & self play, count W/D/L
- Prints metrics + fixed Markdown table + raw vs TWEMA line plot
"""

import argparse
import os
import random
import time
from collections import deque

import gym
import numpy as np
import ray
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 1) Seeding helper
def set_global_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)

# 2) TicTacToe Env & Wrapper
class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    symbols = ['O', ' ', 'X']
    def __init__(self):
        super().__init__()
        self.action_space      = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(54)
        self.state = None
    def seed(self, seed=None):
        random.seed(seed); np.random.seed(seed)
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.state = {'board':[0]*9, 'on_move':1}
        return self.state
    def step(self, action):
        p, sq = action
        b, om = self.state['board'], self.state['on_move']
        if b[sq] != 0 or p != om:
            return self.state, -1.0, True, False, {'illegal_move':True}
        b[sq], self.state['on_move'] = p, -p
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        if any(b[i]==p and b[j]==p and b[k]==p for i,j,k in lines):
            return self.state, 1.0, True, False, {'win':True}
        if all(cell!=0 for cell in b):
            return self.state, 0.0, True, False, {'draw':True}
        return self.state, 0.0, False, False, {}
    def render(self, mode='human', close=False):
        if close: return
        print("On move:", "X" if self.state['on_move']==1 else "O")
        for i in range(9):
            print(self.symbols[self.state['board'][i]+1], end=' ')
            if i % 3 == 2:
                print()

class TicTacToeWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = TicTacToeEnv()
        self.observation_space = gym.spaces.Box(-1,1,shape=(19,),dtype=np.float32)
        self.action_space      = gym.spaces.Discrete(9)
    def seed(self, seed=None):
        self.env.seed(seed)
    def reset(self, seed=None, **kwargs):
        state = self.env.reset(seed=seed, **kwargs)
        b    = np.array(state['board'],dtype=np.float32)
        om   = np.array([state['on_move']],dtype=np.float32)
        mask = np.array([1.0 if c==0 else 0.0 for c in state['board']],dtype=np.float32)
        return np.concatenate([b,om,mask]), {}
    def step(self, idx:int):
        p = self.env.state['on_move']
        s, r, done, _, info = self.env.step((p, idx))
        b    = np.array(s['board'],dtype=np.float32)
        om   = np.array([s['on_move']],dtype=np.float32)
        mask = np.array([1.0 if c==0 else 0.0 for c in s['board']],dtype=np.float32)
        return np.concatenate([b,om,mask]), float(r), done, False, info

# 3) DQN Network
class DQN(nn.Module):
    def __init__(self, obs_dim=19, hidden=128, n_actions=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# 4) Rollout Actor (no learning)
@ray.remote
class RolloutActor:
    def __init__(self, weights, seed, epsilon):
        random.seed(seed); np.random.seed(seed)
        self.env     = TicTacToeWrapper(); self.env.seed(seed)
        self.epsilon = epsilon
        self.device  = torch.device("cpu")
        self.policy  = DQN().to(self.device)
        self.policy.load_state_dict(weights)

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def run(self, episodes):
        trajs, rews = [], []
        for _ in range(episodes):
            obs,_ = self.env.reset(); done,total = False,0.0
            while not done:
                if random.random() < self.epsilon:
                    a = random.randrange(self.env.action_space.n)
                else:
                    with torch.no_grad():
                        t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        a = int(self.policy(t).argmax().item())
                nxt, r, done, _, _ = self.env.step(a)
                trajs.append((obs,a,r,nxt,done))
                total += (r if self.env.env.state['on_move']==-1 else 0)
                obs = nxt
            rews.append(total)
        return trajs, rews

# 5) Distributed DQN Training + Silent Checkpoints
def distributed_dqn(
    total_eps, wave_eps, num_actors, epsilon,
    replay_size, batch_size, gamma,
    train_freq, target_update, broadcast_freq,
    checkpoint_dir, checkpoint_interval
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net  = DQN().to(device)
    target_net  = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer   = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn     = nn.SmoothL1Loss()
    memory      = deque(maxlen=replay_size)
    trans_ct = upd_ct = proc = wave = last_ckpt = 0
    ckpts = []

    ray.init(ignore_reinit_error=True)
    actors = [
        RolloutActor.remote(policy_net.state_dict(), 100+i, epsilon)
        for i in range(num_actors)
    ]

    dq, curve = deque(maxlen=100), []
    pbar = tqdm(total=total_eps, desc="Distributed-DQN")

    while proc < total_eps:
        if wave % broadcast_freq == 0:
            w = policy_net.state_dict()
            for a in actors:
                a.set_weights.remote(w)

        results = ray.get([a.run.remote(wave_eps) for a in actors])
        wave += 1

        for traj, rews in results:
            for obs,a,r,nxt,done in traj:
                memory.append((obs,a,r,nxt,done)); trans_ct+=1
                if trans_ct>=replay_size and trans_ct%train_freq==0:
                    batch   = random.sample(memory, batch_size)
                    obs_arr = np.stack([b[0] for b in batch])
                    nxt_arr = np.stack([b[3] for b in batch])
                    obs_b   = torch.tensor(obs_arr, dtype=torch.float32, device=device)
                    act_b   = torch.tensor([b[1] for b in batch],dtype=torch.int64,device=device)
                    rew_b   = torch.tensor([b[2] for b in batch],dtype=torch.float32,device=device)
                    nxt_b   = torch.tensor(nxt_arr, dtype=torch.float32,device=device)
                    done_b  = torch.tensor([b[4] for b in batch],dtype=torch.float32,device=device)

                    q_vals   = policy_net(obs_b).gather(1,act_b.unsqueeze(-1)).squeeze(-1)
                    next_q   = target_net(nxt_b).max(1)[0].detach()
                    target_q = rew_b + gamma*next_q*(1-done_b)

                    loss = loss_fn(q_vals, target_q)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    upd_ct +=1
                    if upd_ct % target_update == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            for r in rews:
                proc+=1; dq.append(r); curve.append(np.mean(dq)); pbar.update(1)
                if proc-last_ckpt>=checkpoint_interval:
                    path = os.path.join(checkpoint_dir, f"ckpt_{proc}.pt")
                    torch.save(policy_net.state_dict(), path)
                    ckpts.append(path)
                    last_ckpt = proc
            if proc>=total_eps:
                break

    pbar.close(); ray.shutdown()
    return curve, upd_ct, ckpts

# 6) TWEMA smoothing
def twema(curve, s):
    if not curve: return []
    n, w = len(curve), min(np.sqrt(s),0.999)
    last_y = debias = 0.0; rng = n-1 or 1
    out = []
    for y in curve:
        last_y = last_y*(w**(1/rng)) + y
        debias  = debias*(w**(1/rng)) + 1
        out.append(last_y/debias)
    return out

# 7) Evaluation utility (proper W/D/L)
def evaluate_policy(state_dict, env, episodes, opponent):
    device     = torch.device("cpu")
    net        = DQN().to(device)
    net.load_state_dict(state_dict)
    net.eval()
    wins = draws = losses = 0
    for _ in range(episodes):
        obs,_ = env.reset(); done=False
        while not done:
            p = env.env.state['on_move']
            if p==1:
                t = torch.tensor(obs, dtype=torch.float32)
                a = int(net(t).argmax().item())
            else:
                if opponent=='random':
                    a = random.choice([i for i,m in enumerate(env.env.state['board']) if m==0])
                else:
                    t = torch.tensor(obs, dtype=torch.float32)
                    a = int(net(t).argmax().item())
            prev_p = p
            obs, _, done, _, info = env.step(a)
        if 'win' in info and prev_p==1:
            wins += 1
        elif 'draw' in info:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses

# 8) Main & Plot
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',            type=int,   default=100_000)
    parser.add_argument('--wave',                type=int,   default=500)
    parser.add_argument('--actors',              type=int,   default=4)
    parser.add_argument('--epsilon',             type=float, default=0.1)
    parser.add_argument('--replay-size',         type=int,   default=10_000)
    parser.add_argument('--batch-size',          type=int,   default=32)
    parser.add_argument('--gamma',               type=float, default=0.99)
    parser.add_argument('--train-freq',          type=int,   default=4)
    parser.add_argument('--target-update',       type=int,   default=100)
    parser.add_argument('--broadcast-freq',      type=int,   default=1)
    parser.add_argument('--ma-window',           type=float, default=1.0)
    parser.add_argument('--checkpoint-dir',      type=str,   default='./checkpoints')
    parser.add_argument('--checkpoint-interval', type=int,   default=10_000)
    parser.add_argument('--eval-episodes',       type=int,   default=1_000)
    args = parser.parse_args()

    set_global_seed(42)
    start = time.time()
    curve, grad_updates, ckpts = distributed_dqn(
        args.episodes, args.wave, args.actors, args.epsilon,
        args.replay_size, args.batch_size, args.gamma,
        args.train_freq, args.target_update, args.broadcast_freq,
        args.checkpoint_dir, args.checkpoint_interval
    )
    elapsed   = time.time() - start
    eps_per_s = args.episodes / elapsed
    sm_curve  = twema(curve, args.ma_window)

    # Performance
    print("\n=== Performance ===")
    print(f"Time:         {elapsed:.2f}s")
    print(f"Episodes/sec: {eps_per_s:.1f}")
    print(f"Grad updates: {grad_updates}\n")

    # Checkpoint W/D/L table
    env = TicTacToeWrapper()
    print("=== Checkpoint W/D/L Counts (1000 eps) ===")
    print(f"|{'checkpoint':15}|{'vs random':13}|{'vs self':13}|")
    print(f"|{'-'*15}|{'-'*13}|{'-'*13}|")
    for ckpt in ckpts:
        sd        = torch.load(ckpt, map_location='cpu')
        w_r,d_r,l_r = evaluate_policy(sd, env, args.eval_episodes, 'random')
        w_s,d_s,l_s = evaluate_policy(sd, env, args.eval_episodes, 'self')
        name      = os.path.basename(ckpt)
        left      = f"{w_r}/{d_r}/{l_r}"
        right     = f"{w_s}/{d_s}/{l_s}"
        print(f"|{name:15}|{left:13}|{right:13}|")

    # Plot raw vs smoothed curves
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,5))
    plt.plot(curve,    color='#0000FF', label='Raw 100-ep mean', linewidth=1)
    plt.plot(sm_curve, color='#FF0000', label='TWEMA-smoothed',  linewidth=1)
    plt.xlabel("Episodes"); plt.ylabel("Average Reward")
    plt.title("Phase 3: Distributed DQN")
    plt.legend(); plt.tight_layout(); plt.show()
