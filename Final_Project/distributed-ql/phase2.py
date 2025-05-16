#!/usr/bin/env python3
"""
phase2.py: Distributed vs Single‑Process Q‑Learning with TWEMA smoothing
Implements Phase 2 prototype:
 - RolloutActors keep their own Q-table snapshots
 - Head performs 1-step TD updates on every transition
 - After each wave, head pushes only the changed Q-values (delta) back to actors
"""

import argparse
import random
import time
import numpy as np
from collections import defaultdict, deque

import gym
import ray
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm

# --------------------------------------
# 1) Seeding helper
# --------------------------------------
def set_global_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)

# --------------------------------------
# 2) TicTacToe Env & Wrapper
# --------------------------------------
class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    symbols = ['O',' ','X']
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
        self.state = {'board':[0]*9,'on_move':1}
        return self.state

    def step(self, action):
        p, sq = action
        b, om = self.state['board'], self.state['on_move']
        if b[sq]!=0 or p!=om:
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
        if close:
            return
        print("On move:", "X" if self.state['on_move']==1 else "O")
        for i in range(9):
            print(self.symbols[self.state['board'][i]+1], end=' ')
            if i%3==2:
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
        return self._to_obs(state), {}

    def step(self, idx:int):
        p = self.env.state['on_move']
        s, r, done, _, info = self.env.step((p, idx))
        return self._to_obs(s), float(r), done, False, info

    def _to_obs(self, st):
        b    = np.array(st['board'],dtype=np.float32)
        om   = np.array([st['on_move']],dtype=np.float32)
        mask = np.array([1.0 if c==0 else 0.0 for c in st['board']],dtype=np.float32)
        return np.concatenate([b,om,mask])

# --------------------------------------
# 3) Q‑Learning Agent
# --------------------------------------
class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q         = defaultdict(float)
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.n_actions = n_actions

    def key(self, obs):
        return tuple(obs.tolist())

    def act(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        qs = [self.q[(self.key(obs),a)] for a in range(self.n_actions)]
        return int(np.argmax(qs))

    def update(self, obs, act, rew, nxt, done):
        k, kn = self.key(obs), self.key(nxt)
        q_sa  = self.q[(k,act)]
        q_n   = 0.0 if done else max(self.q[(kn,a)] for a in range(self.n_actions))
        target= rew + self.gamma * q_n
        self.q[(k,act)] += self.alpha * (target - q_sa)

# --------------------------------------
# 4) Single‑Process Baseline
# --------------------------------------
def single_train(total_eps, opponent):
    env   = TicTacToeWrapper()
    set_global_seed(42, env)
    agent = QLearningAgent(env.action_space.n)
    dq, curve = deque(maxlen=100), []
    for _ in trange(total_eps, desc="Single"):
        obs,_=env.reset(); done=False; total=0.0
        while not done:
            p = env.env.state['on_move']
            if p==1 or opponent=='self':
                a = agent.act(obs)
            else:
                a = random.choice([i for i,m in enumerate(env.env.state['board']) if m==0])
            nxt, r, done, _, _ = env.step(a)
            if p==1:
                total += r
            agent.update(obs,a,r,nxt,done)
            obs = nxt
        dq.append(total)
        curve.append(np.mean(dq))
    return curve

# --------------------------------------
# 5) Rollout Actor with Delta‑Pull
# --------------------------------------
@ray.remote
class RolloutActor:
    def __init__(self, q_snap, seed, opponent):
        # seed RNGs so each actor is deterministic up to delta pushes
        random.seed(seed); np.random.seed(seed)
        self.env      = TicTacToeWrapper()
        self.env.seed(seed)
        self.agent    = QLearningAgent(self.env.action_space.n)
        self.agent.q  = defaultdict(float, q_snap)
        self.opponent=opponent

    def run(self, episodes):
        traj, rews = [], []
        for _ in range(episodes):
            obs,_=self.env.reset(); done=False; total=0.0
            while not done:
                p = self.env.env.state['on_move']
                if p==1 or self.opponent=='self':
                    a = self.agent.act(obs)
                else:
                    choices = [i for i,m in enumerate(self.env.env.state['board']) if m==0]
                    a = random.choice(choices)
                nxt, r, done, _, _ = self.env.step(a)
                traj.append((obs,a,r,nxt,done))
                self.agent.update(obs,a,r,nxt,done)
                if p==1:
                    total += r
                obs = nxt
            rews.append(total)
        return traj, rews

    def pull_delta(self, delta):
        # merge only the changed Q-values from head
        for key, value in delta.items():
            self.agent.q[key] = value

# --------------------------------------
# 6) Distributed Training with Delta‑Push
# --------------------------------------
def distributed_train(total_eps, wave_eps, num_actors, opponent):
    head  = QLearningAgent(TicTacToeWrapper().action_space.n)
    ray.init(ignore_reinit_error=True)

    # create actors once with initial (empty) Q-snapshot
    initial_snap   = dict(head.q)
    actors         = [
        RolloutActor.remote(initial_snap, 100+i, opponent)
        for i in range(num_actors)
    ]

    dq, curve, proc = deque(maxlen=100), [], 0
    pbar             = tqdm(total=total_eps, desc="Distributed")
    last_push_snap   = dict(head.q)

    while proc < total_eps:
        # launch one wave of parallel rollouts
        futures = [actor.run.remote(wave_eps) for actor in actors]
        results = ray.get(futures)

        # merge every actor's transitions back into head
        for traj, rews in results:
            for obs, a, r, nxt, done in traj:
                head.update(obs, a, r, nxt, done)
            for r in rews:
                if proc >= total_eps:
                    break
                proc += 1
                dq.append(r)
                curve.append(np.mean(dq))
                pbar.update(1)
            if proc >= total_eps:
                break

        # compute and push only the changed Q‐entries
        delta = {
            k: v for k, v in head.q.items()
            if last_push_snap.get(k) != v
        }
        if delta:
            for actor in actors:
                actor.pull_delta.remote(delta)
            last_push_snap = dict(head.q)

    pbar.close()
    ray.shutdown()

    # trim any overshoot so curve length == total_eps
    if len(curve) > total_eps:
        curve = curve[:total_eps]

    return curve

# --------------------------------------
# 7) TWEMA smoothing
# --------------------------------------
def twema(curve, smoothing_param):
    """Time‑Weighted Exponential Moving Average (0–1) with debias."""
    n = len(curve)
    if n == 0:
        return []
    w = min(np.sqrt(smoothing_param), 0.999)
    rng = (n - 1) or 1
    last_y, debias = 0.0, 0.0
    out = []
    for i, y in enumerate(curve):
        prev = i - 1 if i > 0 else 0
        dx = (i - prev) / rng
        wa = w ** dx
        last_y = last_y * wa + y
        debias = debias * wa + 1
        out.append(last_y / debias)
    return out

# --------------------------------------
# 8) Main & Plot
# --------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',  type=int,   default=100000)
    parser.add_argument('--wave',      type=int,   default=500,
                        help="number of episodes per actor per wave")
    parser.add_argument('--actors',    type=int,   default=4)
    parser.add_argument('--opponent',  choices=['random','self'], default='self')
    parser.add_argument('--ma-window', type=float, default=1.0,
                        help="TWEMA smoothing param (0–1)")
    args = parser.parse_args()

    # ---- single-process ----
    t0  = time.time()
    sp  = single_train(args.episodes, args.opponent)
    t_sp= time.time() - t0
    eps_sp = args.episodes / t_sp

    # ---- distributed ----
    t1    = time.time()
    dist  = distributed_train(
        args.episodes, args.wave, args.actors, args.opponent)
    t_dist= time.time() - t1
    eps_dist= args.episodes / t_dist
    speedup = t_sp / t_dist

    # ---- results ----
    print("\n=== Results ===")
    print("Final Average Reward:")
    print(f"    Single:       {sp[-1]:.3f}")
    print(f"    Distributed:  {dist[-1]:.3f}\n")
    print("Performance:")
    print(f"    Single:       time={t_sp:.2f}s, eps/sec={eps_sp:.1f}")
    print(f"    Distributed:  time={t_dist:.2f}s, eps/sec={eps_dist:.1f}, "
          f"speed‑up={speedup:.2f}×\n")

    # ---- apply TWEMA smoothing ----
    sp_s   = twema(sp,   args.ma_window)
    dist_s = twema(dist, args.ma_window)

    # ---- plotting ----
    sns.set_style("whitegrid", {'axes.edgecolor':'black'})
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(sp_s,   color='#0000FF', label='Single',      linewidth=1)
    ax.plot(dist_s, color='#FF0000', label='Distributed', linewidth=1)

    ax.set_xlabel("Episodes",       fontsize=16)
    ax.set_ylabel("Average Reward", fontsize=16)
    ax.set_title(f"Distributed vs Single — opponent: {args.opponent}", fontsize=18)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25,1), frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.show()
