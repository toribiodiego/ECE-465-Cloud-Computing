#!/usr/bin/env python3
"""
phase3.py – Distributed DQN with checkpointing

► same training loop & hyper-params you already had
► extra evaluation as SECOND mover
► final table columns:
      rnd-first | rnd-second | self-first | self-second
"""

import argparse, os, random, time
from collections import deque

import gym, numpy as np, ray, torch, torch.nn as nn
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm


# ---------- 1) helpers ----------
def set_global_seed(seed, env=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if env is not None: env.seed(seed)


# ---------- 2) env ----------
class TicTacToeEnv(gym.Env):
    symbols = ['O',' ','X']
    def __init__(self):
        super().__init__()
        self.action_space      = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(54)
        self.state             = None
    def seed(self, s=None):
        random.seed(s); np.random.seed(s)
    def reset(self, seed=None, **kw):
        if seed is not None: self.seed(seed)
        self.state = {'board':[0]*9, 'on_move':1}
        return self.state
    def step(self, action):
        p, sq = action
        b, om = self.state['board'], self.state['on_move']
        if b[sq]!=0 or p!=om:
            return self.state, -1.0, True, False, {'illegal_move':True}
        b[sq] = p
        self.state['on_move'] = -p
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


class TicTacToeWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = TicTacToeEnv()
        self.action_space      = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            -1,1,shape=(19,),dtype=np.float32
        )
    def seed(self, s=None): self.env.seed(s)
    def reset(self, seed=None, **kw):
        s = self.env.reset(seed=seed, **kw)
        b     = np.array(s['board'], dtype=np.float32)
        om    = np.array([s['on_move']], dtype=np.float32)
        mask  = np.array([1. if c==0 else 0. for c in s['board']], dtype=np.float32)
        return np.concatenate([b,om,mask]), {}
    def step(self, idx:int):
        p = self.env.state['on_move']
        s, r, done, _, info = self.env.step((p, idx))
        b    = np.array(s['board'], dtype=np.float32)
        om   = np.array([s['on_move']], dtype=np.float32)
        mask = np.array([1. if c==0 else 0. for c in s['board']], dtype=np.float32)
        return np.concatenate([b,om,mask]), float(r), done, False, info


# ---------- 3) network ----------
class DQN(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19,hid), nn.ReLU(), nn.Linear(hid,9)
        )
    def forward(self, x): return self.net(x)


# ---------- 4) rollout actor ----------
@ray.remote
class RolloutActor:
    def __init__(self, weights, seed, epsilon):
        random.seed(seed); np.random.seed(seed)
        self.env     = TicTacToeWrapper(); self.env.seed(seed)
        self.epsilon = epsilon
        self.policy  = DQN().to("cpu")
        self.policy.load_state_dict(weights)
    def set_weights(self, w): self.policy.load_state_dict(w)
    def run(self, eps):
        traj, rews = [], []
        for _ in range(eps):
            obs, _ = self.env.reset(); done = False; total = 0.
            while not done:
                if random.random() < self.epsilon:
                    a = random.randrange(9)
                else:
                    a = int(self.policy(torch.tensor(obs)).argmax().item())
                nxt, r, done, _, _ = self.env.step(a)
                traj.append((obs,a,r,nxt,done))
                total += r if self.env.env.state['on_move'] == -1 else 0
                obs = nxt
            rews.append(total)
        return traj, rews


# ---------- 5) distributed training ----------
def distributed_dqn(
    total_eps, wave_eps, n_act, eps_greedy, rep, batch, gamma,
    train_freq, tgt_up, broadcast, ck_dir, ck_int
):
    os.makedirs(ck_dir, exist_ok=True)
    dev         = "cuda" if torch.cuda.is_available() else "cpu"
    pol, tgt    = DQN().to(dev), DQN().to(dev)
    tgt.load_state_dict(pol.state_dict())
    opt         = torch.optim.Adam(pol.parameters(), lr=1e-3)
    loss_fn     = nn.SmoothL1Loss()
    mem         = deque(maxlen=rep)
    trans_ct = upd_ct = proc = wave = last_ck = 0
    ckpts     = []
    ray.init(ignore_reinit_error=True)

    # spawn actors
    actors = [
      RolloutActor.remote(pol.state_dict(), 100+i, eps_greedy)
      for i in range(n_act)
    ]

    dq, curve = deque(maxlen=100), []
    pbar = tqdm(total=total_eps, desc="Distributed-DQN")

    while proc < total_eps:
        if wave % broadcast == 0:
            w = pol.state_dict()
            for a in actors: a.set_weights.remote(w)

        res = ray.get([a.run.remote(wave_eps) for a in actors])
        wave += 1

        for traj, rews in res:
            # training updates
            for obs,a,r,nxt,done in traj:
                mem.append((obs,a,r,nxt,done))
                trans_ct += 1
                if trans_ct >= rep and trans_ct % train_freq == 0:
                    batch_samp = random.sample(mem, batch)
                    ob  = np.stack([b[0] for b in batch_samp])
                    nx  = np.stack([b[3] for b in batch_samp])
                    ob_t   = torch.tensor(ob, dtype=torch.float32, device=dev)
                    nx_t   = torch.tensor(nx, dtype=torch.float32, device=dev)
                    act    = torch.tensor([b[1] for b in batch_samp], dtype=torch.int64, device=dev)
                    rew    = torch.tensor([b[2] for b in batch_samp], dtype=torch.float32, device=dev)
                    done_t = torch.tensor([b[4] for b in batch_samp], dtype=torch.float32, device=dev)

                    q      = pol(ob_t).gather(1, act.unsqueeze(-1)).squeeze(-1)
                    nxt_q  = tgt(nx_t).max(1)[0].detach()
                    target = rew + gamma * nxt_q * (1 - done_t)

                    l = loss_fn(q, target)
                    opt.zero_grad(); l.backward(); opt.step()
                    upd_ct += 1
                    if upd_ct % tgt_up == 0:
                        tgt.load_state_dict(pol.state_dict())

            # record progress
            for r in rews:
                proc += 1
                dq.append(r)
                curve.append(np.mean(dq))
                pbar.update(1)
                if proc - last_ck >= ck_int:
                    path = os.path.join(ck_dir, f"ckpt_{proc}.pt")
                    torch.save(pol.state_dict(), path)
                    ckpts.append(path)
                    last_ck = proc
                if proc >= total_eps:
                    break

    pbar.close(); ray.shutdown()
    return curve, upd_ct, ckpts


# ---------- 6) TWEMA ----------
def twema(curve, s):
    if not curve: return []
    n    = len(curve)
    w    = min(np.sqrt(s), 0.999)
    last = deb = 0.
    rng  = (n-1) or 1
    out  = []
    for y in curve:
        last *= w**(1/rng)
        deb  *= w**(1/rng)
        last += y
        deb  += 1
        out.append(last/deb)
    return out


# ---------- 7) evaluation – first mover ----------
def evaluate_first(sd, env, eps, opp):
    net = DQN(); net.load_state_dict(sd); net.eval()
    w=d=l=0
    for _ in range(eps):
        obs, _ = env.reset(); done=False
        while not done:
            p = env.env.state['on_move']
            if p == 1:
                a = int(net(torch.tensor(obs)).argmax())
            else:
                if opp=="random":
                    a = random.choice([i for i,x in enumerate(env.env.state['board']) if x==0])
                else:
                    a = int(net(torch.tensor(obs)).argmax())
            prev = p
            obs, _, done, _, info = env.step(a)
        if 'win' in info and prev==1: w+=1
        elif 'draw' in info:   d+=1
        else:                  l+=1
    return w,d,l


# ---------- 8) evaluation – second mover ----------
def evaluate_second(sd, env, eps, opp):
    net = DQN(); net.load_state_dict(sd); net.eval()
    w=d=l=0
    for _ in range(eps):
        obs, _ = env.reset(); done=False
        while not done:
            p = env.env.state['on_move']
            if p == -1:
                a = int(net(torch.tensor(obs)).argmax())
            else:
                if opp=="random":
                    a = random.choice([i for i,x in enumerate(env.env.state['board']) if x==0])
                else:
                    a = int(net(torch.tensor(obs)).argmax())
            prev = p
            obs, _, done, _, info = env.step(a)
        if 'win' in info and prev==-1: w+=1
        elif 'draw' in info:          d+=1
        else:                         l+=1
    return w,d,l


# ---------- 9) main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes',           type=int,   default=100_000)
    ap.add_argument('--wave',               type=int,   default=500)
    ap.add_argument('--actors',             type=int,   default=4)
    ap.add_argument('--epsilon',            type=float, default=0.1)
    ap.add_argument('--replay-size',        type=int,   default=10_000)
    ap.add_argument('--batch-size',         type=int,   default=32)
    ap.add_argument('--gamma',              type=float, default=0.99)
    ap.add_argument('--train-freq',         type=int,   default=4)
    ap.add_argument('--target-update',      type=int,   default=100)
    ap.add_argument('--broadcast-freq',     type=int,   default=1)
    ap.add_argument('--ma-window',          type=float, default=1.0)
    ap.add_argument('--checkpoint-dir',     type=str,   default='./checkpoints')
    ap.add_argument('--checkpoint-interval',type=int,   default=10_000)
    ap.add_argument('--eval-episodes',      type=int,   default=1_000)
    # new argument: where to save the reward curve
    ap.add_argument('--plot-dir',           type=str,   default=None,
                    help="Directory in which to save reward-curve PNG")
    args = ap.parse_args()

    # -------- run training --------
    set_global_seed(42)
    t0 = time.time()
    curve, updates, ckpts = distributed_dqn(
        args.episodes, args.wave, args.actors, args.epsilon,
        args.replay_size, args.batch_size, args.gamma,
        args.train_freq, args.target_update, args.broadcast_freq,
        args.checkpoint_dir, args.checkpoint_interval
    )
    elapsed = time.time() - t0
    eps_s   = args.episodes / elapsed
    sm      = twema(curve, args.ma_window)

    # -------- print performance --------
    print("\n=== Performance ===")
    print(f"Time:         {elapsed:.2f}s")
    print(f"Episodes/sec: {eps_s:.1f}")
    print(f"Grad updates: {updates}\n")

    # -------- checkpoint evaluation --------
    env = TicTacToeWrapper()
    print("=== Checkpoint W/D/L (first & second mover) ===")
    print("| checkpoint      | rnd-1st  | rnd-2nd  | self-1st | self-2nd |")
    print("|-----------------|----------|----------|----------|----------|")
    for ck in ckpts:
        sd   = torch.load(ck, map_location='cpu')
        name = os.path.basename(ck)
        w1,d1,l1 = evaluate_first (sd, env, args.eval_episodes, 'random')
        w2,d2,l2 = evaluate_second(sd, env, args.eval_episodes, 'random')
        w3,d3,l3 = evaluate_first (sd, env, args.eval_episodes, 'self')
        w4,d4,l4 = evaluate_second(sd, env, args.eval_episodes, 'self')
        def fmt(a,b,c): return f"{a}/{b}/{c}"
        print(f"| {name:<15} | {fmt(w1,d1,l1):<8} | {fmt(w2,d2,l2):<8} | {fmt(w3,d3,l3):<8} | {fmt(w4,d4,l4):<8} |")

    # -------- plotting (save or show) --------
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,5))
    plt.plot(curve, label='Raw 100-ep mean',   linewidth=1)
    plt.plot(sm,    label='TWEMA-smoothed',     linewidth=1)
    plt.xlabel("Episodes"); plt.ylabel("Average Reward")
    plt.title("Phase 3: Distributed DQN")
    plt.legend(); sns.despine(); plt.tight_layout()

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        fp = os.path.join(args.plot_dir, f"reward_curve_k{args.actors}.png")
        plt.savefig(fp)
        print(f"\nSaved reward plot to {fp}")
    else:
        plt.show()
