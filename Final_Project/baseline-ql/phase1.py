# phase1_revised.py
import argparse
import random
import numpy as np
from collections import defaultdict, deque
from tqdm import trange
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# --------------------------------------
# 1) Seeding helper for reproducibility
# --------------------------------------
def set_global_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)

# --------------------------------------
# 2) Environment & Wrapper
# --------------------------------------
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

    def reset(self, seed=None, **kw):
        if seed is not None: self.seed(seed)
        self.state = {'board': [0]*9, 'on_move': 1}
        return self.state

    def step(self, action: Tuple[int,int]):
        p, sq = action
        b, om = self.state['board'], self.state['on_move']
        if b[sq] != 0 or p != om:
            return self.state, -1.0, True, False, {'illegal_move': True}
        b[sq], self.state['on_move'] = p, -p
        if self.check_win(p):
            return self.state, 1.0, True, False, {'win': True}
        if all(cell != 0 for cell in b):
            return self.state, 0.0, True, False, {'draw': True}
        return self.state, 0.0, False, False, {}

    def check_win(self, p):
        b = self.state['board']
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        return any(b[i]==p and b[j]==p and b[k]==p for i,j,k in lines)

    def render(self, mode='human', close=False):
        if close: return
        print("On move:", "X" if self.state['on_move']==1 else "O")
        for i in range(9):
            sym = self.symbols[self.state['board'][i] + 1]
            print(sym, end=' ')
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

    def reset(self, seed=None, **kw):
        state = self.env.reset(seed=seed, **kw)
        return self._to_obs(state), {}

    def step(self, action: int):
        p = self.env.state['on_move']
        s, r, done, _, info = self.env.step((p, action))
        return self._to_obs(s), float(r), done, False, info

    def _to_obs(self, state):
        b    = np.array(state['board'], dtype=np.float32)
        om   = np.array([state['on_move']], dtype=np.float32)
        mask = np.array([1.0 if c==0 else 0.0 for c in state['board']], dtype=np.float32)
        return np.concatenate([b, om, mask])

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

    def state_key(self, obs):
        return tuple(obs.tolist())

    def choose_action(self, obs):
        key = self.state_key(obs)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        qs = [self.q[(key,a)] for a in range(self.n_actions)]
        return int(np.argmax(qs))

    def update(self, obs, action, reward, next_obs, done):
        k        = self.state_key(obs)
        kn       = self.state_key(next_obs)
        q_sa     = self.q[(k, action)]
        nxt_max  = 0.0 if done else max(self.q[(kn,a)] for a in range(self.n_actions))
        target   = reward + self.gamma * nxt_max
        self.q[(k,action)] += self.alpha * (target - q_sa)

# --------------------------------------
# 4) Training Loop
# --------------------------------------
def train(env, agent, episodes=50000, window=100, opponent_type='random'):
    episode_rewards, running_avg = [], []
    dq = deque(maxlen=window)

    for ep in trange(episodes, desc="QLearning"):
        obs,_ = env.reset()
        done  = False
        total = 0.0
        while not done:
            p = env.env.state['on_move']
            if p == 1 or opponent_type == 'self':
                action = agent.choose_action(obs)
            else:
                valid = [i for i,m in enumerate(env.env.state['board']) if m==0]
                action = random.choice(valid)

            nxt, reward, done, _, info = env.step(action)
            if p == 1:
                total += reward
            agent.update(obs, action, reward, nxt, done)
            obs = nxt

        episode_rewards.append(total)
        dq.append(total)
        running_avg.append(np.mean(dq))
        agent.epsilon = max(agent.epsilon * 0.9999, 0.01)

    return episode_rewards, running_avg

# --------------------------------------
# 5) Evaluation
# --------------------------------------
def evaluate_win_rate(agent, env, opponent_type='random', episodes=1000):
    wins=draws=losses=0
    for _ in range(episodes):
        obs,_   = env.reset()
        done    = False
        while not done:
            p = env.env.state['on_move']
            if p == 1 or opponent_type == 'self':
                action = int(np.argmax([agent.q[(agent.state_key(obs),a)] for a in range(agent.n_actions)]))
            else:
                valid = [i for i,m in enumerate(env.env.state['board']) if m==0]
                action = random.choice(valid)
            nxt,_,done,_,info = env.step(action)
            obs = nxt
        if 'win' in info:   wins+=1
        elif 'draw' in info: draws+=1
        else:                losses+=1
    print(f"Eval vs {opponent_type}: W/D/L = {wins}/{draws}/{losses}")

# --------------------------------------
# 6) Plot comparison
# --------------------------------------
def plot_comparison(runs, window, labels, colors):
    sns.set_style("whitegrid", {'axes.grid':True,'axes.edgecolor':'black'})
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(10,5))

    handles=[]
    for avg, lab, col in zip(runs, labels, colors):
        ax.plot(avg, color=col, linewidth=1)
        handles.append(mpatches.Patch(color=col, label=lab))

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02,1),
              frameon=True, fancybox=True,
              prop={'family':'Times New Roman','weight':'bold','size':12})
    ax.set_xlabel("Episode", fontsize=16, family='Times New Roman')
    ax.set_ylabel("Average Reward", fontsize=16, family='Times New Roman')
    ax.set_title("Average Reward: Random vs Self-Play", fontsize=18, family='Times New Roman')

    max_ep = len(runs[0]); step = max(1, max_ep//10)
    xticks = list(range(0, max_ep+1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks], fontsize=12, family='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=12, family='Times New Roman')

    sns.despine(); plt.tight_layout(); plt.show()

# --------------------------------------
# 7) Main
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--window',   type=int, default=100)
    args = parser.parse_args()

    env = TicTacToeWrapper()
    set_global_seed(42, env)

    # -- random baseline
    agent_r = QLearningAgent(env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.2)
    _, avg_r = train(env, agent_r, episodes=args.episodes, window=args.window, opponent_type='random')
    evaluate_win_rate(agent_r, env, opponent_type='random')

    # -- self-play baseline
    set_global_seed(42, env)  # reset seed
    agent_s = QLearningAgent(env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.2)
    _, avg_s = train(env, agent_s, episodes=args.episodes, window=args.window, opponent_type='self')
    evaluate_win_rate(agent_s, env, opponent_type='self')

    # -- comparison plot
    plot_comparison(
        runs=[avg_r, avg_s],
        window=args.window,
        labels=["Random Opponent", "Self-Play"],
        colors=["#0000FF","#FF0000"]
    )
