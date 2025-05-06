# main.py
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

# --- Reuse your TicTacToeWrapper ---
class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    symbols = ['O', ' ', 'X']
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(54)
        self.state = None
    def seed(self, seed=None):
        random.seed(seed); np.random.seed(seed)
    def reset(self, seed=None, **kwargs):
        if seed is not None: self.seed(seed)
        self.state = {'board': [0]*9, 'on_move': 1}
        return self.state
    def step(self, action: Tuple[int,int]):
        p, sq = action
        board = self.state['board']; om = self.state['on_move']
        # illegal
        if board[sq] != 0 or p != om:
            return self.state, -1.0, True, False, {'illegal_move': True}
        board[sq] = p; self.state['on_move'] = -p
        # win
        if self.check_win(p):
            return self.state, 1.0, True, False, {'win': True}
        # draw
        if all(cell != 0 for cell in board):
            return self.state, 0.0, True, False, {'draw': True}
        # ongoing
        return self.state, 0.0, False, False, {}
    def check_win(self, p):
        b = self.state['board']
        lines = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]
        return any(b[i]==p and b[j]==p and b[k]==p for i,j,k in lines)
    def render(self, mode='human', close=False):
        if close: return
        print("On move:", "X" if self.state['on_move']==1 else "O")
        for i in range(9):
            symbol = self.symbols[self.state['board'][i] + 1]
            print(symbol, end=' ')
            if i%3==2: print()

class TicTacToeWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = TicTacToeEnv()
        self.observation_space = gym.spaces.Box(-1,1,shape=(19,),dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)
    def seed(self, seed=None): self.env.seed(seed)
    def reset(self, seed=None, **kw):
        state = self.env.reset(seed=seed, **kw)
        return self._to_obs(state), {}
    def step(self, action:int):
        p = self.env.state['on_move']
        s, r, done, _, info = self.env.step((p,action))
        return self._to_obs(s), float(r), done, False, info
    def _to_obs(self, state):
        b = np.array(state['board'],dtype=np.float32)
        om = np.array([state['on_move']],dtype=np.float32)
        mask = np.array([1.0 if c==0 else 0.0 for c in state['board']],dtype=np.float32)
        return np.concatenate([b, om, mask])

# --- Q‑Learning Agent ---
class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
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
        k = self.state_key(obs)
        kn = self.state_key(next_obs)
        q_sa = self.q[(k,action)]
        max_q_next = 0.0 if done else max(self.q[(kn,a)] for a in range(self.n_actions))
        target = reward + self.gamma * max_q_next
        self.q[(k,action)] += self.alpha * (target - q_sa)

# --- Training Loop with Opponent Choice ---
def train(env, agent, episodes, window, opponent_type):
    episode_rewards = []
    running_avg = []
    rewards_deque = deque(maxlen=window)

    for ep in trange(episodes, desc="QLearning"):
        obs,_ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            current_player = env.env.state['on_move']
            if current_player == 1 or opponent_type == 'self':
                action = agent.choose_action(obs)
            else:
                # random opponent
                valid = [i for i,m in enumerate(env.env.state['board']) if m==0]
                action = random.choice(valid)

            next_obs, reward, done, _, info = env.step(action)
            # accumulate reward only for player 1
            if current_player == 1:
                total_reward += reward

            # update Q‑table
            if opponent_type == 'self':
                agent.update(obs, action, reward, next_obs, done)
            else:  # random opponent
                if current_player == 1 or 'illegal_move' in info:
                    agent.update(obs, action, reward, next_obs, done)

            obs = next_obs

        episode_rewards.append(total_reward)
        rewards_deque.append(total_reward)
        running_avg.append(np.mean(rewards_deque))
        agent.epsilon = max(agent.epsilon * 0.9999, 0.01)

    return episode_rewards, running_avg

# --- Plotting Function ---
def plot_learning_curve(running_avg, window, opponent_name):
    """
    Plots only the running average (in blue) with legend outside.
    """
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'axes.edgecolor': 'black'
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams['font.family'] = 'Times New Roman'

    color = "#0000FF"
    ax.plot(running_avg, color=color, linewidth=1)

    patch = mpatches.Patch(color=color, label=f"{opponent_name}")
    ax.legend(
        handles=[patch],
        frameon=True,
        fancybox=True,
        prop={'family':'Times New Roman','weight':'bold','size':12},
        loc='upper left',
        bbox_to_anchor=(1.02, 1)
    )

    ax.set_xlabel("Episode", fontsize=16, family='Times New Roman')
    ax.set_ylabel("Average Reward", fontsize=16, family='Times New Roman')
    ax.set_title("Average Reward Over Time", fontsize=18, family='Times New Roman')

    max_eps = len(running_avg)
    step = max(1, max_eps // 10)
    xticks = list(range(0, max_eps+1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks], fontsize=12, family='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=12, family='Times New Roman')

    sns.despine()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--window',   type=int, default=100)
    parser.add_argument('--opponent', choices=['random','self-play'], default='self-play')
    args = parser.parse_args()

    env   = TicTacToeWrapper()
    env.seed(42)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1, gamma=0.99, epsilon=0.2
    )

    ep_rewards, avg_rewards = train(
        env, agent,
        episodes=args.episodes,
        window=args.window,
        opponent_type=args.opponent
    )

    plot_learning_curve(
        avg_rewards,
        window=args.window,
        opponent_name=args.opponent.capitalize()
    )

    print(f"\nFinal avg reward (last {args.window} eps): {avg_rewards[-1]:.3f}")
