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
        self.q = defaultdict(float)    # key = (state_tuple, action)
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
        # greedy
        qs = [self.q[(key,a)] for a in range(self.n_actions)]
        return int(np.argmax(qs))

    def update(self, obs, action, reward, next_obs, done):
        k = self.state_key(obs)
        kn = self.state_key(next_obs)
        q_sa = self.q[(k,action)]
        max_q_next = 0.0 if done else max(self.q[(kn,a)] for a in range(self.n_actions))
        target = reward + self.gamma * max_q_next
        self.q[(k,action)] += self.alpha * (target - q_sa)

# --- Training Loop with Reward Logging ---
def train(env, agent, episodes=50000, window=100):
    episode_rewards = []
    running_avg = []
    rewards_deque = deque(maxlen=window)

    for ep in trange(episodes, desc="QLearning"):
        obs,_ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if env.env.state['on_move'] == 1:
                action = agent.choose_action(obs)
            else:
                valid = [i for i,m in enumerate(env.env.state['board']) if m==0]
                action = random.choice(valid)

            next_obs, reward, done, _, info = env.step(action)
            total_reward += reward

            if env.env.state['on_move'] == -1 or 'illegal_move' in info:
                agent.update(obs, action, reward, next_obs, done)

            obs = next_obs

        episode_rewards.append(total_reward)
        rewards_deque.append(total_reward)
        running_avg.append(np.mean(rewards_deque))
        agent.epsilon = max(agent.epsilon * 0.9999, 0.01)

    return episode_rewards, running_avg



def plot_learning_curve(episode_rewards, running_avg, window):
    """
    Plots the running average of episode rewards over time
    against Episodes, with the legend placed outside.
    """
    # 1) Seaborn whitegrid
    sns.set_style("whitegrid", {
        'axes.grid':       True,
        'axes.edgecolor': 'black'
    })

    # 2) Figure & axis
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams['font.family'] = 'Times New Roman'

    # 3) Plot curve in pure blue (#0000FF) with linewidth=1
    color = "#0000FF"
    ax.plot(running_avg, color=color, linewidth=1)

    # 4) Legend outside top-right
    patch = mpatches.Patch(color=color, label="Random Opponent")
    ax.legend(
        handles=[patch],
        frameon=True,
        fancybox=True,
        prop={'family':'Times New Roman','weight':'bold','size':12},
        loc='upper left',
        bbox_to_anchor=(1.02, 1)
    )

    # 5) Labels & title
    ax.set_xlabel("Episode", fontsize=16, family='Times New Roman')
    ax.set_ylabel("Average Reward", fontsize=16, family='Times New Roman')
    ax.set_title("Average Reward", fontsize=18, family='Times New Roman')

    # 6) Customize x‑ticks every 10%
    max_eps = len(running_avg)
    step = max(1, max_eps // 10)
    xticks = list(range(0, max_eps+1, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks], fontsize=12, family='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=12, family='Times New Roman')

    # 7) Final polish
    sns.despine()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    env = TicTacToeWrapper()
    env.seed(42)
    agent = QLearningAgent(n_actions=env.action_space.n,
                           alpha=0.1, gamma=0.99, epsilon=0.2)

    # Phase 1a training
    ep_rewards, avg_rewards = train(env, agent, episodes=20000, window=100)
    plot_learning_curve(ep_rewards, avg_rewards, window=100)
    print(f"\nFinal avg reward (last 100 eps): {avg_rewards[-1]:.3f}")