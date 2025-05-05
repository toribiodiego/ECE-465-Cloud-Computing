# main.py
import random
import numpy as np
from collections import defaultdict
from tqdm import trange
from typing import Tuple

import gym
import torch
import numpy as np

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

# --- Training Loop ---
def train(env, agent, episodes=50_000):
    stats = {'wins':0, 'losses':0, 'draws':0}
    for ep in trange(episodes, desc="QLearning"):
        obs,_ = env.reset()
        done = False
        while not done:
            # agent plays as X (player=1)
            if env.env.state['on_move'] == 1:
                action = agent.choose_action(obs)
            else:
                # random opponent
                valid = [i for i,m in enumerate(env.env.state['board']) if m==0]
                action = random.choice(valid)

            next_obs, reward, done, _, info = env.step(action)
            # only update when agent moved
            if env.env.state['on_move'] == -1 or 'illegal_move' in info:
                agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

        # record outcome
        if 'win' in info:
            stats['wins'] += 1
        elif 'draw' in info:
            stats['draws'] += 1
        else:
            stats['losses'] += 1

        # optional: decay ε
        agent.epsilon = max(agent.epsilon*0.9999, 0.01)

    return stats

if __name__ == "__main__":
    env = TicTacToeWrapper()
    env.seed(42)
    agent = QLearningAgent(n_actions=env.action_space.n,
                           alpha=0.1, gamma=0.99, epsilon=0.2)

    results = train(env, agent, episodes=20_000)
    print("\nFinal results over 20k episodes:")
    print(f" Wins:   {results['wins']}")
    print(f" Draws:  {results['draws']}")
    print(f" Losses: {results['losses']}")
