import pdb
import numpy as np
from gym.spaces import Discrete, Box
from gym import Env
from mdp import MDP
from collections import defaultdict
from itertools import product
from copy import deepcopy
from scipy.spatial import KDTree

class Discretizer:

    def __init__(self, state_resolution, action_resolution, knn):
        self.state_resolution = state_resolution
        self.action_resolution = action_resolution
        self.knn = knn

        self.S_kdt = None
        self.X_to_S = None

    def __call__(self, env: Env, zero_reward_if_done=False):
        if isinstance(env.action_space, Box):
            pass
        else:
            A = env.action_space.n
        if isinstance(env.observation_space, Box):
            S = self.state_resolution ** len(env.observation_space.low)
            low, high = env.observation_space.low, env.observation_space.high
            S_gridlengths = np.subtract(high, low) / self.state_resolution

            S_coords = []
            for d, gl in enumerate(S_gridlengths):
                S_coords.append(np.linspace(low[d] + gl, high[d] - gl, self.state_resolution))
            
            X_to_S = {k: v for v, k in enumerate(product(*S_coords))}
            
            S_kdt = KDTree(np.stack([x for x in X_to_S.keys()]))
            self.S_kdt = S_kdt
            self.X_to_S = X_to_S
        else:
            pass

        P = defaultdict(lambda: defaultdict(dict))
        R = defaultdict(lambda: 0)
        env_ = deepcopy(env)
        for (x, s), a in product((X_to_S.items()), np.arange(A)):
            env_.reset()
            env_.state = x
            observation, reward, done, _ = env_.step(a)
            if zero_reward_if_done and done:
                R[(s, a)] = 0
            else:
                R[(s, a)] = reward
            distances, indices = S_kdt.query(observation, k=self.knn)
            next_states = {}
            if self.knn > 1:
                for d, i in zip(distances / distances.sum(), indices):
                    next_states[i] = d
            else:
                next_states[indices] = 1

            P[s][a] = next_states
        return S, A, P, R

    def discretize_state(self, x, knn):
        distances, indices = self.S_kdt.query(x, k=knn)
        s_prime = {}
        if knn > 1:
            for d, i in zip(distances / distances.sum(), indices):
                s_prime[i] = d
        else:
            s_prime[indices] = 1
        return s_prime




