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

    # TODO: action space discretization
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
                S_coords.append(
                    np.linspace(low[d] + gl, high[d] - gl, self.state_resolution)
                )

            S_kdt = KDTree(np.stack(list(product(*S_coords))))
            self.S_kdt = S_kdt
        else:
            pass

        P = defaultdict(lambda: defaultdict(dict))
        R = defaultdict(lambda: 0)
        env_ = deepcopy(env)
        for (s, x), a in product(enumerate(S_kdt.data), np.arange(A)):
            env_.reset()
            env_.state = x
            observation, reward, done, _ = env_.step(a)
            if zero_reward_if_done and done:
                R[(s, a)] = 0
            else:
                R[(s, a)] = reward
            P[s][a] = self.discretize_state(observation)
        return S, A, P, R

    def discretize_state(self, x):
        distances, indices = self.S_kdt.query(x, k=self.knn)
        s_prime = {}
        if self.knn > 1:
            for d, i in zip(distances / distances.sum(), indices):
                s_prime[i] = d
        else:
            s_prime[indices] = 1
        return s_prime
