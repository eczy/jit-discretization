import numpy as np
from itertools import count


def value_iteration(mdp, theta=1e-3, lookahead=1):
    V = np.zeros(mdp.S)
    for i in count():
        delta = 0.0
        for s in np.arange(mdp.S):
            v = V[s]
            action_values = np.zeros(mdp.A)
            for a in np.arange(mdp.A):
                action_values[a] = n_step_lookahead(mdp, V, s, a, lookahead)
            V[s] = action_values.max()
            delta = np.max([delta, np.abs(v - V[s])])
        if delta < theta:
            break
    pi = {}
    for s in np.arange(mdp.S):
        action_values = np.zeros(mdp.A)
        for a in np.arange(mdp.A):
            action_values[a] = n_step_lookahead(mdp, V, s, a, lookahead)
        pi[s] = np.argmax(action_values)
    return V, pi, i


def n_step_lookahead(mdp, V, s, a, n):
    if n == 1:
        value = 0
        for s_prime, p in mdp.P[s][a].items():
            value += p * (mdp.R[s, a] + mdp.gamma * V[s_prime])
        return value
    for s_prime, p in mdp.P[s][a].items():
        action_values = np.zeros(mdp.A)
        for action in np.arange(mdp.A):
            action_values[action] = p * (
                mdp.R[s, action]
                + mdp.gamma * n_step_lookahead(mdp, V, s_prime, action, n - 1)
            )
        return action_values.max()
