from collections import defaultdict
import numpy as np

def value_iteration(mdp, theta=1e-3, lookahead=1):
    V = np.zeros(mdp.S)
    while True:
        delta = 0.
        for s in np.arange(mdp.S):
            v = V[s]
            action_values = np.zeros(mdp.A)
            for a in np.arange(mdp.A):
                for s_prime, p in mdp.P[s][a].items():
                    action_values[a] += p * (mdp.R[s, a] + mdp.gamma * V[s_prime])
            V[s] = action_values.max()
            delta = np.max([delta, np.abs(v - V[s])])
        print(delta)
        if delta < theta:
            break
    pi = {}
    for s in np.arange(mdp.S):
        action_values = np.zeros(mdp.A)
        for a in np.arange(mdp.A):
            for s_prime, p in mdp.P[s][a].items():
                action_values[a] += p * (mdp.R[s, a] + mdp.gamma * V[s_prime])
        pi[s] = np.argmax(action_values)
    import pdb
    pdb.set_trace()
    return V, pi

def policy_evaluation():
    pass

def policy_improvement():
    pass