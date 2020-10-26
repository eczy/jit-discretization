from collections import namedtuple


class MDP:
    def __init__(self, S, A, P, R, H, gamma):
        self.S = S
        self.A = A
        self.P = P
        self.R = R
        self.H = H
        self.gamma = gamma
        