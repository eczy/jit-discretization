from matplotlib.pyplot import plot
from policy import value_iteration
import gym

from discretizer import Discretizer
from mdp import MDP

import argparse
from itertools import count

import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import shutil
import os


def mkdir(path, force=False):
    if os.path.exists(path):
        if force:
            shutil.rmtree(path)
        else:
            raise FileExistsError(path)
    os.makedirs(path)


def plot_V(V, path):
    # Note: this assumes we used a square grid for our discretization
    heatmap = V.reshape([np.sqrt(len(V)).astype(int)] * 2)
    sns.heatmap(heatmap)
    plt.savefig(path)
    plt.close()


def run(resolution, knn, lookahead, gamma, episodes, render=False):
    env = gym.make("MountainCar-v0")
    env_unwrapped = env.unwrapped
    discretizer = Discretizer(resolution, resolution, knn)
    print(f"Discretizing at resoluton {resolution}, knn {knn}.")
    S, A, P, R = discretizer(env_unwrapped, zero_reward_if_done=True)
    mdp = MDP(S, A, P, R, 200, gamma)
    print(f"Running value iteration with lookahead {lookahead}.")
    V, pi, vi_iterations = value_iteration(mdp, lookahead=lookahead)
    steps = []
    for _ in range(episodes):
        observation = env.reset()
        for t in count(1):
            if render:
                env.render()
            d_obs = discretizer.discretize_state(observation)
            action = pi[np.random.choice(list(d_obs.keys()), p=list(d_obs.values()))]
            observation, _, done, _ = env.step(action)
            if done:
                steps.append(t)
                break
    env.close()
    print(f"Average steps over {episodes} episodes: {np.mean(steps)}.")
    return vi_iterations, V, pi, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "-f",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument("--ri", type=int, default=2, help="Initial resolution.")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--theta", type=float, default=10)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min_iter", type=int, default=3)
    args = parser.parse_args()
    np.random.seed(args.random_seed)

    mkdir(args.output_dir, args.f)

    results = defaultdict(list)
    last = np.inf
    for i in count():
        resolution = args.ri * (2 ** i)
        knn = 1
        lookahead = 1
        path = os.path.join(args.output_dir, f"resolution_{resolution}")
        mkdir(path)
        n_iter, V, _, steps = run(
            resolution, knn, lookahead, args.gamma, args.episodes
        )
        plot_V(
            V,
            os.path.join(
                path, f"res_{resolution}_knn_{knn}_lookahead_{lookahead}.png"
            ),
        )
        results["resolution"].extend([resolution] * len(steps))
        results["n_iter"].extend([n_iter] * len(steps))
        results["steps"].extend(steps)
        results["knn"].extend([knn] * len(steps))
        results["lookahead"].extend([lookahead] * len(steps))
        results["gamma"].extend([args.gamma] * len(steps))

        mean = np.mean(steps)
        if np.abs(last - mean) < args.theta and i >= args.min_iter:
            break
        last = mean
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "results.csv"))

if __name__ == "__main__":
    main()
