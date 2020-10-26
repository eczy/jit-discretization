import argparse
from policy import value_iteration
import gym
from gym.core import Wrapper

from discretizer import Discretizer
from mdp import MDP

def main():
    env = gym.make('MountainCar-v0')
    env_unwrapped = env.unwrapped
    discretizer = Discretizer(20, 0, 1)
    S, A, P, R = discretizer(env_unwrapped, zero_reward_if_done=True)
    gamma = 0.9
    H = 200
    mdp = MDP(S, A, P, R, H, gamma)
    V, pi = value_iteration(mdp)
    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            d_obs = discretizer.discretize_state(observation, 1)
            action = pi[list(d_obs.keys())[0]]
            print(V[list(d_obs.keys())[0]], list(d_obs.keys())[0], action)
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

if __name__ == "__main__":
    main()