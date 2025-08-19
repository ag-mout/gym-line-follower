#test gym_line_follower package
import gym_line_follower
import gymnasium as gym
from gymnasium.wrappers.rendering import RenderCollection
import matplotlib.pyplot as plt
# from stable_baselines3.common.env_checker import check_env
import imageio


env = gym.make('LineFollower-v0', gui = True, render_mode = 'rgb_array')
env = RenderCollection(env)
# check_env(env, warn=True, skip_render_check=False)

env.reset(seed=123)

frames = []
for i in range(100):
    action = env.action_space.sample()
    obsv, reward, done, truncated, info = env.step(action)
    print("Reward: ", reward)
    
    if done:
        break

frames = env.render()
print(obsv)
print("Done: ", done)

imageio.mimsave('test.gif', frames)

#close the environment
env.close()