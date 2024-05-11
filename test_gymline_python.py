#test gym_line_follower package
import gym_line_follower
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import imageio


env = gym.make('LineFollower-v0', gui = True, render_mode = 'rgb_array')
# check_env(env, warn=True, skip_render_check=False)
env.reset()
frames = []
for i in range(1000):
    action = env.action_space.sample()
    obsv, reward, done, truncated, info = env.step(action)
    # print("Reward: ", reward)
    frames.append(env.render())
    if done:
        break
env.render()
print(obsv)
print("Done: ", done)
imageio.mimsave('test.gif', frames)
#close the environment
env.close()