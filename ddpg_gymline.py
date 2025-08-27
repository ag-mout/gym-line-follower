import gym_line_follower
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import imageio


from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np


env = gym.make('LineFollower-v0', gui = True, render_mode = 'rgb_array')

vec_env = DummyVecEnv([lambda: env])

train_model = False
model_type = "ppo" # "ppo" or "ddpg

model_name = model_type +"_line_follower"
if train_model:
    if model_type == "ddpg":
        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
        eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)
        # The noise objects for DDPG
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=1, tensorboard_log="./ddpg_line_follower_tensorboard/")

        # model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_line_follower_tensorboard/")
        model.learn(total_timesteps=10000, callback=eval_callback)

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # # Save the agent
        # model.save("ddpg_line_follower")

    if model_type == "ppo":
        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
        eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_line_follower_tensorboard/")
        model.learn(total_timesteps=100_000, callback=eval_callback)

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # # Save the agent
        # model.save("ppo_line_follower")

    #save the model
    model.save(model_name)

else:
    if model_type == "ddpg":
       # The noise objects for DDPG
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=1, tensorboard_log="./ddpg_line_follower_tensorboard/")
        
    if model_type == "ppo":
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_line_follower_tensorboard/")


# Load the trained agent and do a test run
if model_type == "ppo":
    model = PPO.load("ppo_line_follower", env=vec_env)
if model_type == "ddpg":
    model = DDPG.load("ddpg_line_follower", env=vec_env)
frames = []
info_record= []
obs = model.env.reset()
frame = model.env.render(mode="rgb_array")

steps = 0
for i in range(1000):
    frames.append(frame)
    action, _state = model.predict(obs, deterministic=True)
    # print("Action: ", *action)
    obs, reward, done, info = vec_env.step(action)
    info_record.append(info)
    steps += 1
    if done:
        print("Done in ", steps, " steps")
        break
    frame = model.env.render(mode="rgb_array")
    if frame is None:
        print("Frame is None!!")

imageio.mimsave(model_name+'.gif', frames)