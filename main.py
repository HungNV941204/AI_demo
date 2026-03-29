import gym
import numpy as np
from stable_baselines3 import PPO
from ipmsm_env import IPMSMEnv

# Create environment
env = IPMSMEnv()

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Save the agent
model.save("ppo_ipmsm")

# Load and test
model = PPO.load("ppo_ipmsm")

obs = env.reset()
total_reward = 0
for i in range(10000):  # 1 second at 10kHz
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        print(f"Episode finished after {i} steps with total reward {total_reward}")
        obs = env.reset()
        total_reward = 0

env.close()