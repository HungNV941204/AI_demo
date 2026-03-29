import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from ipmsm_env import IPMSMEnv

# Create environment
env = IPMSMEnv()

# Instantiate the agent with PPO for continuous control
model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)

# Train the agent (increased timesteps for better accuracy)
model.learn(total_timesteps=20000)

# Save the agent
model.save("ppo_ipmsm")

# Load and test
model = PPO.load("ppo_ipmsm")

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test manually
obs, info = env.reset()
total_reward = 0
steps = 0
for i in range(500):  # Shorter test
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    if done:
        print(f"Episode finished after {steps} steps with total reward {total_reward:.2f}")
        obs, info = env.reset()
        total_reward = 0
        steps = 0

env.close()