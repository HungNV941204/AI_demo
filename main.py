import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from ipmsm_env import IPMSMEnv

# Create environment
env = IPMSMEnv()

# EvalCallback removed due to timeout issues

# Instantiate the agent with optimized hyperparameters
model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, n_steps=4096, batch_size=128, n_epochs=20, gamma=0.99, gae_lambda=0.95)

# Train the agent with more timesteps for optimization
model.learn(total_timesteps=100000)

# Save the agent
model.save("ppo_ipmsm_optimized")

# Load and test
model = PPO.load("ppo_ipmsm_optimized")

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Optimized Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test manually
obs, info = env.reset()
total_reward = 0
steps = 0
for i in range(1000):  # Longer test
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        print(f"Episode finished after {steps} steps with total reward {total_reward:.2f}")
        obs, info = env.reset()
        total_reward = 0
        steps = 0

env.close()