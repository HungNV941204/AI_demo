import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from ipmsm_env import IPMSMEnv

# Load the trained model
try:
    model = PPO.load("ppo_ipmsm")
    print("Model loaded successfully.")
except:
    print("Model not found, training first...")
    env = IPMSMEnv()
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    model.learn(total_timesteps=5000)
    model.save("ppo_ipmsm")
    print("Model trained and saved.")

# Create environment
env = IPMSMEnv()

# Evaluate the policy (commented out due to timeout issues)
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test manually
obs, info = env.reset()
total_reward = 0
steps = 0
episode_rewards = []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        print(f"Episode {len(episode_rewards)+1}: {steps} steps, total reward {total_reward:.2f}")
        episode_rewards.append(total_reward)
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        if len(episode_rewards) >= 5:
            break

if episode_rewards:
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = (sum((r - mean_reward)**2 for r in episode_rewards) / len(episode_rewards))**0.5
    print(f"Overall: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()