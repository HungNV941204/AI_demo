# AI Agent for IPMSM Motor Control using Reinforcement Learning

This project implements an AI agent trained with reinforcement learning to control an Interior Permanent Magnet Synchronous Motor (IPMSM).

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

Currently, it demonstrates a basic RL agent on CartPole. The code will be updated to include the IPMSM control environment and agent.

## Project Structure

- `main.py`: Main script for training and testing the RL agent
- `requirements.txt`: Python dependencies
- `.github/copilot-instructions.md`: Copilot instructions

## Future Work

- Implement IPMSM motor dynamics model
- Create a custom Gym environment for motor control
- Train RL agent for optimal control