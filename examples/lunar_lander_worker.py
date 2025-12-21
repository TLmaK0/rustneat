import gymnasium as gym
import numpy as np
import sys

# The rustneat_py module is passed as a global
rustneat_py = None

# Persistent environment (one per worker process)
_worker_env = None

def set_rustneat_module(module):
    global rustneat_py
    rustneat_py = module

def init_worker():
    """Initialize worker-specific resources (called once per worker)."""
    global _worker_env
    # Create persistent environment for this worker
    _worker_env = gym.make('LunarLander-v3', render_mode=None)

def evaluate_organism(genome_data, render=False, max_steps=1000):
    """Evaluate organism fitness using rustneat_py for neural network activation."""
    global _worker_env

    # Use persistent environment, or create temporary one for rendering
    if render:
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        env = _worker_env

    genes = genome_data['genes']
    neurons_len = genome_data['neurons_len']

    # Create organism using rustneat_py (Rust implementation)
    organism = rustneat_py.create_organism(genes, neurons_len)

    # Run episode with step limit
    reset_result = env.reset()
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Activate neural network (calls Rust implementation)
        outputs = organism.activate(observation.tolist())

        # Softmax to select action
        outputs_array = np.array(outputs[:4])
        exp_outputs = np.exp(outputs_array - np.max(outputs_array))
        softmax = exp_outputs / np.sum(exp_outputs)
        action = int(np.argmax(softmax))

        # Step environment
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    # Only close temporary rendering environments
    if render:
        env.close()

    return total_reward
