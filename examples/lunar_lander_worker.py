import gymnasium as gym
import sys

# The rustneat_py module is passed as a global
rustneat_py = None

def set_rustneat_module(module):
    global rustneat_py
    rustneat_py = module

def evaluate_organism(genome_data, render=False, max_steps=1000):
    """Evaluate organism fitness using rustneat_py for neural network activation."""
    env = gym.make('LunarLander-v3', render_mode='human' if render else None)

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
        import numpy as np
        outputs_array = np.array(outputs[:4])
        exp_outputs = np.exp(outputs_array - np.max(outputs_array))
        softmax = exp_outputs / np.sum(exp_outputs)
        action = int(np.argmax(softmax))

        # Step environment
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    env.close()
    return total_reward
