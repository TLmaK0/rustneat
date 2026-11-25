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

def run_single_episode(organism, env, max_steps=500):
    """Run a single episode and return (reward, behavior_descriptor)."""
    # Reset CTRNN state at the start of each episode
    organism.reset_state()

    reset_result = env.reset()
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    total_reward = 0.0
    done = False
    steps = 0
    action_counts = [0, 0, 0, 0]  # noop, left, main, right

    while not done and steps < max_steps:
        # Activate neural network (calls Rust implementation)
        # 2 outputs: [main_desire, lateral_direction]
        outputs = organism.activate(observation.tolist())

        # Lateral priority action selection:
        # - lateral < 0.33 → left (priority, main ignored)
        # - lateral > 0.66 → right (priority, main ignored)
        # - else: main > 0.5 → main
        # - else: noop
        lateral = outputs[1]
        if lateral < 0.33:
            action = 1  # left thruster
        elif lateral > 0.66:
            action = 3  # right thruster
        elif outputs[0] > 0.5:
            action = 2  # main thruster
        else:
            action = 0  # noop

        # Step environment
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        action_counts[action] += 1

    # Behavior descriptor: final state [x, y, vx, vy] + action distribution [noop%, left%, main%, right%]
    total_actions = max(sum(action_counts), 1)
    action_dist = [c / total_actions for c in action_counts]
    behavior = [float(observation[0]), float(observation[1]),
                float(observation[2]), float(observation[3])] + action_dist

    return total_reward, behavior


def evaluate_organism(genome_data, render=False, max_steps=400, num_episodes=1):
    """Evaluate organism fitness using rustneat_py for neural network activation.

    Returns (fitness, behavior_descriptor) where behavior is the average final state.
    """
    global _worker_env

    # Build the organism with the Rust implementation. genome_data carries the
    # population organism's CTRNN settings (tau, dt, step_time,
    # output_activation) as well as its genome, so the worker evaluates the
    # same dynamics that was evolved. Every key is required.
    organism = rustneat_py.create_organism(
        genome_data['genes'],
        genome_data['neurons_len'],
        genome_data['biases'],
        genome_data['taus'],
        genome_data['input_neurons'],
        genome_data['output_neurons'],
        genome_data['tau'],
        genome_data['dt'],
        genome_data['step_time'],
        genome_data['output_activation'],
    )

    if render:
        # For rendering, just run one episode with visualization
        env = gym.make('LunarLander-v3', render_mode='human')
        reward, behavior = run_single_episode(organism, env, max_steps)
        env.close()
        return reward + 500, behavior
    else:
        # Run multiple episodes and average for more stable fitness
        total_reward = 0.0
        behaviors = []
        for _ in range(num_episodes):
            reward, behavior = run_single_episode(organism, _worker_env, max_steps)
            total_reward += reward
            behaviors.append(behavior)
        total_reward /= num_episodes
        avg_behavior = np.mean(behaviors, axis=0).tolist()

        # Lunar Lander returns [-500, 300]; adding 500 gives [0, 800].
        # rustneat handles negative fitness, but this example keeps the
        # offset: the novelty blend, the thresholds and the saved configs
        # are all on the [0, 800] scale.
        return total_reward + 500, avg_behavior
