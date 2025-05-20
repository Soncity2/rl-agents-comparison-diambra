#!/usr/bin/env python3
import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings
from tqdm import trange, tqdm

def main():
    # Settings specification
    settings = EnvironmentSettings()
    settings.step_ratio = 3
    settings.difficulty = 2

    # Gym wrappers settings
    wrappers_settings = WrappersSettings()

    # reward = reward / (C * fullHealthBarValue)
    wrappers_settings.normalize_reward = True
    wrappers_settings.normalization_factor = 0.5

    # Environment creation
    env = diambra.arena.make("tektagt", env_settings=settings, wrappers_settings=wrappers_settings, render_mode="human")

    # Parameters
    num_episodes = 10

    for episode in trange(num_episodes, desc="Episodes"):
        # Environment reset
        observation, info = env.reset(seed=42)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # (Optional) Environment rendering
            env.render()

            # Action random sampling
            actions = env.action_space.sample()

            # Environment stepping
            observation, reward, terminated, truncated, info = env.step(actions)

            # Accumulate reward and count steps
            total_reward += reward
            step_count += 1

            # Optional: print reward live (comment out if too noisy)
            print(f'\rEpisode {episode+1} | Step {step_count} | Reward: {reward:.3f}|\n Total Reward: {total_reward}', end='', flush=True)
            # Episode end check
            done = terminated or truncated

        # Show summary for the episode
        tqdm.write(f"Episode {episode+1} finished | Total Reward: {total_reward:.3f} | Steps: {step_count}")

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()
