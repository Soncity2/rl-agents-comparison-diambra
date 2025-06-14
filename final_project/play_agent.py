import argparse
import os
import glob
import numpy as np

import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQNConfig

import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings, SpaceTypes
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config


def get_latest_checkpoint(algorithm_name):
    pattern = f"checkpoints/{algorithm_name.upper()}*/checkpoint-*"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found for {algorithm_name.upper()} in checkpoints/")
    return sorted(candidates)[-1]


def build_env_settings(difficulty=1):
    settings = EnvironmentSettings()
    settings.characters = ["Armorking", "Kuma"]
    settings.action_space = SpaceTypes.DISCRETE
    settings.difficulty = difficulty
    settings.step_ratio = 6
    settings.outfits = 1
    settings.frame_shape = (240, 320, 1)
    return settings


def build_wrapper_settings():
    wrapper_settings = WrappersSettings()
    wrapper_settings.normalize_reward = True
    wrapper_settings.normalization_factor = 0.5
    wrapper_settings.stack_frame = 4
    wrapper_settings.dilation = 2
    return wrapper_settings


def build_ppo_agent(env_config):
    config = {
        "env": DiambraArena,
        "env_config": env_config,
        "num_workers": 0,
        "train_batch_size": 1000,
    }
    config = preprocess_ray_config(config)
    return PPO(config=config)


def build_dqn_agent(env_config, use_rainbow=False):
    base_config = {
        "env": DiambraArena,
        "env_config": env_config,
        "num_workers": 0,
        "train_batch_size": 32,
        "_disable_preprocessor_api": True,
    }
    base_config = preprocess_ray_config(base_config)
    algo_config = DQNConfig().update_from_dict(base_config)

    if use_rainbow:
        algo_config = algo_config.training(
            n_step=1,
            dueling=True,
            double_q=True,
            noisy=True,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            gamma=0.99,
            lr=1e-4
        )
    else:
        algo_config = algo_config.training(
            dueling=False,
            double_q=False,
            noisy=False,
            n_step=1,
            lr=1e-4
        )

    return algo_config.build()


def play_agent(args, env_settings, wrap_settings, env_config):
    # Load the correct agent
    if args.algo == "ppo":
        agent = build_ppo_agent(env_config)
    elif args.algo == "rainbow":
        agent = build_dqn_agent(env_config, use_rainbow=True)
    elif args.algo == "dqn":
        agent = build_dqn_agent(env_config, use_rainbow=False)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Determine checkpoint path
    if args.load_latest:
        args.load_checkpoint = get_latest_checkpoint(args.algo)

    if not args.load_checkpoint:
        raise ValueError("Checkpoint must be provided via --load-checkpoint or --load-latest.")

    print(f"\n🔄 Loading checkpoint from: {args.load_checkpoint}")
    agent.restore(args.load_checkpoint)

    print("\n🎮 Running trained agent...\n")
    env = diambra.arena.make("tektagt", env_settings, wrap_settings, render_mode="human")

    wins = 0
    stage_progress = []

    for episode in range(args.max_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            env.render()
            action = agent.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            print("Reward: {}".format(reward))
            done = terminated or truncated

        if info.get("winner") == 1:
            wins += 1
        stage_progress.append(info.get("stage", 0))
        print(f"\n🎮 Episode {episode + 1} result: {'WIN' if info.get('winner') == 1 else 'LOSS'}")

    env.close()
    win_rate = wins / args.max_episodes
    avg_stage = np.mean(stage_progress)
    print(f"\n🏆 Win Rate: {win_rate:.2%}")
    print(f"📊 Avg Stage Progress: {avg_stage:.2f}")


# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained RL agents with DIAMBRA Arena")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "dqn", "rainbow"],
                        help="Algorithm to load: ppo | dqn | rainbow")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load the agent from")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-load the latest checkpoint for the selected algorithm")
    parser.add_argument("--max-episodes", type=int, default=3, help="Episodes to play")
    args = parser.parse_args()

    env_settings = build_env_settings()
    wrap_settings = build_wrapper_settings()
    env_config = {
        "game_id": "tektagt",
        "settings": env_settings,
        "wrapper_settings": wrap_settings,
        "cli_args": "-s=4"
    }

    play_agent(args, env_settings, wrap_settings, env_config)
