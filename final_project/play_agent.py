import argparse
import os
import glob
import csv
import numpy as np

import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQNConfig

import diambra.arena
from diambra.arena import EnvironmentSettings, WrappersSettings, SpaceTypes
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config
from characters import names_to_id_pair


# ---------- CLI Utilities ----------
def restricted_int(val):
    ivalue = int(val)
    if ivalue < 1 or ivalue > 9:
        raise argparse.ArgumentTypeError("Value must be between 1 and 9.")
    return ivalue

# ---------- Logging Utilities ----------
def init_csv_log(file_path, header):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def log_result(file_path, row):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

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
    settings.frame_shape = (240, 320, 0)
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
        "framework": "torch",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
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
        "framework": "torch",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    }
    base_config = preprocess_ray_config(base_config)
    algo_config = DQNConfig().update_from_dict(base_config)

    if use_rainbow:
        algo_config = algo_config.training(
            n_step=1,
            dueling=True,
            double_q=True,
            noisy=False,
            num_atoms=51,
            v_min=-15.0,
            v_max=15.0,
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
        env_config["wrapper_settings"].normalize_reward= False
        env_config["wrapper_settings"].clip_reward = True
        agent = build_dqn_agent(env_config, use_rainbow=True)
    elif args.algo == "dqn":
        env_config["wrapper_settings"].normalize_reward= False
        env_config["wrapper_settings"].clip_reward = True
        agent = build_dqn_agent(env_config, use_rainbow=False)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    difficulty_level = env_settings.difficulty
    log_file = f"results_play/difficulty_{difficulty_level}/{args.algo}.csv"

    # Determine checkpoint path
    if args.load_latest:
        args.load_checkpoint = get_latest_checkpoint(args.algo)

    if not args.load_checkpoint:
        raise ValueError("Checkpoint must be provided via --load-checkpoint or --load-latest.")

    print(f"\n🔄 Loading checkpoint from: {args.load_checkpoint}")
    agent.restore(args.load_checkpoint)
    init_csv_log(log_file, ["Episode", "Num Stages", "Rounds Won", "Total Rounds", "Episode Win", "Total Reward"])
    print("\n🎮 Running trained agent...\n")
    env = diambra.arena.make("tektagt", env_settings, wrap_settings, render_mode="human")

    wins = 0
    stage_progress = []
    rounds_per_episode = []
    total_rounds_progress = []

    for episode in range(args.max_episodes):
        rounds_won = 0
        total_reward = 0
        total_rounds = 0
        obs, info = env.reset()
        done = False

        # Detect agent's character ID
        agent_char_id = names_to_id_pair("tektagt", env_settings.characters[0],env_settings.characters[1])  # Map name to ID
        current_player = (obs["P1"]["character_1"],obs["P1"]["character_2"])
        print(agent_char_id)
        print(current_player)
        # Find which player is the agent
        if current_player == agent_char_id:
            agent_key = "P1"
        else:
            agent_key = "P2"

        while not done:
            env.render()
            action = agent.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            stage = int(obs["stage"][0])
            print("Reward: {}".format(total_reward))
            print("Side {} | Total Rounds: {}".format(agent_key, total_rounds))
            done = terminated or truncated
            if info["round_done"]:
                total_rounds += 1
            if info['episode_done']:
                rounds_won = 2*(stage-1) + obs[agent_key]["wins"][0]
                print("\nEpisode Ended: Won {} round" .format(obs[agent_key]["wins"][0]))


        episode_win = 1 if stage == 8 and obs[agent_key]["wins"][0] == 1 else 0
        if episode_win == 1:
            wins += 1
        stage_progress.append(stage)
        rounds_per_episode.append(rounds_won)
        total_rounds_progress.append(total_rounds)
        print(f"Stage Ended: {stage_progress}")
        print(f"\nWon {rounds_won} rounds")
        print(f"\n🎮 Episode {episode + 1} result: {'WIN' if episode_win == 1 else 'LOSS'}")
        log_result(log_file, [episode + 1, stage, rounds_won, total_rounds, episode_win, total_reward])

    env.close()
    win_rate = wins / args.max_episodes
    rounds_win_rate = sum(rounds_per_episode) / sum(total_rounds_progress)
    avg_stage = np.mean(stage_progress)
    print(f"\n🏆 {args.algo.upper()} Win Rate: {win_rate:.2%}")
    print(f"\n🎯Round Win Rate: {rounds_win_rate:.2%}")
    print(f"\n📊{args.algo.upper()} Avg Stage Progress: {avg_stage:.2f}")


# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play trained RL agents with DIAMBRA Arena")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "dqn", "rainbow"],
                        help="Algorithm to load: ppo | dqn | rainbow")
    parser.add_argument("--difficulty", type=restricted_int, nargs="+",default=1,
                        help="List of difficulty levels (1-9) to evaluate (e.g. --difficulty 1 2 3)")

    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load the agent from")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-load the latest checkpoint for the selected algorithm")
    parser.add_argument("--max-episodes", type=int, default=3, help="Episodes to play")
    args = parser.parse_args()

    for difficulty in args.difficulty:
        print(f"\n=== Playing at Difficulty Level {difficulty} ===")
        env_settings = build_env_settings(difficulty)
        wrap_settings = build_wrapper_settings()
        env_config = {
            "game_id": "tektagt",
            "settings": env_settings,
            "wrapper_settings": wrap_settings,
            "cli_args": "-s=4"
        }
        play_agent(args, env_settings, wrap_settings, env_config)

