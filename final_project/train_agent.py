import os
from tqdm import tqdm
import torch
import csv
import glob
import argparse
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettings, WrappersSettings
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print

import matplotlib.pyplot as plt
import numpy as np


# ---------- Device Info Utilities ----------
def print_device_info():
    print("="*50)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    print("="*50)

print_device_info()



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

# ---------- Helper: Find Latest Checkpoint ----------
def get_latest_checkpoint(algorithm_name):
    pattern = f"checkpoints/{algorithm_name.upper()}*/checkpoint-*"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found for {algorithm_name.upper()} in checkpoints/")
    return sorted(candidates)[-1]

# ---------- Environment Setup ----------
def build_env_settings(difficulty):
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

# ---------- Agent Builders ----------
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
            n_step=1,  # ← disables problematic postprocessor
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

# ---------- Main Training Logic ----------
def main(args, algo_name=None):
    if algo_name is not None:
        args.algo = algo_name
    # Ensure required directories
    os.makedirs(f"results/{args.algo}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Build environment settings
    env_settings = build_env_settings(args.difficulty)
    wrap_settings = build_wrapper_settings()
    env_config = {
        "game_id": "tektagt",
        "settings": env_settings,
        "wrapper_settings": wrap_settings,
        "cli_args": "-s=4"
    }

    # Build agent
    if args.algo == "ppo":
        agent = build_ppo_agent(env_config)
        log_file = f"results/{args.algo}/ppo.csv"
    elif args.algo == "rainbow":
        env_config["wrapper_settings"].clip_reward = True
        agent = build_dqn_agent(env_config, use_rainbow=True)
        log_file = f"results/{args.algo}/rainbow_dqn.csv"
    elif args.algo == "dqn":
        env_config["wrapper_settings"].clip_reward = True
        agent = build_dqn_agent(env_config, use_rainbow=False)
        log_file = f"results/{args.algo}/dqn.csv"
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # ✅ Check where the model is running
    try:
        model_device = next(agent.get_policy().model.parameters()).device
        print(f"\n🧠 Agent model is running on: {model_device}")
    except Exception as e:
        print(f"\n⚠️ Could not determine model device: {e}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU] Memory Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    # Load checkpoint if requested
    if args.load_latest:
        args.load_checkpoint = get_latest_checkpoint(args.algo)

    if args.load_checkpoint:
        print(f"\n🔄 Loading checkpoint from: {args.load_checkpoint}")
        agent.restore(args.load_checkpoint)

    # Training loop
    init_csv_log(log_file, ["iteration", "reward_mean", "total_episodes", "ep_len"])
    total_rewards = 0
    reward_history = []
    episode_history = []
    learning_efficiency = []
    policy_stability = []

    moving_avg_window = args.moving_avg_window  # Or any window you like
    convergence_threshold = args.stop_reward
    not_learning_threshold = args.stop_neg_reward
    try:
        print(f"\n🚀 Starting training using {args.algo.upper()}...\n")
        for i in tqdm(range(args.iters), desc="Training"):
            result = agent.train()
            episodes = result.get("episodes_this_iter", 0)
            total_episodes = result.get("episodes_total", 0)
            reward = result.get("episode_reward_mean", float("nan"))
            ep_len = result.get("episode_len_mean", 0)

            if episodes == 0 or reward is None or np.isnan(reward):
                print(f"[{args.algo.upper()}] Iteration {i} skipped: Incomplete episode or reward is NaN.")
                continue

            print(f"[{args.algo.upper()}] Iteration {i}: reward = {reward}, total episodes = {total_episodes}, ep_len = {ep_len}")
            print(pretty_print(result))
            log_result(log_file, [i, reward, total_episodes, ep_len])

            total_rewards += reward

            # Track per-iteration reward for convergence check
            reward_history.append(reward)
            episode_history.append(total_episodes)
            print(f"Cumulative reward so far: {total_rewards:.2f}")

            # Calculate learning efficiency (reward per episode)
            episodes_total = total_episodes if total_episodes > 0 else 1
            learning_efficiency.append(reward / episodes_total)

            # Calculate rolling policy stability (std dev of last 5 rewards)
            if len(reward_history) >= 2:
                window = min(5, len(reward_history))
                std_reward = np.std(reward_history[-window:])
                policy_stability.append(std_reward)
            else:
                policy_stability.append(0.0)

            if i % 10 == 0 and args.save:
                checkpoint = agent.save(f"checkpoints/{args.algo}/")
                print(f"Checkpoint saved at {checkpoint}")

            # Moving average of per-iteration reward
            if len(reward_history) >= moving_avg_window:
                moving_avg = sum(reward_history[-moving_avg_window:]) / moving_avg_window
                print(f"Moving average reward (last {moving_avg_window} iters): {moving_avg:.2f}")

                if moving_avg >= convergence_threshold:
                    print(f"\n🎉 Converged! Moving average reward reached {moving_avg:.2f} at iteration {i}.")
                    break
                if moving_avg <= not_learning_threshold:
                    print(f"\n🎉 The agent is not learning! Moving average reward reached {moving_avg:.2f} at iteration {i}.")
                    break
    except KeyboardInterrupt:
        if args.save:
            checkpoint = agent.save(f"checkpoints/{args.algo}/")
            print(f"Training interrupted, checkpoint saved at {checkpoint}")

    cumulative_reward = np.cumsum(reward_history)

    plt.figure(figsize=(10, 4))
    plt.plot(episode_history, cumulative_reward, label='Cumulative Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total Cumulative Reward')
    plt.title('Total Cumulative Reward Over Episodes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{args.algo}/total_cumulative_reward.png")
    plt.close()

    # Learning Efficiency Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(learning_efficiency)), learning_efficiency, label='Learning Efficiency')
    plt.xlabel('Iteration')
    plt.ylabel('Reward per Episode')
    plt.title(f'Learning Efficiency Over Iterations {args.algo.upper()}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{args.algo}/learning_efficiency.png")
    plt.close()

    # Policy Stability Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(policy_stability)), policy_stability, label='Policy Stability (Reward Std Dev)', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Rolling Std Dev')
    plt.title(f'Policy Stability {args.algo.upper()} - Rolling Reward Variance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{args.algo}/policy_stability.png")
    plt.close()

    # Save checkpoint
    if args.save:
        checkpoint_path = agent.save(f"checkpoints/{args.algo}/")
        print(f"\n💾 Agent checkpoint saved at: {checkpoint_path}")

    # Export policy
    if args.export_policy:
        export_dir = f"exported_policy/{args.algo}"
        os.makedirs(export_dir, exist_ok=True)
        policy = agent.get_policy()
        policy.export_model(export_dir)
        print(f"\n📦 Policy exported to: {export_dir}")

    # Play
    if args.play:
        print("\n🎮 Running trained agent...\n")
        env = diambra.arena.make("tektagt", env_settings, wrap_settings, render_mode="human")
        max_episodes = 2  # 🔁 Change this if you want more play episodes
        episode_count = 0

        obs, info = env.reset(seed=42)
        while episode_count < max_episodes:
            env.render()
            action = agent.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episode_count += 1
                print(f"\n🏁 Episode {episode_count} completed.")
                if episode_count < max_episodes:
                    obs, info = env.reset()

        env.close()
        print("\nPlay session completed.")

# ---------- CLI Interface ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents on DIAMBRA Arena (tektagt)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["all", "ppo", "dqn", "rainbow"],
                        help="RL algorithm to use: ppo | dqn | rainbow")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--difficulty", type=restricted_int, default=1, help="Environment difficulty (1-9)")
    parser.add_argument("--stop-reward", type=float, default=1000.0,
                        help="Stop if moving average reward exceeds this value")
    parser.add_argument("--stop-neg-reward", type=float, default=-100.0,
                        help="Stop if moving average reward falls to this value")
    parser.add_argument("--moving-avg-window", type=int, default=20,
                        help="Window size for moving average convergence check")
    parser.add_argument("--play", action="store_true", help="Play with trained agent after training")
    parser.add_argument("--save", action="store_true", help="Save the trained agent as a checkpoint")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load the agent from")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-load the latest checkpoint for the selected algorithm")
    parser.add_argument("--export-policy", action="store_true",
                        help="Export the policy model for inference after training or loading")

    args = parser.parse_args()

    if args.algo == "all":
        for a in ["ppo", "dqn", "rainbow"]:
            main(args, algo_name=a)
    else:
        main(args)
