import os
from tqdm import tqdm
import csv
import glob
import argparse
import diambra.arena
from diambra.arena import SpaceTypes, EnvironmentSettings, WrappersSettings
from diambra.arena.ray_rllib.make_ray_env import DiambraArena, preprocess_ray_config

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print

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
def build_env_settings():
    settings = EnvironmentSettings()
    settings.characters = ["Armorking", "Kuma"]

    # 🔀 Auto-select action space based on algorithm
    #if args.algo.lower() in ["dqn", "rainbow"]:
    settings.action_space = SpaceTypes.DISCRETE
    #else:
    #    settings.action_space = SpaceTypes.MULTI_DISCRETE
    settings.difficulty = 1
    settings.step_ratio = 6
    settings.outfits = 1
    settings.frame_shape = (240, 320, 1)
    return settings

#def build_wrapper_settings():
    #wrapper_settings = WrappersSettings()
    #wrapper_settings.stack_frame = 1
    #return wrapper_settings

# ---------- Agent Builders ----------
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
def main(args):
    # Ensure required directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Build environment settings
    env_settings = build_env_settings()
    #wrap_settings = build_wrapper_settings()
    env_config = {
        "game_id": "tektagt",
        "settings": env_settings,
        #"wrapper_settings": wrap_settings,
        "cli_args": "-s=4"
    }

    # Build agent
    if args.algo == "ppo":
        agent = build_ppo_agent(env_config)
        log_file = "results/ppo.csv"
    elif args.algo == "rainbow":
        agent = build_dqn_agent(env_config, use_rainbow=True)
        log_file = "results/rainbow_dqn.csv"
    elif args.algo == "dqn":
        agent = build_dqn_agent(env_config, use_rainbow=False)
        log_file = "results/dqn.csv"
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Load checkpoint if requested
    if args.load_latest:
        args.load_checkpoint = get_latest_checkpoint(args.algo)

    if args.load_checkpoint:
        print(f"\n🔄 Loading checkpoint from: {args.load_checkpoint}")
        agent.restore(args.load_checkpoint)

    # Training loop
    init_csv_log(log_file, ["iteration", "reward_mean"])
    print(f"\n🚀 Starting training using {args.algo.upper()}...\n")
    for i in tqdm(range(args.iters), desc="Training"):
        result = agent.train()
        reward = result["episode_reward_mean"]
        episodes = result.get("episodes_this_iter", 0)
        ep_len = result.get("episode_len_mean", 0)
        print(f"[{args.algo.upper()}] Iteration {i}: reward = {reward}, episodes = {episodes}, ep_len = {ep_len}")
        print(pretty_print(result))
        log_result(log_file, [i, reward])

    print("\n✅ Training completed.")

    # Save checkpoint
    if args.save:
        checkpoint_path = agent.save("checkpoints/")
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
        env = diambra.arena.make("tektagt", env_settings, render_mode="human")
        max_episodes = 2  # 🔁 Change this if you want more play episodes
        episode_count = 0

        obs, info = env.reset()
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
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "rainbow"],
                        help="RL algorithm to use: ppo | dqn | rainbow")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--play", action="store_true", help="Play with trained agent after training")
    parser.add_argument("--save", action="store_true", help="Save the trained agent as a checkpoint")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load the agent from")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-load the latest checkpoint for the selected algorithm")
    parser.add_argument("--export-policy", action="store_true",
                        help="Export the policy model for inference after training or loading")

    args = parser.parse_args()
    main(args)
