import argparse
import diambra.arena
from diambra.arena import EnvironmentSettings, SpaceTypes
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from diambra.arena.ray_rllib.make_ray_env import DiambraArena
import torch
import os
import numpy as np

# ---------- Build Environment Settings ----------
def build_env_settings():
    settings = EnvironmentSettings()
    settings.characters = ["Armorking", "Kuma"]
    settings.action_space = SpaceTypes.DISCRETE
    settings.difficulty = 1
    settings.step_ratio = 6
    settings.outfits = 1
    settings.frame_shape = (240, 320, 1)
    return settings

# ---------- Load Policy ----------
def load_algorithm(checkpoint_dir: str) -> Algorithm:
    if not checkpoint_dir:
        raise FileNotFoundError("No checkpoint found in directory.")

    print(f"Loading RLlib Algorithm checkpoint from: {checkpoint_dir}")
    agent = Algorithm.from_checkpoint(checkpoint_dir)
    return agent

def load_exported_policy(export_dir: str) -> Policy:
    print(f"Loading exported policy from: {export_dir}")
    policy = Policy.from_checkpoint(export_dir)  # Loads Torch/TF weights
    return policy

# ---------- Run Inference Loop ----------
def play_agent(policy: Policy, settings: EnvironmentSettings):

    wins = 0
    total_episodes = 0
    stage_progress = []

    env = diambra.arena.make("tektagt", settings, render_mode="human")

    obs, info = env.reset()
    print("\n🎮 Starting gameplay with exported policy...\n")

    max_episodes = 2  # 🔁 Change this if you want more play episodes
    episode_count = 0

    while episode_count < max_episodes:
        env.render()

        # DEBUG: print structure once
        print(f"[DEBUG] obs type: {type(obs)} | keys: {obs.keys() if isinstance(obs, dict) else 'n/a'}")

        # Extract frame only, convert to torch tensor
        frame = obs["frame"]
        frame_tensor = torch.from_numpy(np.copy(frame)).float().unsqueeze(0)  # add batch dimension

        action = policy.compute_single_action({"obs": frame_tensor})

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            total_episodes += 1
            stage = info.get("stage", 0)
            stage_progress.append(stage)

            if info.get("winner") == 1:  # 1 = agent win, 2 = opponent win
                wins += 1
            print(f"Episode {total_episodes}: {'WIN' if info.get('winner') == 1 else 'LOSS'}")
            print("\n Episode ended. Resetting environment...")
            obs, info = env.reset()

    win_rate = wins / total_episodes if total_episodes > 0 else 0
    print(f"\n Win Rate: {win_rate:.2%}")

    avg_stage = np.mean(stage_progress) if stage_progress else 0
    print(f"\n Average Stage Progression: {avg_stage:.2f}")

    env.close()


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--load_type", choices=["checkpoint", "policy"], default="checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    args = parser.parse_args()

    env_settings = build_env_settings()

    if args.load_type == "checkpoint":
        agent = load_algorithm(args.checkpoint_dir)
        policy = agent.get_policy()
    elif args.load_type == "policy":
        policy = load_exported_policy(f"./exported_policy/{args.algo}")

    play_agent(policy, env_settings)

