import argparse
import diambra.arena
from diambra.arena import EnvironmentSettings, SpaceTypes
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from diambra.arena.ray_rllib.make_ray_env import DiambraArena

import os

# ---------- Build Environment Settings ----------
def build_env_settings():
    settings = EnvironmentSettings()
    settings.characters = ["Armorking", "Kuma"]
    settings.action_space = SpaceTypes.DISCRETE
    settings.difficulty = 4
    settings.step_ratio = 3
    settings.outfits = 1
    return settings

# ---------- Load Policy ----------
def load_policy(algo: str):
    checkpoint_dir = "checkpoints/"  # ✅ This is the correct directory
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    print(f"\n📦 Loading checkpoint from: {checkpoint_dir}")
    algo = Algorithm.from_checkpoint(checkpoint_dir)
    return algo.get_policy()

    # Build dummy Algorithm just to get policy object
    dummy_algo = Algorithm.from_checkpoint(export_dir)
    policy = dummy_algo.get_policy()
    return policy

# ---------- Run Inference Loop ----------
def play_agent(policy: Policy, settings: EnvironmentSettings):
    env = diambra.arena.make("tektagt", settings, render_mode="human")

    obs, info = env.reset()
    print("\n🎮 Starting gameplay with exported policy...\n")

    while True:
        env.render()
        action = policy.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("\n🕹️  Episode ended. Resetting environment...")
            obs, info = env.reset()

    env.close()

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exported policy on DIAMBRA Arena (tektagt)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "rainbow"],
                        help="Algorithm used to export the policy")

    args = parser.parse_args()

    env_settings = build_env_settings()
    policy = load_policy(args.algo)
    play_agent(policy, env_settings)
