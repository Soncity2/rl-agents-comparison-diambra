import os
import csv
import glob
import numpy as np

def summarize_csv(file_path, difficulty):
    total_episodes = 0
    total_wins = 0
    total_rounds_won = 0
    total_rounds = 0
    total_reward = 0
    all_stages = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_episodes += 1
            total_reward += float(row["Total Reward"])
            total_rounds_won += int(row["Rounds Won"])
            total_rounds += int(row["Total Rounds"])
            all_stages.append(int(row["Num Stages"]))
            total_wins += int(row["Episode Win"])

    win_rate = total_wins / total_episodes if total_episodes > 0 else 0
    round_win_rate = total_rounds_won / total_rounds if total_rounds > 0 else 0
    avg_stage = np.mean(all_stages) if all_stages else 0

    algo_name = os.path.splitext(os.path.basename(file_path))[0].upper()

    return {
        "Difficulty": difficulty,
        "Agent": algo_name,
        "Episodes": total_episodes,
        "Cumulative Mean Reward": round(total_reward, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Round Win Rate (%)": round(round_win_rate * 100, 2),
        "Avg Stage": round(avg_stage, 2)
    }

def main():
    results_dir = "../results_play"
    summary = []

    for difficulty_folder in glob.glob(f"{results_dir}/difficulty_*"):
        difficulty_name = os.path.basename(difficulty_folder)
        difficulty_num = difficulty_name.split("_")[-1]  # ✅ Extracts just the number
        for csv_file in glob.glob(f"{difficulty_folder}/*.csv"):
            summary.append(summarize_csv(csv_file, difficulty_num))

    # ✅ Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "summary_metrics.csv")
    fieldnames = [
        "Difficulty", "Agent", "Episodes",
        "Cumulative Mean Reward", "Win Rate (%)", "Round Win Rate (%)", "Avg Stage"
    ]

    with open(output_file, mode="w", newline='') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"\n✅ Summary written to: {output_file}")

if __name__ == "__main__":
    main()
