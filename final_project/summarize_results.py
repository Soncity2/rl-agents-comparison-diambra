import os
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== AESTHETICS ==========
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'axes.facecolor': '#0d0d0d',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'figure.facecolor': '#0d0d0d',
    'savefig.facecolor': '#0d0d0d',
    'text.color': 'white',
    'axes.titleweight': 'bold'
})

AGENT_COLORS = {
    "DQN": "#e0bf00",
    "PPO": "#c06f05",
    "RAINBOW": "#d62728"
}

# ========== 1. SUMMARIZATION ==========
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

# ========== 2. LOAD RAW DATA ==========
def load_agent_data(base_path, difficulties, agents):
    data = []
    for diff in difficulties:
        for agent in agents:
            path = os.path.join(base_path, f'difficulty_{diff}', f'{agent}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Agent'] = agent.upper()
                df['Difficulty'] = diff
                data.append(df)
            else:
                print(f"Missing file: {path}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# ========== 3. METRICS ==========
def calculate_metrics(df):
    grouped = df.groupby(['Difficulty', 'Agent']).agg({
        'Rounds Won': 'sum',
        'Total Rounds': 'sum',
        'Episode Win': 'sum',
        'Num Stages': 'mean',
        'Total Reward': 'mean',
        'Episode': 'count'
    }).rename(columns={
        'Rounds Won': 'Total Rounds Won',
        'Total Rounds': 'Total Rounds Played',
        'Episode Win': 'Episodes Won',
        'Num Stages': 'Avg Stage',
        'Total Reward': 'Cumulative Mean Reward',
        'Episode': 'Episodes'
    }).reset_index()

    grouped['Round Win Rate (%)'] = (grouped['Total Rounds Won'] / grouped['Total Rounds Played']) * 100
    grouped['Win Rate (%)'] = (grouped['Episodes Won'] / grouped['Episodes']) * 100

    grouped['Cumulative Mean Reward'] = grouped['Cumulative Mean Reward'].round(2)
    grouped['Avg Stage'] = grouped['Avg Stage'].round(2)

    return grouped

# ========== 4. TABLE ==========
def create_comparison_table(metrics_df):
    return metrics_df[['Difficulty', 'Agent', 'Cumulative Mean Reward', 'Avg Stage']]

# ========== 5. TED PLOTS ==========
def plot_ted_comparison(metrics_df, difficulty, save_dir=None):
    subset = metrics_df[metrics_df["Difficulty"] == difficulty]
    colors = subset["Agent"].map(AGENT_COLORS)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d0d0d')

    axs[0].bar(subset["Agent"], subset["Round Win Rate (%)"], color=colors)
    axs[0].set_title("Round Win Rate (%)")
    axs[0].set_ylabel("Rate (%)")

    axs[1].bar(subset["Agent"], subset["Avg Stage"], color=colors)
    axs[1].set_title("Average Stage Reached")
    axs[1].set_ylabel("Stage")

    axs[2].bar(subset["Agent"], subset["Cumulative Mean Reward"], color=colors)
    axs[2].set_title("Cumulative Mean Reward")
    axs[2].set_ylabel("Reward")

    for ax in axs:
        for bar in ax.patches:
            ax.annotate(f'{bar.get_height():.2f}',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, color='white')
        ax.set_xlabel("Agent")
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')

    plt.suptitle(f"Agent Performance – Difficulty {difficulty}", fontsize=18, color='white', weight='bold')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"ted_comparison_difficulty_{difficulty}.png")
        plt.savefig(plot_path, facecolor=fig.get_facecolor(), dpi=300)
        print(f"✅ Saved TED bar plot for difficulty {difficulty} to {plot_path}")
        plt.close()
    else:
        plt.show()

# ========== 6. HEATMAPS ==========
def plot_ted_heatmaps(metrics_df, save_dir=None):
    metrics = ["Cumulative Mean Reward", "Win Rate (%)", "Round Win Rate (%)", "Avg Stage"]
    for metric in metrics:
        pivot = metrics_df.pivot(index='Agent', columns='Difficulty', values=metric)
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, cbar=True)

        plt.title(f"{metric} Across Difficulties", fontsize=14, color='white', weight='bold')
        plt.xlabel("Difficulty", color='white')
        plt.ylabel("Agent", color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_safe_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(save_dir, f"heatmap_{file_safe_metric}.png")
            plt.savefig(save_path, facecolor='#0d0d0d', dpi=300)
            print(f"✅ Saved heatmap for '{metric}' to {save_path}")
            plt.close()
        else:
            plt.show()

# ========== 7. MAIN ==========
def main():
    results_dir = "../results_play"
    summary = []
    difficulties = []
    agents = set()

    for difficulty_folder in glob.glob(f"{results_dir}/difficulty_*"):
        difficulty_name = os.path.basename(difficulty_folder)
        difficulty_num = difficulty_name.split("_")[-1]
        difficulties.append(difficulty_num)
        for csv_file in glob.glob(f"{difficulty_folder}/*.csv"):
            row = summarize_csv(csv_file, difficulty_num)
            summary.append(row)
            agents.add(row["Agent"])

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

    df = load_agent_data(results_dir, difficulties, agents)
    if df.empty:
        print("No gameplay data found. Exiting.")
        return

    metrics_df = calculate_metrics(df)

    print("\n📊 Agent Comparison Table:")
    print(create_comparison_table(metrics_df).to_string(index=False))

    plot_dir = os.path.join(results_dir, "plots")
    for diff in difficulties:
        plot_ted_comparison(metrics_df, diff, save_dir=plot_dir)

    plot_ted_heatmaps(metrics_df, save_dir=plot_dir)

# ========== 8. RUN ==========
if __name__ == "__main__":
    main()
