# RL Agent Comparisons Using DIAMBRA Arena

This repository provides a collection of reinforcement-learning agents built on top of the [DIAMBRA Arena](https://diambra.ai/) framework. The codebase includes a reproducible final project and a set of experimental scripts for prototyping and research. A shared Python virtual environment keeps setup simple across the different components.

---

## Table of Contents

1. [Project Structure](#-project-structure)
2. [Setup](#setup-python-version-39)
3. [Running Experiments](#-run-experiments)
4. [Final Project](#-final-project)
5. [Command Line Options](#-command-line-options)
6. [Results](#-results)
7. [Notes](#-notes)
8. [License](#-license)

---

## 🗂 Project Structure

```
rl-agents-comparison-diambra/
│
├── final_project/               # Final project training and evaluation
│   ├── train_agent.py
│   ├── play_agent.py
│   ├── characters.py
│   └── summarize_results.py
│
├── basic_examples/              # Experimental code and sandbox testing
│   ├── main.py
│   └── ray-rllib-train.py
│
├── results/                     # Training logs and metrics
├── results_play/                # Evaluation summaries
├── exported_policy/             # Trained models (excluded from Git)
├── checkpoints/                 # Model checkpoints (excluded from Git)
├── Roms/                        # Game ROMs (excluded from Git)
│
├── venv/                        # Shared virtual environment (excluded from Git)
├── requirements.txt             # Shared dependencies
├── .gitignore
├── README.md
└── LICENSE
```

---

## Setup (Python version 3.9)

```bash
# Clone the repository
git clone https://github.com/Soncity2/rl-agents-comparison-diambra.git
cd rl-agents-comparison-diambra

# Create and activate the Python environment
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### ROM File

Download `tektagt.zip` from DIAMBRA and place it in the `Roms/` directory. See the [official instructions](https://docs.diambra.ai/envs/games/tektagt/).

### Docker

DIAMBRA relies on Docker for its backend. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) before running any training commands.

---

## 🧪 Run Experiments

These scripts showcase basic integration and RLlib training examples.

```bash
diambra run -r <Roms path> python basic_examples/main.py            # Environment test
diambra run -r <Roms path> python basic_examples/ray-rllib-train.py  # Train via RLlib
```

---

## 🎓 Final Project

Use the following commands to train and evaluate the final agent.

```bash
diambra run -r <Roms path> python final_project/train_agent.py --algo ppo --iters 10 --save --export-policy

diambra run -r <Roms path> python final_project/play_agent.py
```

---

### 🧾 Command Line Options (Train)

The main training script exposes several useful flags:

| Argument            | Type   | Default | Description                                                |
|---------------------|--------|---------|------------------------------------------------------------|
| `--algo`            | `str`  | `ppo`   | RL algorithm to use (`ppo`, `dqn`, or `rainbow`).          |
| `--iters`           | `int`  | `10`    | Number of training iterations.                             |
| `--difficulty`      | `int`  | `1`     | Environment difficulty level (1-9).                        |
| `--play`            | flag   | `False` | Run the agent in play mode after training.                 |
| `--save`            | flag   | `False` | Save the trained agent as a checkpoint.                    |
| `--load-checkpoint` | `str`  | `None`  | Path to a checkpoint to load the agent from.               |
| `--load-latest`     | flag   | `False` | Automatically load the most recent checkpoint.             |
| `--export-policy`   | flag   | `False` | Export the trained policy for inference.                   |

----

### 🧾 Command Line Options (Play)
The main play script exposes several useful flags:

| Argument            | Type   | Default | Description                                       |
|---------------------|--------|---------|---------------------------------------------------|
| `--algo`            | `str`  | `ppo`   | RL algorithm to use (`ppo`, `dqn`, or `rainbow`). |
| `--max-episodes`    | `int`  | `3`     | Number of Episodes to play.                       |
| `--difficulty`      | `int`  | `1`     | Environment difficulty level (1-9).               |
| `--save`            | flag   | `False` | Save the trained agent as a checkpoint.           |
| `--load-checkpoint` | `str`  | `None`  | Path to a checkpoint to load the agent from.      |
| `--load-latest`     | flag   | `False` | Automatically load the most recent checkpoint.    |


---

## 📊 Results

- Training logs and performance metrics are written to the `results/` folder. Evaluation statistics generated with `play_agent.py` are stored under `results_play/`.
- Summarize Playing logs and performance metrics are written to `summary_metrics.csv` using `summarize_results.py`
- Plots and Heatmaps are save in the `plots` directory in `results_play/`

---

## 📁 Notes

- Game ROMs are not included for licensing reasons. Add your own under `Roms/`.
- The `venv/`, `checkpoints/`, and other large files are excluded via `.gitignore`.

---

## 📜 License

This project is released under the [MIT License](LICENSE).
