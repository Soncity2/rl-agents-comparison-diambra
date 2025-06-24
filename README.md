# RL Agent Comparisons using Diambra Arena

A reinforcement learning project using the DIAMBRA Arena to train and evaluate agents in fighting games. The repository is structured into two main sections: the **final project**, and **experimental scripts** used during research and prototyping. A single shared virtual environment is used for both.

---

## 🗂 Project Structure

```
rl-agents-comparison-diambra/
│
├── final_project/               # Final project training and evaluation
│   ├── train_agent.py
│   └── play_agent.py
│
├── experiments/                # Experimental code and sandbox testing
│   ├── main.py
│   └── ray-rllib-train.py
│
├── results/                    # Logs and metrics
├── exported_policy/            # Trained models or agents (excluded from Git)
├── checkpoints/                # Model checkpoints (excluded from Git)
├── Roms/                       # Game ROMs (excluded from Git)
│
├── venv/                       # Shared virtual environment (excluded from Git)
├── requirements.txt            # Shared dependencies
├── .gitignore
├── README.md
└── LICENSE
```

---

## Setup (Python version 3.9)

```bash
# Clone the repo
git clone https://github.com/Soncity2/rl-agents-comparison-diambra.git
cd rl-agents-comparison-diambra

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### *Add the tektagt.zip ROM file to the rom directory.
ROM file instructions: https://docs.diambra.ai/envs/games/tektagt/

### *Install Docker Desktop 
https://www.docker.com/products/docker-desktop/

---

## 🧪 Run Experiments

These are experimental scripts used to test features and train with RLlib.

```bash
diambra run -r **Roms Directory path** python basic_examples/main.py              # DIAMBRA Arena integration test
diambra run -r **Roms Directory path** python basic_examples/ray-rllib-train.py   # Training via Ray RLlib
```

---

## 🎓 Final Project

This is the main logic for training and evaluating the final agent.

```bash
diambra run -r **Roms Directory path** python final_project/train_agent.py --algo ppo --iters 10 --save --export-policy     # Train the final model
diambra run -r **Roms Directory path** python final_project/play_agent.py      # Run a trained model
```

---

### 🧾 Command Line Options

The training and evaluation scripts accept the following command-line arguments:

| Argument                 | Type     | Default     | Choices                   | Description                                                                 |
|--------------------------|----------|-------------|---------------------------|-----------------------------------------------------------------------------|
| `--algo`                 | `str`    | `"ppo"`     | `ppo`, `dqn`, `rainbow`   | Select the RL algorithm to use for training.                               |
| `--iters`                | `int`    | `10`        | _Any positive integer_    | Number of training iterations.                                             |
| `--play`                 | `flag`   | `False`     | —                         | If set, runs the agent in play mode after training.                        |
| `--save`                 | `flag`   | `False`     | —                         | If set, saves the trained agent as a checkpoint.                           |
| `--load-checkpoint`      | `str`    | `None`      | _File path_               | Loads an agent from a specified checkpoint path.                           |
| `--load-latest`          | `flag`   | `False`     | —                         | Automatically loads the latest checkpoint for the selected algorithm.      |
| `--export-policy`        | `flag`   | `False`     | —                         | Exports the trained policy model for inference use after training/loading. |

---

## 📊 Results

Training logs and performance metrics are saved as CSV files in the `results/` directory. You can visualize and compare various RL algorithms from here.

---

## 📁 Notes

- Game ROMs are not included for licensing reasons. Add your own under the `Roms/` folder.
- The `venv/`, `checkpoints/`, and other large or sensitive files are excluded via `.gitignore`.

---

## 📜 License

[MIT License](LICENSE)
