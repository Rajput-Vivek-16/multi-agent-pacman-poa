"""
Master Runner Script:
Executes all simulations, generates visual renders, and produces all
tables and publication-quality plots.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from visualize import generate_maze_visualization
from experiments.run_experiment_1 import run_experiment_1
from experiments.run_experiment_2 import run_experiment_2
from experiments.run_experiment_3 import run_experiment_3
from experiments.run_experiment_4 import run_experiment_4


def main():
    start_time = time.time()
    print("=====================================================================")
    print("STARTING FULL REPRODUCIBILITY SUITE: MULTI-AGENT PAC-MAN PoA TESTBED")
    print("=====================================================================")

    # 1. Visualization
    print("\n[Stage 1/5] Rendering Environment and Region Partition Visuals...")
    generate_maze_visualization(layout_name="small_grid", num_pacman=3)

    # 2. Experiment 1: PoA vs k and beta
    print("\n[Stage 2/5] Running Experiment 1: Team Size & Ghost Aggression vs PoA...")
    run_experiment_1(layout_name="small_grid", num_episodes=20)

    # 3. Experiment 2: DCOP Message Budget
    print("\n[Stage 3/5] Running Experiment 2: DCOP Message Budget Recovery...")
    run_experiment_2(layout_name="small_grid", k=3, num_episodes=20)

    # 4. Experiment 3: Shapley Shaping and Fairness
    print("\n[Stage 4/5] Running Experiment 3: Shapley Parameter lambda & Gini Fairness...")
    run_experiment_3(layout_name="small_grid", k=3, num_episodes=20)

    # 5. Experiment 4: Comprehensive Benchmark
    print("\n[Stage 5/5] Running Experiment 4: Comprehensive Multi-Method Benchmark...")
    run_experiment_4(layout_name="small_grid", k=3, num_episodes=25)

    # Copy all figures from plots/ to media/ for LaTeX compatibility
    import shutil
    os.makedirs("media", exist_ok=True)
    for f in os.listdir("plots"):
        if f.endswith(".png") or f.endswith(".pdf"):
            shutil.copy(os.path.join("plots", f), os.path.join("media", f))

    elapsed = time.time() - start_time
    print("\n=====================================================================")
    print(f"ALL EXPERIMENTS AND FIGURES SUCCESSFULLY GENERATED IN {elapsed:.1f}s!")
    print("Results stored in: results/")
    print("Plots stored in: plots/ and media/")
    print("=====================================================================")


if __name__ == "__main__":
    main()
