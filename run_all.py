"""
Master Runner & Reproducibility Suite:
Executes the full end-to-end multi-agent evaluation pipeline:
1. Renders Retro Arcade Maze Visualization matching Figure 1 of the Project Proposal.
2. Generates animated simulation GIFs (Selfish vs DCOP comparison & DCOP gameplay).
3. Executes Experiment 1: Team Size k and Ghost Aggression beta vs Price of Anarchy (PoA).
4. Executes Experiment 2: DCOP Message Budget B recovery (DSA vs Max-Sum).
5. Executes Experiment 3: Shapley Value reward shaping parameter lambda and Gini fairness.
6. Executes Experiment 4: Comprehensive Multi-Method Benchmark across all 8 baselines & remedies.
7. Generates publication-grade figures in plots/ and media/.
"""

import os
import shutil
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from generate_animations import generate_side_by_side_gif, generate_single_policy_gif
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

    # 1. Publication-Grade Maze Visualization matching Proposal Figure 1
    print("\n[Stage 1/6] Rendering Retro Arcade Environment (Matching Proposal Figure 1)...")
    generate_maze_visualization(layout_name="proposal_maze", num_pacman=3, num_ghosts=2)

    # 2. Animated Simulation GIFs
    print("\n[Stage 2/6] Generating Animated Gameplay GIFs (Selfish vs DCOP & Shapley)...")
    generate_side_by_side_gif(num_steps=45, out_path="plots/comparison_selfish_vs_dcop.gif")
    generate_single_policy_gif(policy_type="dcop", num_steps=40, out_path="plots/pacman_dcop_simulation.gif")
    generate_single_policy_gif(policy_type="shapley", num_steps=40, out_path="plots/pacman_shapley_simulation.gif")

    # 3. Experiment 1: PoA vs k and beta
    print("\n[Stage 3/6] Running Experiment 1: Team Size k & Ghost Aggression beta vs PoA...")
    run_experiment_1(layout_name="proposal_maze", num_episodes=20)

    # 4. Experiment 2: DCOP Message Budget Recovery
    print("\n[Stage 4/6] Running Experiment 2: DCOP Message Budget Recovery (DSA vs Max-Sum)...")
    run_experiment_2(layout_name="proposal_maze", k=3, num_episodes=20)

    # 5. Experiment 3: Shapley Shaping and Fairness
    print("\n[Stage 5/6] Running Experiment 3: Shapley Parameter lambda & Gini Fairness...")
    run_experiment_3(layout_name="proposal_maze", k=3, num_episodes=20)

    # 6. Experiment 4: Comprehensive Benchmark
    print("\n[Stage 6/6] Running Experiment 4: Comprehensive Multi-Method Benchmark...")
    run_experiment_4(layout_name="proposal_maze", k=3, num_episodes=20)

    # Synchronize all generated plots and animations to media/ for LaTeX/documentation compatibility
    os.makedirs("media", exist_ok=True)
    for f in os.listdir("plots"):
        if f.endswith((".png", ".pdf", ".gif")):
            shutil.copy(os.path.join("plots", f), os.path.join("media", f))

    elapsed = time.time() - start_time
    print("\n=====================================================================")
    print(f"ALL EXPERIMENTS, ANIMATIONS, AND PLOTS GENERATED IN {elapsed:.1f}s!")
    print("Results stored in: results/ (JSON and CSV)")
    print("Plots stored in: plots/ and media/ (High-Res PNG and Animated GIF)")
    print("Interactive GUI: run 'python play_game.py'")
    print("Web Arcade Game: open 'play.html' in your browser")
    print("=====================================================================")


if __name__ == "__main__":
    main()
