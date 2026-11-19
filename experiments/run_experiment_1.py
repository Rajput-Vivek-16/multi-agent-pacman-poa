"""
Experiment 1: Effect of Team Size k and Ghost Aggression beta on Price of Anarchy (PoA).
Evaluates k in [2, 3, 4] and beta in [0.0, 0.3, 0.6, 0.9].
Computes PoA = J(pi*) / J(pi_selfish).
"""

import json
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import compute_poa
from src.trainer import evaluate_policy, train_q_learning


def run_experiment_1(layout_name: str = "small_grid", num_episodes: int = 25):
    print("==================================================")
    print("Running Experiment 1: PoA vs Team Size k and Ghost Danger beta")
    print("==================================================")

    k_values = [2, 3, 4]
    beta_values = [0.0, 0.3, 0.6, 0.9]

    results = {
        "k_values": k_values,
        "beta_values": beta_values,
        "optimal_returns": {},
        "selfish_returns": {},
        "poa_matrix": {},
    }

    for k in k_values:
        results["optimal_returns"][k] = []
        results["selfish_returns"][k] = []
        results["poa_matrix"][k] = []

        for beta in beta_values:
            print(f"Testing k={k}, beta={beta} ...")

            # Evaluate Centralized Planner (pi*)
            res_opt = evaluate_policy(
                policy_type="centralized",
                layout_name=layout_name,
                num_pacman=k,
                num_ghosts=1 if k <= 2 else 2,
                ghost_beta=beta,
                num_episodes=num_episodes,
                seed=100 + k * 10 + int(beta * 10),
            )

            # Train and evaluate Selfish Independent Q-learners (pi_selfish)
            selfish_agents = train_q_learning(
                layout_name=layout_name,
                num_pacman=k,
                num_ghosts=1 if k <= 2 else 2,
                ghost_beta=beta,
                mode="selfish",
                num_episodes=60,
                seed=200 + k * 10 + int(beta * 10),
            )

            res_selfish = evaluate_policy(
                policy_type="q_learning",
                layout_name=layout_name,
                num_pacman=k,
                num_ghosts=1 if k <= 2 else 2,
                ghost_beta=beta,
                trained_agents=selfish_agents,
                num_episodes=num_episodes,
                seed=300 + k * 10 + int(beta * 10),
            )

            j_opt = res_opt["team_return_mean"]
            j_selfish = res_selfish["team_return_mean"]
            poa_val = compute_poa(j_opt, j_selfish)

            results["optimal_returns"][k].append(j_opt)
            results["selfish_returns"][k].append(j_selfish)
            results["poa_matrix"][k].append(poa_val)

            print(
                f"  -> J(pi*)={j_opt:.2f}, J(pi_selfish)={j_selfish:.2f}, PoA={poa_val:.3f}"
            )

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, k in enumerate(k_values):
        plt.plot(
            beta_values,
            results["poa_matrix"][k],
            marker=markers[idx],
            linewidth=2.2,
            markersize=8,
            color=colors[idx],
            label=f"k = {k} Pac-Men",
        )

    plt.title(
        r"Price of Anarchy (PoA) vs. Ghost Aggression ($\beta$)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(r"Ghost Aggression Parameter ($\beta$)", fontsize=12)
    plt.ylabel(r"Price of Anarchy ($\mathrm{PoA}$)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(beta_values)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()

    plot_path = "plots/fig_e1_poa_vs_k_beta.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    print("Experiment 1 Complete!\n")
    return results


if __name__ == "__main__":
    run_experiment_1()
