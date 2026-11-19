"""
Experiment 2: DCOP Message Budget B vs. Fraction of Selfish Gap Recovered.
Compares DSA and Max-Sum across message budgets B in [0, 2, 6, 12, 20, 30].
Computes Gap Recovered = (J(DCOP) - J(Selfish)) / (J(Opt) - J(Selfish)).
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import compute_gap_recovered
from src.trainer import evaluate_policy, train_q_learning


def run_experiment_2(layout_name: str = "small_grid", k: int = 3, num_episodes: int = 25):
    print("==================================================")
    print("Running Experiment 2: DCOP Message Budget B (DSA vs Max-Sum)")
    print("==================================================")

    budgets = [0, 2, 6, 12, 20, 30]

    # Baseline 1: Optimal (Centralized)
    print("Evaluating Centralized Upper Bound (pi*)...")
    res_opt = evaluate_policy(
        policy_type="centralized",
        layout_name=layout_name,
        num_pacman=k,
        num_ghosts=1,
        ghost_beta=0.5,
        num_episodes=num_episodes,
        seed=42,
    )
    j_opt = res_opt["team_return_mean"]

    # Baseline 2: Selfish Independent Q-learners
    print("Training and Evaluating Selfish Baseline...")
    selfish_agents = train_q_learning(
        layout_name=layout_name,
        num_pacman=k,
        num_ghosts=1,
        ghost_beta=0.5,
        mode="selfish",
        num_episodes=60,
        seed=101,
    )
    res_selfish = evaluate_policy(
        policy_type="q_learning",
        layout_name=layout_name,
        num_pacman=k,
        num_ghosts=1,
        ghost_beta=0.5,
        trained_agents=selfish_agents,
        num_episodes=num_episodes,
        seed=202,
    )
    j_selfish = res_selfish["team_return_mean"]

    print(f"Base Returns: J(pi*) = {j_opt:.2f}, J(pi_selfish) = {j_selfish:.2f}")

    results = {
        "budgets": budgets,
        "j_opt": j_opt,
        "j_selfish": j_selfish,
        "dsa_returns": [],
        "dsa_recovered": [],
        "max_sum_returns": [],
        "max_sum_recovered": [],
    }

    for b in budgets:
        print(f"Testing Message Budget B = {b} ...")

        # DSA
        res_dsa = evaluate_policy(
            policy_type="dcop",
            layout_name=layout_name,
            num_pacman=k,
            num_ghosts=1,
            ghost_beta=0.5,
            dcop_algo="dsa",
            dcop_budget=b,
            num_episodes=num_episodes,
            seed=500 + b,
        )
        recov_dsa = compute_gap_recovered(res_dsa["team_return_mean"], j_selfish, j_opt)
        results["dsa_returns"].append(res_dsa["team_return_mean"])
        results["dsa_recovered"].append(recov_dsa * 100.0)

        # Max-Sum
        res_ms = evaluate_policy(
            policy_type="dcop",
            layout_name=layout_name,
            num_pacman=k,
            num_ghosts=1,
            ghost_beta=0.5,
            dcop_algo="max_sum",
            dcop_budget=b,
            num_episodes=num_episodes,
            seed=600 + b,
        )
        recov_ms = compute_gap_recovered(res_ms["team_return_mean"], j_selfish, j_opt)
        results["max_sum_returns"].append(res_ms["team_return_mean"])
        results["max_sum_recovered"].append(recov_ms * 100.0)

        print(
            f"  B={b:2d} | DSA: J={res_dsa['team_return_mean']:.2f} ({recov_dsa*100:.1f}%) | "
            f"Max-Sum: J={res_ms['team_return_mean']:.2f} ({recov_ms*100:.1f}%)"
        )

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(
        budgets,
        results["dsa_recovered"],
        marker="o",
        linewidth=2.2,
        markersize=8,
        color="#1f77b4",
        label="DCOP-DSA (Distributed Stochastic Alg.)",
    )
    plt.plot(
        budgets,
        results["max_sum_recovered"],
        marker="s",
        linewidth=2.2,
        markersize=8,
        color="#d62728",
        label="DCOP-Max-Sum (Belief Propagation)",
    )

    plt.axhline(
        100.0,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=r"Upper Bound $J(\pi^\star)$ (100% Recovery)",
    )
    plt.axhline(
        0.0,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=r"Selfish Baseline $J(\pi_{\mathrm{selfish}})$ (0% Recovery)",
    )

    plt.title(
        r"Selfish Gap Recovery vs. DCOP Message Budget ($B$)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(r"Message Budget per Step ($B$)", fontsize=12)
    plt.ylabel(r"Gap Recovered (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(budgets)
    plt.ylim(-5, 115)
    plt.legend(fontsize=10.5, loc="lower right")
    plt.tight_layout()

    plot_path = "plots/fig_e2_dcop_budget.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    print("Experiment 2 Complete!\n")
    return results


if __name__ == "__main__":
    run_experiment_2()
