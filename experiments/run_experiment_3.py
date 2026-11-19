"""
Experiment 3: Shapley Reward Shaping Parameter lambda vs. Team Return & Gini Fairness.
Evaluates lambda in [0.0, 0.25, 0.5, 0.75, 1.0].
Also includes Equal-Share baseline for comparison.
Generates a dual-axis publication plot (Team Return vs. Gini Index).
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.trainer import evaluate_policy, train_q_learning


def run_experiment_3(layout_name: str = "small_grid", k: int = 3, num_episodes: int = 25):
    print("==================================================")
    print("Running Experiment 3: Shapley Shaping Parameter lambda & Fairness")
    print("==================================================")

    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {
        "lambdas": lambdas,
        "team_returns": [],
        "team_returns_ci": [],
        "gini_indices": [],
        "gini_indices_ci": [],
        "equal_share": {},
    }

    # 1. Evaluate Shapley-Shaped Q-learning across lambda
    for lmb in lambdas:
        print(f"Training with lambda = {lmb:.2f} ...")
        mode = "selfish" if lmb == 0.0 else "shapley"
        agents = train_q_learning(
            layout_name=layout_name,
            num_pacman=k,
            num_ghosts=1,
            ghost_beta=0.5,
            mode=mode,
            shapley_lambda=lmb,
            num_episodes=70,
            seed=700 + int(lmb * 100),
        )

        res = evaluate_policy(
            policy_type="q_learning",
            layout_name=layout_name,
            num_pacman=k,
            num_ghosts=1,
            ghost_beta=0.5,
            trained_agents=agents,
            num_episodes=num_episodes,
            seed=800 + int(lmb * 100),
        )

        results["team_returns"].append(res["team_return_mean"])
        results["team_returns_ci"].append(res["team_return_ci"])
        results["gini_indices"].append(res["gini_mean"])
        results["gini_indices_ci"].append(res["gini_ci"])

        print(
            f"  lambda={lmb:.2f} | Team Return: {res['team_return_mean']:.2f} +/- {res['team_return_ci']:.2f} | "
            f"Gini: {res['gini_mean']:.3f} +/- {res['gini_ci']:.3f}"
        )

    # 2. Train and evaluate Equal-Share Baseline
    print("Training Equal-Share Baseline (r_team / k) ...")
    eq_agents = train_q_learning(
        layout_name=layout_name,
        num_pacman=k,
        num_ghosts=1,
        ghost_beta=0.5,
        mode="equal_share",
        num_episodes=70,
        seed=999,
    )
    res_eq = evaluate_policy(
        policy_type="q_learning",
        layout_name=layout_name,
        num_pacman=k,
        num_ghosts=1,
        ghost_beta=0.5,
        trained_agents=eq_agents,
        num_episodes=num_episodes,
        seed=1111,
    )
    results["equal_share"] = {
        "team_return": res_eq["team_return_mean"],
        "team_return_ci": res_eq["team_return_ci"],
        "gini": res_eq["gini_mean"],
        "gini_ci": res_eq["gini_ci"],
    }
    print(
        f"  Equal-Share | Team Return: {res_eq['team_return_mean']:.2f} | Gini: {res_eq['gini_mean']:.3f}"
    )

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_3.json", "w") as f:
        json.dump(results, f, indent=2)

    # Dual-axis Plot
    os.makedirs("plots", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    color_ret = "#1f77b4"
    ax1.set_xlabel(r"Shapley Shaping Parameter ($\lambda$)", fontsize=12)
    ax1.set_ylabel(r"Team Return $J(\pi)$", color=color_ret, fontsize=12)
    line1 = ax1.errorbar(
        lambdas,
        results["team_returns"],
        yerr=results["team_returns_ci"],
        fmt="-o",
        color=color_ret,
        linewidth=2.2,
        capsize=4,
        markersize=7,
        label=r"Team Return $J(\pi)$",
    )
    ax1.tick_params(axis="y", labelcolor=color_ret)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Horizontal line for equal share return
    ax1.axhline(
        results["equal_share"]["team_return"],
        color=color_ret,
        linestyle=":",
        alpha=0.7,
        label="Equal-Share Return",
    )

    ax2 = ax1.twinx()
    color_gini = "#d62728"
    ax2.set_ylabel(
        "Gini Index of Individual Returns (Fairness)", color=color_gini, fontsize=12
    )
    line2 = ax2.errorbar(
        lambdas,
        results["gini_indices"],
        yerr=results["gini_indices_ci"],
        fmt="-s",
        color=color_gini,
        linewidth=2.2,
        capsize=4,
        markersize=7,
        label="Gini Coefficient",
    )
    ax2.tick_params(axis="y", labelcolor=color_gini)

    # Horizontal line for equal share gini
    ax2.axhline(
        results["equal_share"]["gini"],
        color=color_gini,
        linestyle=":",
        alpha=0.7,
        label="Equal-Share Gini",
    )

    plt.title(
        r"Effect of Shapley Reward Shaping ($\lambda$) on Team Performance & Fairness",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()

    plot_path = "plots/fig_e3_shapley_lambda.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    print("Experiment 3 Complete!\n")
    return results


if __name__ == "__main__":
    run_experiment_3()
