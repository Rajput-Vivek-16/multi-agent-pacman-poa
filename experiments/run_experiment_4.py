"""
Experiment 4: Comprehensive Multi-Method Benchmark & Comparative Suite
Evaluates all 8 baselines and remedies under identical experimental controls:
1. Random Baseline
2. Independent Greedy Baseline
3. Selfish Q-Learning (lambda=0)
4. Equal-Share Q-Learning (r_team / k)
5. DCOP-DSA (B=12)
6. DCOP-Max-Sum (B=24)
7. Shapley-Shaped Q-Learning (lambda=0.5)
8. Centralized Social Optimum (pi*)
Outputs:
- results/experiment_4_benchmark.csv
- results/experiment_4_benchmark.json
- plots/fig_e4_method_comparison.png (2x2 Multi-Panel Publication Figure)
"""

import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import compute_gap_recovered, compute_poa
from src.trainer import evaluate_policy, train_q_learning


def run_experiment_4(layout_name: str = "proposal_maze", k: int = 3, num_episodes: int = 25):
    print("==================================================")
    print("Running Experiment 4: Comprehensive Multi-Method Benchmark")
    print("==================================================")

    methods = [
        "Random",
        "Independent Greedy",
        "Selfish Q-Learning",
        "Equal-Share Q-Learning",
        "DCOP-DSA (B=12)",
        "DCOP-Max-Sum (B=24)",
        "Shapley Q-Learning (λ=0.5)",
        "Centralized Planner (π*)",
    ]

    benchmark_data = []

    # 1. Random
    print("Evaluating 1/8: Random Agents...")
    res_rand = evaluate_policy(
        "random", layout_name=layout_name, num_pacman=k, num_ghosts=2, num_episodes=num_episodes, seed=10
    )

    # 2. Independent Greedy
    print("Evaluating 2/8: Independent Greedy...")
    res_greedy = evaluate_policy(
        "greedy", layout_name=layout_name, num_pacman=k, num_ghosts=2, num_episodes=num_episodes, seed=20
    )

    # 3. Selfish Q-Learning
    print("Evaluating 3/8: Selfish Q-Learning...")
    selfish_agents = train_q_learning(
        layout_name=layout_name, num_pacman=k, num_ghosts=2, mode="selfish", num_episodes=40, seed=30
    )
    res_selfish = evaluate_policy(
        "q_learning", layout_name=layout_name, num_pacman=k, num_ghosts=2, trained_agents=selfish_agents, num_episodes=num_episodes, seed=35
    )

    # 4. Equal-Share Q-Learning
    print("Evaluating 4/8: Equal-Share Q-Learning...")
    eq_agents = train_q_learning(
        layout_name=layout_name, num_pacman=k, num_ghosts=2, mode="equal_share", num_episodes=40, seed=40
    )
    res_eq = evaluate_policy(
        "q_learning", layout_name=layout_name, num_pacman=k, num_ghosts=2, trained_agents=eq_agents, num_episodes=num_episodes, seed=45
    )

    # 5. DCOP-DSA (B=12)
    print("Evaluating 5/8: DCOP-DSA (B=12)...")
    res_dsa = evaluate_policy(
        "dcop", layout_name=layout_name, num_pacman=k, num_ghosts=2, dcop_algo="dsa", dcop_budget=12, num_episodes=num_episodes, seed=50
    )

    # 6. DCOP-Max-Sum (B=24)
    print("Evaluating 6/8: DCOP-Max-Sum (B=24)...")
    res_ms = evaluate_policy(
        "dcop", layout_name=layout_name, num_pacman=k, num_ghosts=2, dcop_algo="max_sum", dcop_budget=24, num_episodes=num_episodes, seed=60
    )

    # 7. Shapley Q-Learning (lambda=0.5)
    print("Evaluating 7/8: Shapley-Shaped Q-Learning (λ=0.5)...")
    shapley_agents = train_q_learning(
        layout_name=layout_name, num_pacman=k, num_ghosts=2, mode="shapley", shapley_lambda=0.5, num_episodes=40, seed=70
    )
    res_shapley = evaluate_policy(
        "q_learning", layout_name=layout_name, num_pacman=k, num_ghosts=2, trained_agents=shapley_agents, num_episodes=num_episodes, seed=75
    )

    # 8. Centralized Social Optimum (pi*)
    print("Evaluating 8/8: Centralized Team Planner (pi*)...")
    res_opt = evaluate_policy(
        "centralized", layout_name=layout_name, num_pacman=k, num_ghosts=2, num_episodes=num_episodes, seed=80
    )

    raw_results = [
        res_rand,
        res_greedy,
        res_selfish,
        res_eq,
        res_dsa,
        res_ms,
        res_shapley,
        res_opt,
    ]

    j_opt = float(res_opt["team_return_mean"])
    j_selfish = float(res_selfish["team_return_mean"])

    for name, r in zip(methods, raw_results):
        j_val = float(r["team_return_mean"])
        recov = compute_gap_recovered(j_val, j_selfish, j_opt) * 100.0
        poa = compute_poa(j_opt, j_val)
        benchmark_data.append(
            {
                "Method": name,
                "Team Return": round(j_val, 2),
                "Return 95% CI": round(r["team_return_ci"], 2),
                "PoA": round(poa, 3),
                "Gap Recovered (%)": round(recov, 1),
                "Gini Index": round(r["gini_mean"], 3),
                "Captures": round(r["captures_mean"], 2),
                "Steps to Clear": round(r["steps_mean"], 2),
            }
        )

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    csv_file = "results/experiment_4_benchmark.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_data[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_data)

    # Save to JSON
    json_file = "results/experiment_4_benchmark.json"
    with open(json_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # -------------------------------------------------------------
    # Publication-Grade 2x2 Multi-Panel Figure
    # -------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    os.makedirs("media", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    names = [row["Method"] for row in benchmark_data]
    short_names = [
        "Random", "Greedy", "Selfish Q", "Eq-Share",
        "DCOP-DSA", "DCOP-MS", "Shapley Q", "Opt (π*)"
    ]
    bar_colors = [
        "#94a3b8", "#64748b", "#ea580c", "#d97706",
        "#0284c7", "#2563eb", "#10b981", "#059669"
    ]

    # Panel A: Team Return J(pi)
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#f8fafc")
    returns = [row["Team Return"] for row in benchmark_data]
    ci_err = [row["Return 95% CI"] for row in benchmark_data]
    ax_a.bar(short_names, returns, yerr=ci_err, color=bar_colors, capsize=4, edgecolor="#1e293b", linewidth=1)
    ax_a.set_title("(a) Team Return $J(\pi)$ (Higher is Better)", fontsize=11.5, fontweight="bold", pad=8)
    ax_a.set_ylabel("Expected Return", fontsize=10, fontweight="bold")
    ax_a.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, axis="y")
    ax_a.tick_params(axis="x", rotation=25)

    # Panel B: Price of Anarchy (PoA)
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#f8fafc")
    poas = [row["PoA"] for row in benchmark_data]
    ax_b.bar(short_names, poas, color=bar_colors, edgecolor="#1e293b", linewidth=1)
    ax_b.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1.5, label="Ideal Optimum (PoA=1.0)")
    ax_b.set_title("(b) Price of Anarchy $\mathrm{PoA} = J(\pi^\star) / J(\pi)$", fontsize=11.5, fontweight="bold", pad=8)
    ax_b.set_ylabel("PoA (Closer to 1.0 is Better)", fontsize=10, fontweight="bold")
    ax_b.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, axis="y")
    ax_b.tick_params(axis="x", rotation=25)
    ax_b.legend(fontsize=9, loc="upper right")

    # Panel C: Gap Recovered (%)
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#f8fafc")
    gaps = [row["Gap Recovered (%)"] for row in benchmark_data]
    ax_c.bar(short_names, gaps, color=bar_colors, edgecolor="#1e293b", linewidth=1)
    ax_c.axhline(100.0, color="#16a34a", linestyle="--", linewidth=1.5, label="100% Upper Bound")
    ax_c.axhline(0.0, color="#64748b", linestyle=":", linewidth=1.5)
    ax_c.set_title("(c) Fraction of Selfish Gap Recovered (%)", fontsize=11.5, fontweight="bold", pad=8)
    ax_c.set_ylabel("Recovery Percentage", fontsize=10, fontweight="bold")
    ax_c.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, axis="y")
    ax_c.tick_params(axis="x", rotation=25)
    ax_c.legend(fontsize=9, loc="upper left")

    # Panel D: Gini Index (Fairness)
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#f8fafc")
    ginis = [row["Gini Index"] for row in benchmark_data]
    ax_d.bar(short_names, ginis, color=bar_colors, edgecolor="#1e293b", linewidth=1)
    ax_d.set_title("(d) Reward Inequality (Gini Index, Lower is Fairer)", fontsize=11.5, fontweight="bold", pad=8)
    ax_d.set_ylabel("Gini Coefficient (0 = Equality)", fontsize=10, fontweight="bold")
    ax_d.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, axis="y")
    ax_d.tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Comprehensive Multi-Agent Pac-Man Benchmark: Baselines vs. DCOP & Shapley Remedies",
        fontsize=14, fontweight="bold", y=0.99, color="#0f172a"
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    out_plot = "plots/fig_e4_method_comparison.png"
    out_media = "media/fig_e4_method_comparison.png"
    plt.savefig(out_plot, dpi=300)
    plt.savefig(out_media, dpi=300)
    plt.close()

    print(f"\nMulti-panel plot saved to {out_plot} and {out_media}")
    print(f"Benchmark CSV saved to {csv_file}")
    print("Experiment 4 Complete!\n")
    return benchmark_data


if __name__ == "__main__":
    run_experiment_4()
