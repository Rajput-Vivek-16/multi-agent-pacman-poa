"""
Experiment 4: Comprehensive Benchmark across all Methods & Baselines:
1. Random
2. Independent Greedy
3. Selfish Q-Learning (Independent Q)
4. Equal-Share Q-Learning
5. DCOP-DSA (B=12)
6. DCOP-Max-Sum (B=24)
7. Shapley-Shaped Q-Learning (lambda=0.5)
8. Centralized Team Planner (pi*)
Outputs:
- CSV Table
- JSON dump
- Multi-metric comparative bar chart
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


def run_experiment_4(layout_name: str = "small_grid", k: int = 3, num_episodes: int = 30):
    print("==================================================")
    print("Running Experiment 4: Comprehensive Multi-Method Comparison")
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
        "random", layout_name=layout_name, num_pacman=k, num_episodes=num_episodes, seed=10
    )

    # 2. Independent Greedy
    print("Evaluating 2/8: Independent Greedy...")
    res_greedy = evaluate_policy(
        "greedy", layout_name=layout_name, num_pacman=k, num_episodes=num_episodes, seed=20
    )

    # 3. Selfish Q-Learning
    print("Evaluating 3/8: Selfish Q-Learning...")
    selfish_agents = train_q_learning(
        layout_name=layout_name,
        num_pacman=k,
        mode="selfish",
        num_episodes=70,
        seed=30,
    )
    res_selfish = evaluate_policy(
        "q_learning",
        layout_name=layout_name,
        num_pacman=k,
        trained_agents=selfish_agents,
        num_episodes=num_episodes,
        seed=35,
    )

    # 4. Equal-Share Q-Learning
    print("Evaluating 4/8: Equal-Share Q-Learning...")
    eq_agents = train_q_learning(
        layout_name=layout_name,
        num_pacman=k,
        mode="equal_share",
        num_episodes=70,
        seed=40,
    )
    res_eq = evaluate_policy(
        "q_learning",
        layout_name=layout_name,
        num_pacman=k,
        trained_agents=eq_agents,
        num_episodes=num_episodes,
        seed=45,
    )

    # 5. DCOP-DSA (B=12)
    print("Evaluating 5/8: DCOP-DSA (B=12)...")
    res_dsa = evaluate_policy(
        "dcop",
        layout_name=layout_name,
        num_pacman=k,
        dcop_algo="dsa",
        dcop_budget=12,
        num_episodes=num_episodes,
        seed=50,
    )

    # 6. DCOP-Max-Sum (B=24)
    print("Evaluating 6/8: DCOP-Max-Sum (B=24)...")
    res_ms = evaluate_policy(
        "dcop",
        layout_name=layout_name,
        num_pacman=k,
        dcop_algo="max_sum",
        dcop_budget=24,
        num_episodes=num_episodes,
        seed=60,
    )

    # 7. Shapley-Shaped Q-Learning
    print("Evaluating 7/8: Shapley Q-Learning (lambda=0.5)...")
    shapley_agents = train_q_learning(
        layout_name=layout_name,
        num_pacman=k,
        mode="shapley",
        shapley_lambda=0.5,
        num_episodes=70,
        seed=70,
    )
    res_shapley = evaluate_policy(
        "q_learning",
        layout_name=layout_name,
        num_pacman=k,
        trained_agents=shapley_agents,
        num_episodes=num_episodes,
        seed=75,
    )

    # 8. Centralized Team Planner
    print("Evaluating 8/8: Centralized Team Planner (pi*)...")
    res_opt = evaluate_policy(
        "centralized",
        layout_name=layout_name,
        num_pacman=k,
        num_episodes=num_episodes,
        seed=80,
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

    j_opt = res_opt["team_return_mean"]
    j_selfish = res_selfish["team_return_mean"]

    for name, r in zip(methods, raw_results):
        j_val = r["team_return_mean"]
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

    # Print Table to console
    print("\n================ BENCHMARK SUMMARY TABLE ================")
    headers = [
        "Method",
        "Team Return",
        "PoA",
        "Gap Recov (%)",
        "Gini",
        "Captures",
        "Steps",
    ]
    print(
        f"{headers[0]:<28} | {headers[1]:<12} | {headers[2]:<6} | {headers[3]:<14} | {headers[4]:<6} | {headers[5]:<8} | {headers[6]:<6}"
    )
    print("-" * 92)
    for row in benchmark_data:
        print(
            f"{row['Method']:<28} | {row['Team Return']:>6.2f} +/- {row['Return 95% CI']:<4.1f} | "
            f"{row['PoA']:>6.3f} | {row['Gap Recovered (%)']:>13.1f}% | {row['Gini Index']:>6.3f} | "
            f"{row['Captures']:>8.2f} | {row['Steps to Clear']:>6.1f}"
        )

    # Save to JSON and CSV
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_4_benchmark.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)

    with open("results/experiment_4_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_data[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_data)

    # Generate Publication Plot
    os.makedirs("plots", exist_ok=True)
    names = [row["Method"] for row in benchmark_data]
    returns = [row["Team Return"] for row in benchmark_data]
    cis = [row["Return 95% CI"] for row in benchmark_data]

    colors = [
        "#7f7f7f",  # Random
        "#bcbd22",  # Greedy
        "#d62728",  # Selfish Q
        "#ff7f0e",  # Equal Share
        "#1f77b4",  # DCOP DSA
        "#17becf",  # DCOP Max-Sum
        "#9467bd",  # Shapley Q
        "#2ca02c",  # Centralized
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(names, returns, xerr=cis, capsize=4, color=colors, alpha=0.88, edgecolor="black")

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        offset = 4 if width >= 0 else -18
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlabel(r"Mean Team Return $J(\pi)$ (with 95% CI)", fontsize=12)
    ax.set_title("Multi-Agent Pac-Man Benchmark: Method Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()

    plot_path = "plots/fig_e4_method_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    print("Experiment 4 Complete!\n")
    return benchmark_data


if __name__ == "__main__":
    run_experiment_4()
