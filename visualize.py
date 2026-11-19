"""
Maze Visualization and Trajectory Renderer:
Generates visual snapshot figures showing:
1. Environment state: walls, pellets, ghosts, and Pac-Men.
2. DCOP Region partitioning: color-coded Voronoi regions with centroids.
3. Episode trajectory comparison: Selfish vs Coordinated paths.
Saves publication-ready figures to plots/fig_maze.png and plots/fig_trajectories.png.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.environment import MultiAgentPacmanEnv
from src.dcop import RegionPartition, DCOPSolver


def generate_maze_visualization(layout_name: str = "small_grid", num_pacman: int = 3):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("media", exist_ok=True)

    env = MultiAgentPacmanEnv(layout_name=layout_name, num_pacman=num_pacman, num_ghosts=1, seed=42)
    partition = RegionPartition(env, num_regions=num_pacman)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_facecolor("#111111")

    # Region colors with soft transparency
    region_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Draw Region background shading
    for r in range(partition.num_regions):
        for (cx, cy) in partition.region_cells[r]:
            rect = patches.Rectangle(
                (cx - 0.5, cy - 0.5), 1, 1,
                facecolor=region_colors[r % len(region_colors)],
                alpha=0.18,
                edgecolor="none"
            )
            ax.add_patch(rect)

    # 2. Draw Walls
    for (wx, wy) in env.walls:
        rect = patches.Rectangle(
            (wx - 0.5, wy - 0.5), 1, 1,
            facecolor="#1e293b",
            edgecolor="#334155",
            linewidth=1.5
        )
        ax.add_patch(rect)

    # 3. Draw Pellets
    for (px, py) in env.pellets:
        circle = patches.Circle((px, py), 0.14, facecolor="#facc15", edgecolor="none")
        ax.add_patch(circle)

    # 4. Draw Region Centers (Centroids)
    for r, (cx, cy) in enumerate(partition.centers):
        star = patches.RegularPolygon(
            (cx, cy), numVertices=5, radius=0.35,
            orientation=np.pi/2, facecolor="none",
            edgecolor="white", linestyle="--", linewidth=1.5
        )
        ax.add_patch(star)
        ax.text(cx, cy - 0.45, f"$R_{r+1}$", color="white", fontsize=9, ha="center", fontweight="bold")

    # 5. Draw Pac-Man agents
    pacman_colors = ["#38bdf8", "#fbbf24", "#a3e635", "#f472b6"]
    for i, (px, py) in enumerate(env.pacman_positions):
        circle = patches.Circle((px, py), 0.38, facecolor=pacman_colors[i % len(pacman_colors)], edgecolor="black", linewidth=1.2)
        ax.add_patch(circle)
        ax.text(px, py, f"$P_{i+1}$", color="black", fontsize=9.5, ha="center", va="center", fontweight="bold")

    # 6. Draw Ghosts
    for j, (gx, gy) in enumerate(env.ghost_positions):
        circle = patches.Circle((gx, gy), 0.40, facecolor="#ef4444", edgecolor="black", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(gx, gy, f"$G_{j+1}$", color="white", fontsize=9.5, ha="center", va="center", fontweight="bold")

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Top-down grid representation
    ax.set_xticks([])
    ax.set_yticks([])

    # Custom legend
    legend_elements = [
        patches.Patch(facecolor="#1e293b", edgecolor="#334155", label="Maze Wall"),
        patches.Circle((0, 0), 0.1, facecolor="#facc15", label="Shared Pellet"),
        patches.Circle((0, 0), 0.2, facecolor="#38bdf8", label="Pac-Man Agent ($P_i$)"),
        patches.Circle((0, 0), 0.2, facecolor="#ef4444", label="Ghost ($G_j$)"),
        patches.Patch(facecolor="#1f77b4", alpha=0.3, label="DCOP Region $R_r$"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=5, fontsize=9.5, frameon=True, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    plt.title("Multi-Agent Pac-Man Simulation Environment with DCOP Regions", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_plot = "plots/fig_maze_simulation.png"
    out_media = "media/fig_maze.png"
    plt.savefig(out_plot, bbox_inches="tight")
    plt.savefig(out_media, bbox_inches="tight")
    plt.close()
    print(f"Visual maze saved to {out_plot} and {out_media}")


if __name__ == "__main__":
    generate_maze_visualization()
