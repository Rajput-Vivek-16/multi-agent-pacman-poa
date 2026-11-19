"""
Ghost behavior models for Multi-Agent Pac-Man.
Controlled by aggression parameter beta in [0, 1]:
- With probability beta: greedy pursuit toward nearest Pac-Man along shortest path.
- With probability 1 - beta: random valid direction.
"""

import random
from typing import List, Tuple, Dict


class GhostPolicy:
    def __init__(self, beta: float = 0.5, rng: random.Random = None):
        self.beta = beta
        self.rng = rng if rng is not None else random.Random()

    def set_beta(self, beta: float):
        self.beta = beta

    def choose_action(
        self,
        ghost_pos: Tuple[int, int],
        pacman_positions: List[Tuple[int, int]],
        valid_moves: List[Tuple[int, int]],
        dist_fn
    ) -> Tuple[int, int]:
        """
        Choose next position for the ghost.
        valid_moves: list of candidate (x, y) cells adjacent to ghost_pos (and valid).
        dist_fn: function (u, v) -> shortest path distance in grid.
        """
        if not valid_moves:
            return ghost_pos

        if not pacman_positions:
            return self.rng.choice(valid_moves)

        # Pursue with probability beta
        if self.rng.random() < self.beta:
            # Find nearest Pac-Man
            best_dist = float('inf')
            nearest_pm = pacman_positions[0]
            for p_pos in pacman_positions:
                d = dist_fn(ghost_pos, p_pos)
                if d < best_dist:
                    best_dist = d
                    nearest_pm = p_pos

            # Pick move that minimizes distance to nearest Pac-Man
            min_d = float('inf')
            candidates = []
            for move in valid_moves:
                d = dist_fn(move, nearest_pm)
                if d < min_d:
                    min_d = d
                    candidates = [move]
                elif d == min_d:
                    candidates.append(move)

            return self.rng.choice(candidates)
        else:
            # Random valid move
            return self.rng.choice(valid_moves)
