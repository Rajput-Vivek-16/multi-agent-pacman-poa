"""
Baseline Agents and Centralized Team Planner:
1. RandomAgent: picks uniformly among valid actions.
2. IndependentGreedyAgent: selfish greedy to nearest pellet with local ghost evasion.
3. CentralizedTeamPlanner: social optimum coordinator using Hungarian matching for non-overlapping pellet dispatch.
"""

import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.environment import (
    ACTION_DELTAS,
    ACTION_STAY,
    ACTIONS,
    MultiAgentPacmanEnv,
)


class RandomAgent:
    def __init__(self, agent_id: int, rng: Optional[random.Random] = None):
        self.agent_id = agent_id
        self.rng = rng if rng is not None else random.Random()

    def get_action(self, env: MultiAgentPacmanEnv) -> int:
        pos = env.pacman_positions[self.agent_id]
        valid = env.get_valid_actions(pos)
        return self.rng.choice(valid)


class IndependentGreedyAgent:
    """
    Selfish greedy agent: moves toward its nearest pellet while avoiding ghosts.
    Ignores other Pac-Men, causing contention when pellets are shared.
    """
    def __init__(
        self,
        agent_id: int,
        danger_dist: int = 2,
        ghost_repulsion_weight: float = 20.0,
        rng: Optional[random.Random] = None,
    ):
        self.agent_id = agent_id
        self.danger_dist = danger_dist
        self.ghost_repulsion_weight = ghost_repulsion_weight
        self.rng = rng if rng is not None else random.Random()

    def get_action(self, env: MultiAgentPacmanEnv) -> int:
        my_pos = env.pacman_positions[self.agent_id]
        valid_actions = env.get_valid_actions(my_pos)
        if not valid_actions:
            return ACTION_STAY

        pellets = list(env.pellets)
        ghosts = env.ghost_positions

        best_score = -float("inf")
        best_actions = []

        for action in valid_actions:
            dx, dy = ACTION_DELTAS[action]
            next_pos = (my_pos[0] + dx, my_pos[1] + dy)
            if next_pos not in env.open_cells and action != ACTION_STAY:
                continue

            score = 0.0

            # 1. Ghost evasion term
            min_ghost_dist = float("inf")
            for g_pos in ghosts:
                d = env.get_distance(next_pos, g_pos)
                if d < min_ghost_dist:
                    min_ghost_dist = d

            if min_ghost_dist <= 1:
                score -= 1000.0  # Immediate danger
            elif min_ghost_dist <= self.danger_dist:
                score -= self.ghost_repulsion_weight / (min_ghost_dist + 0.1)

            # 2. Pellet attraction term (greedy to nearest pellet)
            if pellets:
                min_pellet_dist = min(env.get_distance(next_pos, p) for p in pellets)
                # Maximize negative distance -> minimize distance
                score -= min_pellet_dist * 2.0
                if next_pos in env.pellets:
                    score += 50.0  # Direct immediate reward
            else:
                score += 0.0

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) < 1e-5:
                best_actions.append(action)

        return self.rng.choice(best_actions) if best_actions else ACTION_STAY


class CentralizedTeamPlanner:
    """
    Centralized Team Planner (Social Optimum Approximation pi*):
    Jointly assigns each Pac-Man to a distinct pellet using the Hungarian algorithm
    (minimum-cost bipartite matching), eliminating redundant competition,
    while coordinating evasive maneuvers around ghosts.
    """
    def __init__(self, danger_dist: int = 2, rng: Optional[random.Random] = None):
        self.danger_dist = danger_dist
        self.rng = rng if rng is not None else random.Random()

    def get_joint_action(self, env: MultiAgentPacmanEnv) -> List[int]:
        k = env.num_pacman
        pellets = list(env.pellets)
        pacman_positions = env.pacman_positions
        ghosts = env.ghost_positions

        # 1. Optimal bipartite matching between Pac-Men and Pellets
        assigned_targets: Dict[int, Optional[Tuple[int, int]]] = {i: None for i in range(k)}

        if pellets:
            if len(pellets) >= k:
                # Build cost matrix [k x len(pellets)]
                cost_matrix = np.zeros((k, len(pellets)))
                for i in range(k):
                    for j, p in enumerate(pellets):
                        cost_matrix[i, j] = env.get_distance(pacman_positions[i], p)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    assigned_targets[r] = pellets[c]
            else:
                # Fewer pellets than agents: assign each available pellet to closest agent
                assigned_pellets = set()
                # Sort (agent, pellet) by distance
                pairs = []
                for i in range(k):
                    for p in pellets:
                        pairs.append((env.get_distance(pacman_positions[i], p), i, p))
                pairs.sort(key=lambda x: x[0])

                assigned_agents = set()
                for d, i, p in pairs:
                    if i not in assigned_agents and p not in assigned_pellets:
                        assigned_targets[i] = p
                        assigned_agents.add(i)
                        assigned_pellets.add(p)

        # 2. If any agent is unassigned (e.g. fewer pellets than agents),
        # assign them to the closest remaining pellet to assist clearing
        if pellets:
            for i in range(k):
                if assigned_targets[i] is None:
                    assigned_targets[i] = min(
                        pellets, key=lambda p: env.get_distance(pacman_positions[i], p)
                    )

        # 3. Determine action for each agent toward assigned target with evasion
        joint_action = []
        for i in range(k):
            my_pos = pacman_positions[i]
            target = assigned_targets[i]
            valid_actions = env.get_valid_actions(my_pos)

            best_score = -float("inf")
            best_actions = []

            for action in valid_actions:
                dx, dy = ACTION_DELTAS[action]
                next_pos = (my_pos[0] + dx, my_pos[1] + dy)
                if next_pos not in env.open_cells and action != ACTION_STAY:
                    continue

                score = 0.0

                # Ghost evasion
                min_ghost_dist = min(
                    (env.get_distance(next_pos, g) for g in ghosts), default=999
                )
                if min_ghost_dist <= 1:
                    score -= 1000.0
                elif min_ghost_dist <= self.danger_dist:
                    score -= 35.0 / (min_ghost_dist + 0.1)

                # Target progress
                if target is not None:
                    dist_to_target = env.get_distance(next_pos, target)
                    score -= dist_to_target * 3.5
                    if next_pos in env.pellets:
                        score += 50.0
                    if next_pos == target:
                        score += 30.0
                elif pellets:
                    min_p_dist = min(env.get_distance(next_pos, p) for p in pellets)
                    score -= min_p_dist * 3.0
                    if next_pos in env.pellets:
                        score += 50.0

                if score > best_score:
                    best_score = score
                    best_actions = [action]
                elif abs(score - best_score) < 1e-5:
                    best_actions.append(action)

            action_choice = self.rng.choice(best_actions) if best_actions else ACTION_STAY
            joint_action.append(action_choice)

        return joint_action
