"""
Multi-Agent Q-Learning with Linear Feature Approximation:
Supports:
- Independent Q-Learning (Selfish Equilibrium Baseline)
- Equal-Share Q-Learning (r_team / k)
- Shapley-Shaped Q-Learning (Equation P4)
"""

import collections
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.environment import (
    ACTION_DELTAS,
    ACTION_STAY,
    ACTIONS,
    MultiAgentPacmanEnv,
)


class FeatureExtractor:
    """Extracts state-action features for a Pac-Man agent."""
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def extract_features(
        self, env: MultiAgentPacmanEnv, action: int
    ) -> Dict[str, float]:
        my_pos = env.pacman_positions[self.agent_id]
        dx, dy = ACTION_DELTAS[action]
        next_pos = (my_pos[0] + dx, my_pos[1] + dy)

        # If invalid move, return harsh penalty features
        if next_pos not in env.open_cells and action != ACTION_STAY:
            return {"bias": 1.0, "invalid": 1.0}

        features: Dict[str, float] = {"bias": 1.0}

        # 1. Pellet features
        pellets = list(env.pellets)
        if pellets:
            dists = [env.get_distance(next_pos, p) for p in pellets]
            min_dist = min(dists)
            nearest_pellet = pellets[int(np.argmin(dists))]
            # Normalized pellet distance
            features["closest_pellet_dist"] = min_dist / 10.0
            features["eats_pellet"] = 1.0 if next_pos in env.pellets else 0.0

            # Contention feature: is another teammate closer to this same pellet?
            contention = 0.0
            for other_i, other_pos in enumerate(env.pacman_positions):
                if other_i != self.agent_id:
                    other_dist = env.get_distance(other_pos, nearest_pellet)
                    if other_dist <= min_dist:
                        contention = 1.0
                        break
            features["teammate_contention"] = contention
        else:
            features["closest_pellet_dist"] = 0.0
            features["eats_pellet"] = 0.0
            features["teammate_contention"] = 0.0

        # 2. Ghost features
        ghosts = env.ghost_positions
        if ghosts:
            min_ghost_dist = min(env.get_distance(next_pos, g) for g in ghosts)
            features["ghost_danger_1"] = 1.0 if min_ghost_dist <= 1 else 0.0
            features["ghost_danger_2"] = 1.0 if min_ghost_dist == 2 else 0.0
        else:
            features["ghost_danger_1"] = 0.0
            features["ghost_danger_2"] = 0.0

        # 3. Teammate proximity (spatial collision avoidance)
        min_tm_dist = min(
            (
                env.get_distance(next_pos, other_pos)
                for oi, other_pos in enumerate(env.pacman_positions)
                if oi != self.agent_id
            ),
            default=999,
        )
        features["teammate_crowding"] = 1.0 if min_tm_dist <= 1 else 0.0

        return features


class QLearningAgent:
    """Approximate Q-Learning agent using linear feature weights."""
    def __init__(
        self,
        agent_id: int,
        alpha: float = 0.05,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        rng: Optional[random.Random] = None,
    ):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng if rng is not None else random.Random()

        self.feature_extractor = FeatureExtractor(agent_id)
        self.weights: collections.defaultdict = collections.defaultdict(float)

        # Initialize sensible heuristic weights for fast, stable training
        self.weights["eats_pellet"] = 15.0
        self.weights["closest_pellet_dist"] = -5.0
        self.weights["ghost_danger_1"] = -40.0
        self.weights["ghost_danger_2"] = -10.0

    def get_q_value(self, env: MultiAgentPacmanEnv, action: int) -> float:
        features = self.feature_extractor.extract_features(env, action)
        q = 0.0
        for f_name, f_val in features.items():
            q += self.weights[f_name] * f_val
        return q

    def get_max_q(self, env: MultiAgentPacmanEnv) -> float:
        my_pos = env.pacman_positions[self.agent_id]
        valid_actions = env.get_valid_actions(my_pos)
        if not valid_actions:
            return 0.0
        return max(self.get_q_value(env, a) for a in valid_actions)

    def get_action(self, env: MultiAgentPacmanEnv, explore: bool = True) -> int:
        my_pos = env.pacman_positions[self.agent_id]
        valid_actions = env.get_valid_actions(my_pos)
        if not valid_actions:
            return ACTION_STAY

        # Epsilon-greedy
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)

        q_values = [self.get_q_value(env, a) for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if abs(q - max_q) < 1e-6]
        return self.rng.choice(best_actions)

    def update(
        self,
        env: MultiAgentPacmanEnv,
        action: int,
        reward: float,
        next_env: MultiAgentPacmanEnv,
        done: bool,
    ):
        """Standard TD-error linear Q-update."""
        features = self.feature_extractor.extract_features(env, action)
        current_q = self.get_q_value(env, action)
        max_next_q = 0.0 if done else self.get_max_q(next_env)

        td_error = reward + self.gamma * max_next_q - current_q

        for f_name, f_val in features.items():
            self.weights[f_name] += self.alpha * td_error * f_val
