"""
Distributed Constraint Optimization Problem (DCOP) Module:
1. Region Partitioning: Voronoi graph partitioning into m regions.
2. DCOP Cost Model (Equation 3):
   f_i(z_i; s) = -rho * (|P cap R_{z_i}| / (1 + d(x^i, c_{z_i})))
                 + c * sum_{j} 1[d(y^j, R_{z_i}) <= zeta]
                 + eta * sum_{l != i} 1[z_l == z_i]
3. Solvers under message budget B:
   - DSA (Distributed Stochastic Algorithm)
   - Max-Sum (Factor Graph Belief Propagation)
   - Exact Centralized Solver (for theoretical benchmark)
"""

import collections
import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.environment import (
    ACTION_DELTAS,
    ACTION_STAY,
    ACTIONS,
    MultiAgentPacmanEnv,
)


class RegionPartition:
    """Partitions the open cells of a maze into m balanced contiguous regions."""
    def __init__(self, env: MultiAgentPacmanEnv, num_regions: int = 3):
        self.env = env
        self.num_regions = num_regions
        self.region_cells: Dict[int, Set[Tuple[int, int]]] = {
            r: set() for r in range(num_regions)
        }
        self.cell_to_region: Dict[Tuple[int, int], int] = {}
        self.centers: List[Tuple[int, int]] = []
        self._partition()

    def _partition(self):
        open_list = sorted(list(self.env.open_cells))
        if not open_list:
            return

        # Choose m well-separated seed centers using furthest-point sampling
        seeds = [open_list[0]]
        for _ in range(1, self.num_regions):
            best_cell = None
            max_dist = -1
            for cell in open_list:
                if cell in seeds:
                    continue
                min_d = min(self.env.get_distance(cell, s) for s in seeds)
                if min_d > max_dist:
                    max_dist = min_d
                    best_cell = cell
            seeds.append(best_cell if best_cell is not None else open_list[len(seeds)])

        self.centers = seeds

        # Voronoi assignment: each open cell belongs to closest center
        for cell in open_list:
            dists = [self.env.get_distance(cell, c) for c in self.centers]
            r = int(np.argmin(dists))
            self.region_cells[r].add(cell)
            self.cell_to_region[cell] = r

        # Recompute geometric graph centers (medoids) of each region
        new_centers = []
        for r in range(self.num_regions):
            cells = list(self.region_cells[r])
            if not cells:
                new_centers.append(self.centers[r])
                continue
            # Medoid minimizes sum of distances to other cells in region
            medoid = min(
                cells,
                key=lambda u: sum(self.env.get_distance(u, v) for v in cells),
            )
            new_centers.append(medoid)
        self.centers = new_centers


class DCOPSolver:
    def __init__(
        self,
        partition: RegionPartition,
        rho: float = 10.0,
        c_ghost: float = 15.0,
        zeta: int = 2,
        eta: float = 25.0,
        rng: Optional[random.Random] = None,
    ):
        self.partition = partition
        self.env = partition.env
        self.m = partition.num_regions
        self.rho = rho
        self.c_ghost = c_ghost
        self.zeta = zeta
        self.eta = eta
        self.rng = rng if rng is not None else random.Random()

    def _unary_cost(self, agent_id: int, region: int) -> float:
        """
        Unary cost for agent i choosing region:
        -rho * |P cap R| / (1 + d(x^i, c_R)) + c * sum_j 1[d(y^j, R) <= zeta]
        """
        x_i = self.env.pacman_positions[agent_id]
        c_r = self.partition.centers[region]
        d_x_c = self.env.get_distance(x_i, c_r)

        # Count remaining pellets in region
        r_cells = self.partition.region_cells[region]
        pellets_in_r = sum(1 for p in self.env.pellets if p in r_cells)

        cost = -self.rho * (pellets_in_r / (1.0 + d_x_c))

        # Ghost danger
        for y_j in self.env.ghost_positions:
            d_ghost_r = min(
                (self.env.get_distance(y_j, u) for u in r_cells), default=999
            )
            if d_ghost_r <= self.zeta:
                cost += self.c_ghost

        return float(cost)

    def compute_joint_cost(self, assignment: List[int]) -> float:
        """Total social cost under assignment z."""
        total = 0.0
        k = len(assignment)
        for i in range(k):
            total += self._unary_cost(i, assignment[i])
            for l in range(k):
                if l != i and assignment[l] == assignment[i]:
                    total += self.eta
        return total

    def solve_exact(self) -> List[int]:
        """Exhaustive search for ground-truth optimal assignment."""
        k = self.env.num_pacman
        import itertools
        best_cost = float("inf")
        best_z = [0] * k
        for z in itertools.product(range(self.m), repeat=k):
            cost = self.compute_joint_cost(list(z))
            if cost < best_cost:
                best_cost = cost
                best_z = list(z)
        return best_z

    def solve_dsa(self, message_budget: int = 10, p_dsa: float = 0.7) -> Tuple[List[int], int]:
        """
        Distributed Stochastic Algorithm (DSA).
        message_budget: Maximum messages exchanged across all agents.
        Returns: (final_assignment, messages_used)
        """
        k = self.env.num_pacman
        # Initial assignment: purely selfish unary choice (0 messages)
        z = [int(np.argmin([self._unary_cost(i, r) for r in range(self.m)])) for i in range(k)]
        messages_sent = 0

        if message_budget <= 0:
            return z, 0

        # Each communication round costs k * (k - 1) messages (all-to-all broadcast)
        msgs_per_round = k * (k - 1) if k > 1 else 1
        max_rounds = max(1, message_budget // msgs_per_round)

        for _ in range(max_rounds):
            if messages_sent + msgs_per_round > message_budget:
                break
            messages_sent += msgs_per_round

            new_z = list(z)
            for i in range(k):
                # Evaluate local cost for every candidate region given others' current choices
                candidate_costs = []
                for cand_r in range(self.m):
                    local_cost = self._unary_cost(i, cand_r)
                    for l in range(k):
                        if l != i and z[l] == cand_r:
                            local_cost += 2 * self.eta
                    candidate_costs.append(local_cost)

                best_r = int(np.argmin(candidate_costs))
                curr_cost = candidate_costs[z[i]]
                if candidate_costs[best_r] < curr_cost:
                    # Stochastic activation
                    if self.rng.random() < p_dsa:
                        new_z[i] = best_r

            z = new_z

        return z, messages_sent

    def solve_max_sum(self, message_budget: int = 20) -> Tuple[List[int], int]:
        """
        Max-Sum Algorithm (Factor Graph Belief Propagation).
        message_budget: Maximum messages exchanged.
        Returns: (final_assignment, messages_used)
        """
        k = self.env.num_pacman
        if k <= 1 or message_budget <= 0:
            z = [int(np.argmin([self._unary_cost(i, r) for r in range(self.m)])) for i in range(k)]
            return z, 0

        # Messages from variable i to factor (i, l): q_{i->il}[z_i]
        # Messages from factor (i, l) to variable i: r_{il->i}[z_i]
        edges = [(i, l) for i in range(k) for l in range(k) if i != l]
        msgs_per_iteration = len(edges) * 2  # Bidirectional factor-variable messages
        num_iters = max(1, message_budget // max(1, msgs_per_iteration))

        q_msg: Dict[Tuple[int, int], np.ndarray] = {
            (i, l): np.zeros(self.m) for i, l in edges
        }
        r_msg: Dict[Tuple[int, int], np.ndarray] = {
            (i, l): np.zeros(self.m) for i, l in edges
        }

        unary = np.zeros((k, self.m))
        for i in range(k):
            for r in range(self.m):
                unary[i, r] = self._unary_cost(i, r)

        messages_sent = 0
        for _ in range(num_iters):
            if messages_sent + msgs_per_iteration > message_budget:
                break
            messages_sent += msgs_per_iteration

            # 1. Variable to factor messages
            for i, l in edges:
                # Sum unary plus incoming r_msg from other neighbors
                in_sum = np.array(unary[i])
                for other in range(k):
                    if other != i and other != l:
                        in_sum += r_msg[(other, i)]
                # Normalize by subtracting min to avoid numerical explosion
                in_sum -= np.min(in_sum)
                q_msg[(i, l)] = in_sum

            # 2. Factor to variable messages
            for i, l in edges:
                # Pairwise cost: 2*eta if z_i == z_l else 0
                r_vec = np.zeros(self.m)
                for zi in range(self.m):
                    min_val = float("inf")
                    for zl in range(self.m):
                        cost_pair = (2.0 * self.eta) if (zi == zl) else 0.0
                        val = cost_pair + q_msg[(l, i)][zl]
                        if val < min_val:
                            min_val = val
                    r_vec[zi] = min_val
                r_vec -= np.min(r_vec)
                r_msg[(l, i)] = r_vec

        # Final marginal belief computation
        z = []
        for i in range(k):
            belief = np.array(unary[i])
            for l in range(k):
                if l != i:
                    belief += r_msg[(l, i)]
            z.append(int(np.argmin(belief)))

        return z, messages_sent


class DCOPAgent:
    """Agent executing policies directed by DCOP region assignment."""
    def __init__(
        self,
        agent_id: int,
        partition: RegionPartition,
        danger_dist: int = 2,
        rng: Optional[random.Random] = None,
    ):
        self.agent_id = agent_id
        self.partition = partition
        self.danger_dist = danger_dist
        self.rng = rng if rng is not None else random.Random()
        self.assigned_region: int = 0

    def set_region(self, region: int):
        self.assigned_region = region

    def get_action(self, env: MultiAgentPacmanEnv) -> int:
        my_pos = env.pacman_positions[self.agent_id]
        valid_actions = env.get_valid_actions(my_pos)
        if not valid_actions:
            return ACTION_STAY

        # Find target pellet: prefer pellets in assigned region
        region_cells = self.partition.region_cells[self.assigned_region]
        region_pellets = [p for p in env.pellets if p in region_cells]
        target_pellets = region_pellets if region_pellets else list(env.pellets)

        ghosts = env.ghost_positions
        best_score = -float("inf")
        best_actions = []

        for action in valid_actions:
            dx, dy = ACTION_DELTAS[action]
            next_pos = (my_pos[0] + dx, my_pos[1] + dy)
            if next_pos not in env.open_cells and action != ACTION_STAY:
                continue

            score = 0.0

            # Ghost avoidance
            min_ghost_dist = min(
                (env.get_distance(next_pos, g) for g in ghosts), default=999
            )
            if min_ghost_dist <= 1:
                score -= 1000.0
            elif min_ghost_dist <= self.danger_dist:
                score -= 25.0 / (min_ghost_dist + 0.1)

            # Navigation to assigned region / pellets
            if target_pellets:
                min_p_dist = min(env.get_distance(next_pos, p) for p in target_pellets)
                score -= min_p_dist * 2.5
                if next_pos in target_pellets:
                    score += 50.0
            else:
                # If all pellets cleared, move toward center of region
                center = self.partition.centers[self.assigned_region]
                score -= env.get_distance(next_pos, center) * 1.0

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif abs(score - best_score) < 1e-5:
                best_actions.append(action)

        return self.rng.choice(best_actions) if best_actions else ACTION_STAY
