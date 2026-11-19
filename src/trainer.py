"""
Training and Evaluation Framework for Multi-Agent Pac-Man:
Coordinates:
- Training loops for Selfish Q, Equal-Share Q, and Shapley-Shaped Q
- Evaluation loops across multiple random seeds
- Computation of returns, fairness, captures, and completion steps
"""

import copy
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.baselines import CentralizedTeamPlanner, IndependentGreedyAgent, RandomAgent
from src.dcop import DCOPAgent, DCOPSolver, RegionPartition
from src.environment import MultiAgentPacmanEnv
from src.metrics import compute_ci95, compute_gini_index
from src.q_learning import QLearningAgent
from src.shapley import StepwiseShapleyCredit


def train_q_learning(
    layout_name: str = "small_grid",
    num_pacman: int = 2,
    num_ghosts: int = 1,
    ghost_beta: float = 0.5,
    mode: str = "selfish",  # 'selfish', 'equal_share', or 'shapley'
    shapley_lambda: float = 0.5,
    num_episodes: int = 100,
    alpha: float = 0.05,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.02,
    seed: int = 42,
) -> List[QLearningAgent]:
    """
    Trains k Q-learning agents under the specified reward formulation.
    """
    rng = random.Random(seed)
    env = MultiAgentPacmanEnv(
        layout_name=layout_name,
        num_pacman=num_pacman,
        num_ghosts=num_ghosts,
        ghost_beta=ghost_beta,
        seed=seed,
    )

    agents = [
        QLearningAgent(
            agent_id=i,
            alpha=alpha,
            gamma=env.gamma,
            epsilon=epsilon_start,
            rng=random.Random(seed + i * 17),
        )
        for i in range(num_pacman)
    ]

    shapley_credit = StepwiseShapleyCredit(num_pacman)

    for ep in range(num_episodes):
        # Epsilon decay
        cur_eps = epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, num_episodes))
        for ag in agents:
            ag.epsilon = cur_eps

        state, _ = env.reset(seed=seed + ep * 101)
        done = False

        while not done:
            # Snapshot environment for feature calculation
            env_before = copy.deepcopy(env)

            # Choose actions
            joint_action = [ag.get_action(env_before, explore=True) for ag in agents]

            # Collect contender information for Shapley credit before step
            next_positions = []
            for i, a in enumerate(joint_action):
                dx, dy = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)][a]
                cand = (env.pacman_positions[i][0] + dx, env.pacman_positions[i][1] + dy)
                next_positions.append(cand if cand in env.open_cells else env.pacman_positions[i])

            pellet_eaten_by = {}
            for i, p_pos in enumerate(next_positions):
                if p_pos in env.pellets:
                    if p_pos not in pellet_eaten_by:
                        pellet_eaten_by[p_pos] = []
                    pellet_eaten_by[p_pos].append(i)

            # Step environment
            next_state, rewards, done, info = env.step(joint_action)

            # Determine reward shaping
            if mode == "selfish":
                effective_rewards = list(rewards)
            elif mode == "equal_share":
                team_r = sum(rewards)
                effective_rewards = [team_r / num_pacman] * num_pacman
            elif mode == "shapley":
                caught_set = {
                    i for i in range(num_pacman) if info["agent_captures"][i] > 0
                }
                phi = shapley_credit.compute_step_shapley(
                    individual_rewards=rewards,
                    pellet_eaten_by=pellet_eaten_by,
                    caught_agents=caught_set,
                    pellet_reward=env.pellet_reward,
                    ghost_penalty=env.ghost_penalty,
                    step_penalty=env.step_penalty,
                )
                effective_rewards = shapley_credit.shape_rewards(
                    rewards, phi, lmbda=shapley_lambda
                )
            else:
                effective_rewards = list(rewards)

            # Q-learning update
            for i in range(num_pacman):
                agents[i].update(
                    env=env_before,
                    action=joint_action[i],
                    reward=effective_rewards[i],
                    next_env=env,
                    done=done,
                )

    return agents


def evaluate_policy(
    policy_type: str,
    layout_name: str = "small_grid",
    num_pacman: int = 2,
    num_ghosts: int = 1,
    ghost_beta: float = 0.5,
    trained_agents: Optional[List[QLearningAgent]] = None,
    dcop_algo: str = "dsa",  # 'dsa' or 'max_sum'
    dcop_budget: int = 10,
    num_episodes: int = 30,
    seed: int = 1234,
) -> Dict[str, any]:
    """
    Evaluates a specific policy type over num_episodes.
    Supported types:
    - 'random'
    - 'greedy'
    - 'centralized'
    - 'q_learning'
    - 'dcop'
    """
    team_returns = []
    agent_returns_all = []
    captures_all = []
    steps_all = []
    ginis = []

    rng = random.Random(seed)

    for ep in range(num_episodes):
        ep_seed = seed + ep * 73
        env = MultiAgentPacmanEnv(
            layout_name=layout_name,
            num_pacman=num_pacman,
            num_ghosts=num_ghosts,
            ghost_beta=ghost_beta,
            seed=ep_seed,
        )

        # Policy setup
        if policy_type == "random":
            agents = [
                RandomAgent(i, rng=random.Random(ep_seed + i))
                for i in range(num_pacman)
            ]
        elif policy_type == "greedy":
            agents = [
                IndependentGreedyAgent(i, rng=random.Random(ep_seed + i))
                for i in range(num_pacman)
            ]
        elif policy_type == "centralized":
            planner = CentralizedTeamPlanner(rng=random.Random(ep_seed))
        elif policy_type == "dcop":
            partition = RegionPartition(env, num_regions=max(2, num_pacman))
            solver = DCOPSolver(partition, rng=random.Random(ep_seed))
            agents = [
                DCOPAgent(i, partition=partition, rng=random.Random(ep_seed + i))
                for i in range(num_pacman)
            ]
        elif policy_type == "q_learning":
            agents = trained_agents
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        state, _ = env.reset(seed=ep_seed)
        done = False
        discounted_returns = [0.0] * num_pacman
        current_gamma = 1.0

        while not done:
            if policy_type == "centralized":
                joint_action = planner.get_joint_action(env)
            elif policy_type == "dcop":
                # Solve DCOP region assignment under message budget
                if dcop_algo == "dsa":
                    z, _ = solver.solve_dsa(message_budget=dcop_budget)
                else:
                    z, _ = solver.solve_max_sum(message_budget=dcop_budget)

                for i, r in enumerate(z):
                    agents[i].set_region(r)
                joint_action = [agents[i].get_action(env) for i in range(num_pacman)]
            elif policy_type == "q_learning":
                joint_action = [agents[i].get_action(env, explore=False) for i in range(num_pacman)]
            else:  # random or greedy
                joint_action = [agents[i].get_action(env) for i in range(num_pacman)]

            next_state, rewards, done, info = env.step(joint_action)

            for i in range(num_pacman):
                discounted_returns[i] += current_gamma * rewards[i]
            current_gamma *= env.gamma

        team_ret = sum(discounted_returns)
        team_returns.append(team_ret)
        agent_returns_all.append(discounted_returns)
        captures_all.append(info["total_captures"])
        steps_all.append(info["step_count"])
        ginis.append(compute_gini_index(discounted_returns))

    mean_ret, ci_ret, _ = compute_ci95(team_returns)
    mean_captures, ci_captures, _ = compute_ci95(captures_all)
    mean_steps, ci_steps, _ = compute_ci95(steps_all)
    mean_gini, ci_gini, _ = compute_ci95(ginis)

    return {
        "policy_type": policy_type,
        "team_return_mean": mean_ret,
        "team_return_ci": ci_ret,
        "captures_mean": mean_captures,
        "captures_ci": ci_captures,
        "steps_mean": mean_steps,
        "steps_ci": ci_steps,
        "gini_mean": mean_gini,
        "gini_ci": ci_gini,
        "all_team_returns": team_returns,
        "all_agent_returns": agent_returns_all,
    }
