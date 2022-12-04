import logging
import random
import sys
from dataclasses import dataclass
from typing import Callable

sys.path.append(".")

from lab3.utils import (Player0Strategies, Player1Strategies, Statistics,
                        evaluate, play_nim)


@dataclass
class Individual:
    genome: list[float]  # probability of choosing a given strategy
    fitness: float  # win rate
    strategies_names: list[str]  # list of strategies names, just for logging

    def __post_init__(self):
        assert len(self.genome) == len(
            self.strategies_names
        ), "array of probability different in length from array of strategies"

        # list of tuple with
        # (strategy_name, probability to pick that strategy)
        # just for logging
        self.probability_of_strategy = [
            (strategy, prob)
            for (strategy, prob) in zip(self.strategies_names, self.genome)
        ]

    def __repr__(self) -> str:
        return (
            f"strategies  : {self.probability_of_strategy},\n"
            f"fitness: {self.fitness}"
        )


def align_probabilities(list_prob: list[float]) -> list[float]:
    """Utility function that given an array of float nr from 0 to 1
    makes it possible that the sum of these numbers sum up to 1"""
    sum_ = sum(list_prob)
    return [x / sum_ for x in list_prob]


def generate_population(
    population_size: int,
    p0_strategies: list[Callable],
    play_game: Callable[[], float],
) -> list[Individual]:
    """Creates an array of len(population_size) of Individuals
    each long len(p0_strategies) hence the number of
    available strategies of Player0

    Args:
        population_size (int)
        p0_strategies (list[Callable]): list of strategies of player 0 (human)
        play_game (Callable[[], float])

    Returns:
        list[Individual]
    """
    population: list[Individual] = list()

    tot_p0_strategies: int = len(p0_strategies)

    for _ in range(population_size):
        random_prob = [random.random() for _ in range(tot_p0_strategies)]
        # ensure that all probabilities in genome sum up to 1
        aligned_random_prob = align_probabilities(random_prob)

        population.append(
            Individual(
                genome=aligned_random_prob,
                fitness=play_game(),
                strategies_names=[fn.__name__ for fn in p0_strategies],
            )
        )

    logging.info(
        f"init: pop_size={len(population)}; max={max(population, key=lambda i: i.fitness)}"
    )
    return population


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # Nim Sum parameters
    NUM_MATCHES = 20
    NIM_SIZE = 10
    K = 3

    # Genetic Algorithm parameters
    POPULATION_SIZE = 20
    OFFSPRING_SIZE = 50
    NUM_GENERATIONS = 200

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0Strategies(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    play_game_fn: Callable = lambda: play_nim(
        num_matches=NUM_MATCHES,
        nim_size=NIM_SIZE,
        k=K,
        p0_strategies_list=p0_strategies.get_all_strategies(),
        p1_strategy=p1_strategy,
    )

    population = generate_population(
        population_size=POPULATION_SIZE,
        p0_strategies=p0_strategies.get_all_strategies(),
        play_game=play_game_fn,
    )
