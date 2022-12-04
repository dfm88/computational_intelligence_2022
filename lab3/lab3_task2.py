import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     handlers=[logging.FileHandler("debug3.log"), logging.StreamHandler()],
# )
import functools
import operator
import random
import sys
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable

sys.path.append(".")

from lab3.utils import (
    Player0Strategies,
    Player1Strategies,
    Statistics,
    play_nim,
)


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


def tournament(population, tournament_size=2):
    chosen = random.choices(population, k=tournament_size)
    selected = max(chosen, key=lambda i: i.fitness)
    return selected


def mutation(genome: list[float]) -> list[float]:
    """The mutation consists in decreasing one of the gene
    by 80% on of the probabilities of picking the corresponding strategy

    Args:
        genome (list[float]): genome to mutate

    Returns:
        list[float]
    """
    max_range = len(genome)

    index_to_flip = random.randint(0, max_range - 1)

    # decrease by 80%
    genome[index_to_flip] = genome[index_to_flip] * 0.8

    new_genome = align_probabilities(list_prob=genome)

    return new_genome


def cross_over(g1: list[float], g2: list[float]) -> list[float]:
    max_range = len(g1)
    cut = random.randint(0, max_range)
    genome = g1[:cut] + g2[cut:]

    new_genome = align_probabilities(list_prob=genome)

    return new_genome


def get_strategy_by_genome(genome: list[float], strategies: list[Callable]) -> Callable:
    """accumulates the probabilities of the genome
    and given a random probability, extract the corresponding
    strategy from the strategy list

    Args:
        genome (list[float])
        strategies (list[Callable])

    Returns:
        Callable: chosen strategy
    """
    prob = random.random()

    genome_accum = accumulate(genome, operator.add)

    index = next(i for i, p in enumerate(genome_accum) if prob <= p)

    p0_strategy = strategies[index]

    return p0_strategy


def generate_population(
    population_size: int,
    p0_strategies: list[Callable],
    play_game: Callable[[Callable], Statistics],
) -> list[Individual]:
    """Creates an array of len(population_size) of Individuals
    each long len(p0_strategies) hence the number of
    available strategies of Player0

    Args:
        population_size (int)
        p0_strategies (list[Callable]): list of strategies of player 0 (human)
        play_game: Callable[[Callable], Statistics]

    Returns:
        list[Individual]
    """
    logging.info("Creating population..")
    population: list[Individual] = list()

    tot_p0_strategies: int = len(p0_strategies)

    for _ in range(population_size):
        random_prob = [random.random() for _ in range(tot_p0_strategies)]
        # ensure that all probabilities in genome sum up to 1
        aligned_random_prob = align_probabilities(random_prob)

        p0_strategy = get_strategy_by_genome(
            genome=aligned_random_prob, strategies=p0_strategies
        )

        game_stat: Statistics = play_game(p0_strategy=p0_strategy)
        population.append(
            Individual(
                genome=aligned_random_prob,
                fitness=game_stat.win_rate_player0,
                strategies_names=[fn.__name__ for fn in p0_strategies],
            )
        )

    logging.info(
        f"init: pop_size={len(population)}; max={max(population, key=lambda i: i.fitness)}"
    )
    return population


if __name__ == "__main__":

    # Nim Sum parameters
    NUM_MATCHES = 20
    NIM_SIZE = 10
    K = 3

    # Genetic Algorithm parameters
    POPULATION_SIZE = 30
    OFFSPRING_SIZE = 50
    NUM_GENERATIONS = 50

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0Strategies(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    # `p0_strategy` to be provided
    play_game_fn: Callable = functools.partial(
        play_nim,
        num_matches=NUM_MATCHES,
        nim_size=NIM_SIZE,
        k=K,
        p1_strategy=p1_strategy,
    )

    p0_strategies_list = p0_strategies.get_all_strategies()
    p0_strategies_name_list = [fn.__name__ for fn in p0_strategies_list]
    population = generate_population(
        population_size=POPULATION_SIZE,
        p0_strategies=p0_strategies_list,
        play_game=play_game_fn,
    )

    total_generations = 0
    for g in range(NUM_GENERATIONS):
        offspring = list()
        total_generations += 1
        for i in range(OFFSPRING_SIZE):
            if random.random() > 0.2:
                p = tournament(population, tournament_size=2)
                gen = mutation(p.genome)
            else:
                p1 = tournament(population, tournament_size=2)
                p2 = tournament(population, tournament_size=2)
                gen = cross_over(p1.genome, p2.genome)

            p0_strategy = get_strategy_by_genome(
                genome=gen, strategies=p0_strategies_list
            )
            logging.debug(f"{g}-{i} strategy {p0_strategy}\n")
            logging.debug(gen)

            game_stat: Statistics = play_game_fn(p0_strategy=p0_strategy)
            offspring.append(
                Individual(
                    genome=gen,
                    fitness=game_stat.win_rate_player0,
                    strategies_names=p0_strategies_name_list,
                )
            )

        population += offspring
        population = sorted(population, key=lambda i: i.fitness, reverse=True)[
            :POPULATION_SIZE
        ]
        logging.info("POPUL")
        logging.info(population)
        # # break if already found the best solution
        # if population[0].fitness == (N, -N):
        #     break

    print("f", population[0].fitness)
    print(
        f"Found best solution in {total_generations} over {NUM_GENERATIONS} generations"
    )
