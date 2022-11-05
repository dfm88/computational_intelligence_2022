import random
import sys

sys.path.append(".")

import functools
import itertools
import logging
import math
from dataclasses import dataclass

logging.basicConfig(format="%(message)s", level=logging.DEBUG)


@dataclass
class Individual:
    genome: tuple[bool]
    fitness: tuple[int]  # (completeness, weight)

    def __post_init__(self):
        self.completeness = self.fitness[0]
        self.weight = self.fitness[1]

    def __repr__(self) -> str:
        return (
            f"genome: {self.genome}, "
            f"completeness: {self.completeness}, "
            f"weight: {self.weight}"
        )


@functools.lru_cache(512)
def get_fitness(genome: tuple[bool], all_lists: list[set[int]], N) -> tuple[int]:
    """
    Takes the corresponding value in `problem_list` if the value
    is True and evaluates its fitness (sum of lengths of all lists
    in the corresponding `problem` value)

    Args:
        genome (tuple[bool])

    Returns:
        tuple[int]: (solution_completeness, weight)
    """

    # select from `all_lists` only the element where the gene in genome is True
    genome_solution = list(itertools.compress(data=all_lists, selectors=genome))

    weight = sum((len(el) for el in genome_solution))
    # how many single number are in solution from 0 to N (computes only if accumulator is lower than N)
    solution_completeness = len(
        functools.reduce(lambda x, y: x if len(x) >= N else x | y, genome_solution)
    )
    result = (solution_completeness, - weight)
    return result


def generate_population(all_lists: list[set[int]]) -> list[Individual]:
    population = list()

    # generates a len(POPULATION_SIZE) of individuals, each long
    # len(PROBLEM_SIZE) and evaluates its fitness
    for _ in range(POPULATION_SIZE):
        genome = tuple((random.choice([True, False]) for _ in range(PROBLEM_SIZE)))
        genome = validate_and_mutate(
            genome=genome,
        )

        population.append(
            Individual(genome=genome, fitness=get_fitness(genome, all_lists, N))
        )

    logging.info(
        f"init: pop_size={len(population)}; max={max(population, key=lambda i: i.fitness)}"
    )
    return population


def tournament(population, tournament_size=2):
    chosen = random.choices(population, k=tournament_size)
    selected = max(chosen, key=lambda i: i.fitness)
    return selected


def cross_over(g1, g2):
    cut = random.randint(0, PROBLEM_SIZE)
    genome = g1[:cut] + g2[cut:]
    return genome


def mutation(genome: tuple[bool], n_genes: int = None) -> tuple[bool]:
    """Turns as many genome value as indicated
    by `n_genes`, default value is 30 % of genome length

    Args:
        genome (tuple[bool],): genome to mutate
        n_genes (int, optional): Nr of genes to flip. Defaults to 1.

    Returns:
        tuple[bool]
    """
    max_range = len(genome)
    if not n_genes:
        n_genes = math.ceil(0.3 * max_range)

    if n_genes > max_range:
        raise ValueError('Nr of genes greater than genome length')

    indexes_to_flip = random.sample(range(0, max_range), n_genes)
    new_genome = tuple()
    for gene in indexes_to_flip:
        new_genome = genome[:gene] + (not genome[gene],) + genome[gene + 1:]
        genome = new_genome

    return new_genome


@functools.lru_cache(512)
def validate_and_mutate(
    genome: tuple[bool],
    nr_genes: int = 1
) -> list[bool]:
    """If genome has all values as False, mutate
    a `nr_genes` genes in genome

    Returns:
        list[bool]: new genome if it was not valid
    """

    # if not True value in genome
    if not any(genome):
        genome = mutation(
            genome,
            n_genes=nr_genes
        )
    return genome


def problem(N=5, seed=42) -> list[set[int]]:
    random.seed(seed)
    return tuple(
        frozenset(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2)))
        for _ in range(random.randint(N, N * 5))
    )


if __name__ == "__main__":
    N = 10
    all_lists = problem(N=N)
    PROBLEM_SIZE = len(all_lists)
    POPULATION_SIZE = 600
    OFFSPRING_SIZE = 200
    NUM_GENERATIONS = 1000

    population = generate_population(all_lists=all_lists)

    for g in range(NUM_GENERATIONS):
        offspring = list()
        for i in range(OFFSPRING_SIZE):
            if random.random() > 0.2:
                p = tournament(population, tournament_size=2)
                o = mutation(p.genome, n_genes=1)
            else:
                p1 = tournament(population, tournament_size=2)
                p2 = tournament(population, tournament_size=2)
                o = cross_over(p1.genome, p2.genome)
            o = validate_and_mutate(o)
            f = get_fitness(o, all_lists, N)
            offspring.append(Individual(o, f))

        population += offspring
        population = sorted(population, key=lambda i: i.fitness, reverse=True)[
            :POPULATION_SIZE
        ]

    print('f', population[0].fitness)
    print(get_fitness.cache_info())
    print(validate_and_mutate.cache_info())
