import logging
from collections import namedtuple
import random
from typing import Callable
from copy import deepcopy
from itertools import accumulate
from operator import xor


Nimply = namedtuple("Nimply", "row, num_objects")


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


def pure_random(state: Nim) -> Nimply:
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)


def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))


def nim_sum(state: Nim) -> int:
    *_, result = accumulate(state.rows, xor)
    return result


def cook_status(state: Nim) -> dict:
    """Giving a State, returns a dict with some high level information"""
    cooked = dict()
    cooked["possible_moves"] = [
        (r, o)
        for r, c in enumerate(state.rows)
        for o in range(1, c + 1)
        if state.k is None or o <= state.k
    ]
    # counting the nr of rows where there is more than 1 object
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    # row with less elements
    cooked["shortest_row"] = min(
        (x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1]
    )[0]
    # row with more elements
    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[
        0
    ]
    cooked["nim_sum"] = nim_sum(state)

    brute_force = list()
    # for all possible moves apply nimsum
    for m in cooked["possible_moves"]:
        tmp = deepcopy(state)
        tmp.nimming(m)
        brute_force.append((m, nim_sum(tmp)))
    cooked["brute_force"] = brute_force

    return cooked


def optimal_startegy(state: Nim) -> Nimply:
    """between all possible nimsum moves,
    choose the one that give the result 0
    (0 sum of binary column see wikipedia)"""
    data = cook_status(state)
    # if there is a value with nimsum 0, takes that (optimal)
    # otherwise goes random
    return next(
        (bf for bf in data["brute_force"] if bf[1] == 0),
        random.choice(data["brute_force"]),
    )[0]


def make_strategy(genome: dict) -> Callable:
    def evolvable(state: Nim) -> Nimply:
        """Evolvable strategy because changing the value of probability 'p'
        of taking a decision with respect to the other, changes the behavior"""
        # get the high level interpretation of board
        data = cook_status(state)
        # random choice for taking elements from the longest or shorter row
        if random.random() < genome["p"]:
            ply = Nimply(
                data["shortest_row"],
                random.randint(1, state.rows[data["shortest_row"]]),
            )
        else:
            ply = Nimply(
                data["longest_row"], random.randint(1, state.rows[data["longest_row"]])
            )

        return ply

    return evolvable


def evaluate(strategy: Callable, NUM_MATCHES: int, NIM_SIZE: int) -> int:
    """Evaluates a strategy VS the always optimal one (nimsum)"""
    # if optimal strategy starts first, will certainly win
    opponent = (strategy, optimal_startegy)
    won = 0

    # run 100 matches
    for m in range(NUM_MATCHES):
        nim = Nim(NIM_SIZE)
        player = 0
        logging.debug(f"status: Initial board game {m+1} -> {nim}")
        while nim:
            ply = opponent[player](nim)
            nim.nimming(ply)
            logging.debug(f"status: After player {player} -> {nim}")
            player = 1 - player
        if player == 1:
            won += 1
    return won


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    NUM_MATCHES = 10
    NIM_SIZE = 10

    won_by_player_1 = evaluate(make_strategy({"p": 0.9999}), NUM_MATCHES, NIM_SIZE)

    won_by_player_2 = NUM_MATCHES - won_by_player_1

    win_rate = won_by_player_1 / NUM_MATCHES

    logging.info(
        f"\nNUM MATCHED {NUM_MATCHES}:\n"
        f"Nr wins Player 1: {won_by_player_1} |\n"
        f"Nr wins Player 2: {won_by_player_2} |\n"
        f"Win Rate plyer 1 {win_rate} |"
    )
