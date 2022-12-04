import inspect
import logging
import random
from abc import ABC as AbstractClass
from copy import deepcopy
from dataclasses import dataclass
from itertools import accumulate
from operator import and_, xor
from typing import Callable, NamedTuple

logging.getLogger().setLevel(logging.DEBUG)


class Nimply(NamedTuple):
    row: int  # row index
    num_objects: int  # num of object to take


class Statistics(NamedTuple):
    strategy_name: str
    nr_wins_player0: int
    nr_wins_player1: int
    win_rate_player0: float


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    def __repr__(self):
        return self.__str__()

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


def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))


def nim_sum(state: Nim) -> int:
    *_, result = accumulate(state.rows, xor)
    return result


def and_bitwise(state: Nim) -> int:
    *_, result = accumulate(state.rows, and_)
    return result


def cook_status(state: Nim) -> dict:
    """Giving a State, returns a dict with some high level information"""
    cooked = dict()
    cooked["possible_moves_my"]: dict[int, list[int]] = {
        r: [o for o in range(1, c + 1)] for r, c in enumerate(state.rows)
    }
    cooked["possible_moves"]: list[Nimply] = [
        Nimply(r, o)
        for r, c in enumerate(state.rows)
        for o in range(1, c + 1)
        if state.k is None or o <= state.k
    ]
    # counting the nr of rows where there is more than 1 object
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)

    # possible moves on rows with odd index
    cooked["odd_index_moves"]: list[Nimply] = [
        Nimply(x.row, x.num_objects)
        for x in cooked["possible_moves"]
        if (x.row % 2) != 0
    ]
    # possible moves on rows with even index
    cooked["even_index_moves"]: list[Nimply] = [
        Nimply(x.row, x.num_objects)
        for x in cooked["possible_moves"]
        if (x.row % 2) == 0
    ]
    # row with less elements
    cooked["shortest_row"] = min(
        (x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1]
    )[0]
    # row with more elements
    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[
        0
    ]
    cooked["nim_sum"] = nim_sum(state)

    cooked["and_bitwise"] = and_bitwise(state)

    return cooked


@dataclass
class BaseStrategies(AbstractClass):
    """Base Abstract class to use as container of strategies
    all strategies, if don't found a row among the available ones, should
    return a random row and random nr of objects
    """

    k: int  # upper bound to how many objects take from one row

    def get_all_strategies(self) -> list[Callable]:
        """Utility function to return all methods whose name ends with
        the `strategy` word

        Returns:
            list[Callable]
        """
        return [
            x[1]
            for x in filter(
                lambda fn: fn[0].endswith("strategy"),
                inspect.getmembers(self, inspect.ismethod),
            )
        ]


@dataclass
class Player0Strategies(BaseStrategies):
    """Possible hard coded strategies of our player"""

    def take_k_from_odd_row_strategy(self, state: Nim) -> Nimply:
        data = cook_status(state)
        return next(
            (move for move in data["odd_index_moves"] if move.num_objects <= self.k),
            random.choice(data["possible_moves"]),
        )

    def take_k_from_even_row_strategy(self, state: Nim) -> Nimply:
        data = cook_status(state)
        return next(
            (move for move in data["even_index_moves"] if move.num_objects <= self.k),
            random.choice(data["possible_moves"]),
        )

    def take_k_from_longest_row_strategy(self, state: Nim) -> Nimply:
        data = cook_status(state)
        longest_row = data["longest_row"]
        move = Nimply(row=longest_row, num_objects=self.k)
        return next(
            (m for m in data["possible_moves"] if m == move),
            random.choice(data["possible_moves"]),
        )

    def take_k_from_shortest_row_strategy(self, state: Nim) -> Nimply:
        data = cook_status(state)
        shortest_row = data["shortest_row"]
        move = Nimply(row=shortest_row, num_objects=self.k)
        return next(
            (m for m in data["possible_moves"] if m == move),
            random.choice(data["possible_moves"]),
        )

    def take_k_from_zero_andbitwise_row_strategy(self, state: Nim) -> Nimply:
        """Takes the row whose bitwise AND operation gives 0"""
        data = cook_status(state)
        # for all possible moves apply bitwise and
        moves: list[tuple[Nimply, int]] = list()
        for m in data["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming(m)
            moves.append((m, and_bitwise(tmp)))
        # if there is a value with andbitwise 0, takes that
        # otherwise goes random
        return next(
            (bf for bf in moves if bf[1] == 0 and bf[0].num_objects <= self.k),
            random.choice(moves),
        )[0]

    def take_k_from_one_andbitwise_row_strategy(self, state: Nim) -> Nimply:
        """Takes the row whose bitwise AND operation gives 1"""
        data = cook_status(state)
        # for all possible moves apply bitwise and
        moves: list[tuple[Nimply, int]] = list()
        for m in data["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming(m)
            moves.append((m, and_bitwise(tmp)))
        # if there is a value with andbitwise 1, takes that
        # otherwise goes random
        return next(
            (bf for bf in moves if bf[1] == 1 and bf[0].num_objects <= self.k),
            random.choice(moves),
        )[0]

    def take_k_from_zero_xorbitwise_row_strategy(self, state: Nim) -> Nimply:
        """Takes the row whose bitwise XOR operation gives 0"""
        data = cook_status(state)
        # for all possible moves apply bitwise and
        moves: list[tuple[Nimply, int]] = list()
        for m in data["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming(m)
            moves.append((m, nim_sum(tmp)))
        # if there is a value with andbitwise 0, takes that
        # otherwise goes random
        return next(
            (bf for bf in moves if bf[1] == 0 and bf[0].num_objects <= self.k),
            random.choice(moves),
        )[0]

    def take_k_from_one_xorbitwise_row_strategy(self, state: Nim) -> Nimply:
        """Takes the row whose bitwise XOR operation gives 1 (actually the nim sum)"""
        data = cook_status(state)
        # for all possible moves apply bitwise and
        moves: list[tuple[Nimply, int]] = list()
        for m in data["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming(m)
            moves.append((m, nim_sum(tmp)))
        # if there is a value with andbitwise 1, takes that
        # otherwise goes random
        return next(
            (bf for bf in moves if bf[1] == 1 and bf[0].num_objects <= self.k),
            random.choice(moves),
        )[0]


@dataclass
class Player1Strategies(BaseStrategies):
    """Possible hard coded strategies of opponent player"""

    def pure_random_strategy(self, state: Nim) -> Nimply:
        data = cook_status(state)
        nimply: Nimply = random.choice(data["possible_moves"])
        num_objects = random.randint(1, nimply.num_objects)
        num_objects = num_objects if num_objects <= self.k else self.k
        return Nimply(nimply.row, num_objects)

    def optimal_strategy(self, state: Nim) -> Nimply:
        """between all possible nimsum moves,
        choose the one that give the result 0
        (0 sum of binary column see wikipedia)"""
        data = cook_status(state)
        # for all possible moves apply nimsum
        brute_force = list()
        for m in data["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming(m)
            brute_force.append((m, nim_sum(tmp)))
        # if there is a value with nimsum 0, takes that (optimal)
        # otherwise goes random
        return next(
            (bf for bf in brute_force if bf[1] == 0 and bf[0].num_objects <= self.k),
            random.choice(brute_force),
        )[0]


def evaluate(
    strategy_p0: Callable,
    strategy_p1: Callable,
    NUM_MATCHES: int,
    NIM_SIZE: int,
    k: int,
) -> tuple[int, float]:
    """Evaluates a strategy

    Args:
        strategy_p0 (Callable): strategy of player 0 (human)
        strategy_p1 (Callable): strategy of player 1 (AI)
        NUM_MATCHES (int): nr of matches to play
        NIM_SIZE (int): size of nim board
        k (int): upper bound on nr. of objects to remove from a Nim row.

    Returns:
        tuple[int, float]: (
            nr of win from human player, win rate of human player
        )
    """

    # if optimal strategy starts first, will certainly win
    opponent = (strategy_p0, strategy_p1)
    won = 0
    # run 100 matches
    for m in range(NUM_MATCHES):
        nim = Nim(NIM_SIZE, k=k)
        player = 0
        logging.debug(f"status: Initial board game {m+1} -> {nim}")
        while nim:
            ply = opponent[player](nim)
            nim.nimming(ply)
            logging.debug(f"status: After player {player} -> {nim}")
            player = 1 - player
        if player == 1:
            won += 1
    return (won, won / NUM_MATCHES)


def play_nim(
    num_matches: int,
    nim_size: int,
    k: int,
    p0_strategies_list: list[Callable],
    p1_strategy: Callable,
) -> float:
    """Plays Nim

    Args:
        num_matches (int)
        nim_size (int): size of Nim board
        k (int): upper bound to objects to take from a row
        p0_strategies_list (list[Callable]): list of strategies of player 0 (human)
        p1_strategy (Callable): strategy of player 1 (AI)

    Returns:
        float: win rate of player 0 (human)
    """

    statistics_list = list()

    for strategy in p0_strategies_list:
        logging.debug(f"Using strategy {strategy.__name__} vs {p1_strategy.__name__}")

        won_by_player_0, win_rate_player_0 = evaluate(
            strategy_p0=strategy,
            strategy_p1=p1_strategy,
            NUM_MATCHES=num_matches,
            NIM_SIZE=nim_size,
            k=k,
        )

        won_by_player_1 = num_matches - won_by_player_0

        statistics_list.append(
            Statistics(
                strategy_name=strategy.__name__,
                nr_wins_player0=won_by_player_0,
                nr_wins_player1=won_by_player_1,
                win_rate_player0=win_rate_player_0,
            )
        )

    for stat in statistics_list:
        logging.info(
            f"\nNUM MATCHES={num_matches}, K={k}, using '{stat.strategy_name}' vs '{p1_strategy.__name__}':\n"
            f"Nr wins Player 0: {stat.nr_wins_player0} |\n"
            f"Nr wins Player 1: {stat.nr_wins_player1} |\n"
            f"Win Rate plyer 0 {stat.win_rate_player0} |\n\n"
        )

    return win_rate_player_0
