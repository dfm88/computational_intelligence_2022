import functools
import logging
import sys

sys.path.append(".")

from lab3.utils import (Player0MinMaxStrategy, Player1Strategies,
                        Statistics, play_nim)

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":

    NUM_MATCHES = 20
    NIM_SIZE = 5
    K = 3

    MAX_DEPTH = 20

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0MinMaxStrategy(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    p0_strategy = functools.partial(p0_strategies.min_max_strategy, max_depth=MAX_DEPTH)
    p0_strategy.__name__ = p0_strategy.func.__name__

    game_stat: Statistics = play_nim(
        num_matches=NUM_MATCHES,
        nim_size=NIM_SIZE,
        k=K,
        p0_strategy=p0_strategy,
        p1_strategy=p1_strategy,
    )

    logging.info(
        f"\nNUM MATCHES={NUM_MATCHES}, K={K}, using "
        f"{game_stat.strategy_p0_name}' vs '{game_stat.strategy_p1_name}':\n"
        f"Nr wins Player 0: {game_stat.nr_wins_player0} |\n"
        f"Nr wins Player 1: {game_stat.nr_wins_player1} |\n"
        f"Win Rate plyer 0 {game_stat.win_rate_player0} |\n\n"
    )
