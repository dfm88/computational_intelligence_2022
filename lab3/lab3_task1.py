import logging
import sys

sys.path.append(".")

from lab3.utils import (
    Player0Strategies,
    Player1Strategies,
    Statistics,
    play_nim,
)

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":

    NUM_MATCHES = 20
    NIM_SIZE = 10
    K = 3

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0Strategies(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    statistics_list: list[Statistics] = list()

    for strategy in p0_strategies.get_all_strategies():
        game_stat: Statistics = play_nim(
            num_matches=NUM_MATCHES,
            nim_size=NIM_SIZE,
            k=K,
            p0_strategy=strategy,
            p1_strategy=p1_strategy,
        )
        statistics_list.append(game_stat)

    for stat in statistics_list:
        logging.info(
            f"\nNUM MATCHES={NUM_MATCHES}, K={K}, using "
            f"{stat.strategy_p0_name}' vs '{stat.strategy_p1_name}':\n"
            f"Nr wins Player 0: {stat.nr_wins_player0} |\n"
            f"Nr wins Player 1: {stat.nr_wins_player1} |\n"
            f"Win Rate plyer 0 {stat.win_rate_player0} |\n\n"
        )
