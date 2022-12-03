import logging
import sys

sys.path.append(".")

from lab3.utils import (Player0Strategies, Player1Strategies, Statistics,
                        evaluate)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    NUM_MATCHES = 20
    NIM_SIZE = 10
    K = 3

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0Strategies(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    statistics_list = list()

    for strategy in p0_strategies.get_all_strategies():
        logging.info(f"Using strategy {strategy.__name__} vs {p1_strategy.__name__}")

        won_by_player_0 = evaluate(
            strategy_p0=strategy,
            strategy_p1=p1_strategy,
            NUM_MATCHES=NUM_MATCHES,
            NIM_SIZE=NIM_SIZE,
        )

        won_by_player_1 = NUM_MATCHES - won_by_player_0

        win_rate = won_by_player_0 / NUM_MATCHES

        statistics_list.append(
            Statistics(
                strategy_name=strategy.__name__,
                nr_wins_player0=won_by_player_0,
                nr_wins_player1=won_by_player_1,
                win_rate_player0=win_rate,
            )
        )

    for stat in statistics_list:
        logging.info(
            f"\nNUM MATCHES={NUM_MATCHES}, K={K}, using '{stat.strategy_name}' vs '{p1_strategy.__name__}':\n"
            f"Nr wins Player 0: {stat.nr_wins_player0} |\n"
            f"Nr wins Player 1: {stat.nr_wins_player1} |\n"
            f"Win Rate plyer 0 {stat.win_rate_player0} |\n\n"
        )
