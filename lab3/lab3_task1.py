import logging
import sys

sys.path.append(".")

from lab3.utils import (Player0Strategies, Player1Strategies, Statistics,
                        evaluate, play_nim)

if __name__ == "__main__":

    NUM_MATCHES = 20
    NIM_SIZE = 10
    K = 3

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0Strategies(k=K)

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    play_nim(
        num_matches=NUM_MATCHES,
        nim_size=NIM_SIZE,
        k=K,
        p0_strategies_list=p0_strategies.get_all_strategies(),
        p1_strategy=p1_strategy,
    )
