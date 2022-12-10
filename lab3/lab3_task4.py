import functools
import logging
import sys

sys.path.append(".")

from lab3.utils import (
    Player0ReinforcementLearning,
    Player1Strategies,
    RLAgent,
    Statistics,
    play_nim,
)

logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":

    NUM_MATCHES = 50
    NIM_SIZE = 6
    K = 3

    EPISODES = 5000

    p1_strategies = Player1Strategies(k=K)
    p0_strategies = Player0ReinforcementLearning(
        k=K, nim_size=NIM_SIZE, rl_agent=RLAgent()
    )

    p1_strategy = p1_strategies.pure_random_strategy
    p1_strategy = p1_strategies.optimal_strategy

    p0_strategy = functools.partial(
        p0_strategies.reinforcement_learning_strategy, episodes=EPISODES
    )
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
