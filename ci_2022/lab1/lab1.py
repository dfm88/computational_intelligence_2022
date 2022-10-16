import random
import sys
sys.path.append(".")  # noqa

from dataclasses import dataclass, field  # noqa
from typing import Iterable  # noqa
import logging  # noqa

from ci_2022.utils.base_search import Search, State  # noqa


logging.basicConfig(format="%(message)s", level=logging.DEBUG)


def problem(N=5, seed=42):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]


class StateLab1(State):
    def __init__(self, data: set(), history=None):
        self._data = data.copy()
        if not history:
            self.history = []

    def __repr__(self):
        return f'<{self.__class__.__name__}>: {self._data}'

    def update_history(self, state):
        self.history.append(state.data)


@dataclass
class SearchLab1(Search):

    N: int = field(default=5)
    SEED: int = field(init=False, default=42)
    all_lists: list[State] = field(default_factory=list)

    def __post_init__(self):
        if not self.all_lists:
            self.all_lists = [
                StateLab1(data=set(el)) for el in sorted(problem(N=self.N, seed=42), key=lambda l: len(l))
            ]

    def _actions_from_state(self, *args, **kwargs) -> Iterable:
        yield from self.all_lists

    def _get_new_state(self, state: State, a: State, *args, **kwargs) -> State:
        # create a new state with the union of `state` and `a`
        # data and save in `new_state` history the `state` data
        new_state = StateLab1(state.data)
        new_state.update_history(state=a)
        new_state._data |= a.data
        return new_state

    def _priority_function(self, state: State, *args, **kwargs) -> int:
        return self.state_cost[state] + self.h(state=state)

    def _unit_cost(self, a: State, *args, **kwargs) -> int:
        return len(a.data)

    def h(self, state: State) -> int:
        """Heuristic calculate the difference between the len
        of elements in the current state and the len of elements
        of the goal state

        Args:
            state (State): _description_

        Returns:
            int: _description_
        """
        return len(self.goal.data) - len(state.data)

    def search(self):

        self.parent_state.clear()
        self.state_cost.clear()

        state = self.initial_state
        self.parent_state[state] = None
        self.state_cost[state] = 0

        while state is not None and not self._goal_test(state):

            for a in self._actions_from_state(state):  #
                new_state = self._get_new_state(state, a)  #
                cost = self._unit_cost(a)
                if new_state not in self.state_cost and new_state not in self.frontier:
                    self.parent_state[new_state] = state
                    self.state_cost[new_state] = self.state_cost[state] + cost
                    self.frontier.push(new_state, p=self._priority_function(new_state))
                    logging.debug(f"Added new node to frontier (cost={self.state_cost[new_state]})")
                elif new_state in self.frontier and self.state_cost[new_state] > self.state_cost[state] + cost:
                    old_cost = self.state_cost[new_state]
                    self.parent_state[new_state] = state
                    self.state_cost[new_state] = self.state_cost[state] + cost
                    logging.debug(f"Updated node cost in frontier: {old_cost} -> {self.state_cost[new_state]}")
                else:
                    logging.debug(f'skipping state {new_state}')
            if self.frontier:
                state = self.frontier.pop()
            else:
                state = None

        path = list()
        s = state

        while s:
            if s.history:
                path.append(*s.history)
            s = self.parent_state[s]

        weight = sum(len(_) for _ in path)
        bloat = ((sum(len(_) for _ in path)-self.N)/self.N*100)

        print(
            f"Found a solution in {len(path)} steps; \
            weight: {weight}; \
            visited {len(self.state_cost)} states; \
            bloat={bloat:.0f}%"
        )
        print(path)


if __name__ == '__main__':
    N_ = 5
    goal_state = StateLab1(set(range(N_)))
    all_lists = [StateLab1(data=set(el)) for el in sorted(problem(N=N_, seed=42), key=lambda l: len(l))]

    initial_state = StateLab1(
        data=set(),
    )

    search_lab1 = SearchLab1(
        N=N_,
        initial_state=initial_state,
        goal=goal_state,
        all_lists=all_lists
    )

    search_lab1.search()
