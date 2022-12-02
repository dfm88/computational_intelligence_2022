# Copyright © 2022 Giovanni Squillero <squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free for personal or classroom use; see 'LICENSE.md' for details.

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from utils.gx_utils import PriorityQueue

sys.path.append(".")


logging.basicConfig(format="%(message)s", level=logging.DEBUG)


class State:
    def __init__(self, data: np.ndarray):
        self._data = data.copy()
        self._data.flags.writeable = False

    def __hash__(self):
        return hash(bytes(self._data))

    def __eq__(self, other):
        return bytes(self._data) == bytes(other._data)

    def __lt__(self, other):
        return bytes(self._data) < bytes(other._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    @property
    def data(self):
        return self._data

    def copy_data(self):
        return self._data.copy()


@dataclass
class Search(ABC):
    initial_state: State
    goal: State
    frontier: PriorityQueue = field(default_factory=PriorityQueue)
    parent_state: dict[State, State] = field(default_factory=dict)
    state_cost: dict[State, int] = field(default_factory=dict)

    def _goal_test(self, state: State) -> bool:
        """Checks whther the final state was reached

        Args:
            state (State): state to check

        Returns:
            bool: True if final check was reached
        """
        return state == self.goal

    @abstractmethod
    def _actions_from_state(self, state: State) -> Iterable:
        """Return an iterable with all possible actions from state

        Args:
            state (State): current state

        Returns:
            Iterable: actions
        """
        pass

    @abstractmethod
    def _get_new_state(self, state: State, *args, **kwargs) -> State:
        """Evaluate a new state given the current

        Args:
            state (State): current state

        Returns:
            State: new state
        """
        pass

    @abstractmethod
    def _priority_function(self, *args, **kwargs) -> int:
        """Evaluate the priority, and return its value, the bigget the less prior

        Returns:
            int: priority level
        """
        pass

    @abstractmethod
    def _unit_cost(self, *args, **kwargs) -> int:
        """Evaluate the cost

        Returns:
            int: _description_
        """
        pass

    def search(
        self,
        initial_state: State,
        parent_state: dict,
        state_cost: dict,
    ):

        state = initial_state
        parent_state[state] = None
        state_cost[state] = 0

        while state is not None and not self._goal_test(state):
            for a in self._actions_from_state(state):
                new_state = self._get_new_state(state, a)
                cost = self._unit_cost(a)
                if new_state not in state_cost and new_state not in self.frontier:
                    parent_state[new_state] = state
                    state_cost[new_state] = state_cost[state] + cost
                    self.frontier.push(new_state, p=self._priority_function(new_state))
                    logging.debug(
                        f"Added new node to frontier (cost={state_cost[new_state]})"
                    )
                elif (
                    new_state in self.frontier
                    and state_cost[new_state] > state_cost[state] + cost
                ):
                    old_cost = state_cost[new_state]
                    parent_state[new_state] = state
                    state_cost[new_state] = state_cost[state] + cost
                    logging.debug(
                        f"Updated node cost in frontier: {old_cost} -> {state_cost[new_state]}"
                    )
            if self.frontier:
                state = self.frontier.pop()
            else:
                state = None

        path = list()
        s = state
        while s:
            path.append(s.copy_data())
            s = self.parent_state[s]

        logging.info(
            f"Found a solution in {len(path):,} steps; visited {len(self.state_cost):,} states"
        )
        res = list(reversed(path))
        logging.info(f"{res}")
        return res
