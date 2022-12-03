## Lab 03 - Adversarial Search

### Setup environment

* Using **Poetry**

From the root execute the following commands
```shell
poetry shell
poetry install
```

* Using **pip**

From the root, after creating and activating the virtual environment, execute the following comand
```shell
pip install .
```

### Run the script

To run the solution run the files `lab3/lab3_task*.py`.

In general the implementations doesn't accept condition where the upper bound `K` is None

In the `lab3/utils.py` module there are the following classes:

* `Nimply` named tuple to handle row index and the amount of objects to remove from a give row index
* `Statistics` named tuple as a wrapper to log statistics
* `Nim` is the class that represent the board. the attribute `K` is the upper bound to the nr. of rows that can be removed from a row. The method `nimming` is the one that actually does the operation of removing objects from a row as indicated by its `Nimply` argument.
* `BaseStrategies(AbstractClass)` is an interface with the `K` attribute that is the upper bound and with a utility function that return all `strategy` methods that must follow the naming convention of having the `strategy` word at the end of the function name. By default for all strategies if no available row respect the policy of the given strategy, a random row is selected and a k nr. of elements are removed.
* `Player0Strategies(BaseStrategies)` is a wrapper class, subclass of BaseStrategies that has all the hardcoded strategy methods that can be used by the human player.
* `Player1Strategies(BaseStrategies)` is a wrapper class, subclass of BaseStrategies that has all the hardcoded strategy methods that can be used by the AI player.

### Task 1 - hard coded strategies

These are some result of hardcoded strategies, playing VS the random and the nimsum strategy:

VS Random
```
NUM MATCHES=20, K=3, using 'take_k_from_even_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 2 |
Nr wins Player 1: 18 |
Win Rate plyer 0 0.1 |

NUM MATCHES=20, K=3, using 'take_k_from_longest_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_odd_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_one_andbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 2 |
Nr wins Player 1: 18 |
Win Rate plyer 0 0.1 |

NUM MATCHES=20, K=3, using 'take_k_from_one_xorbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_shortest_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 3 |
Nr wins Player 1: 17 |
Win Rate plyer 0 0.15 |

NUM MATCHES=20, K=3, using 'take_k_from_zero_andbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_zero_xorbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 10 |
Nr wins Player 1: 10 |
Win Rate plyer 0 0.5 |
```

VS NimSum (Optimal)
```
NUM MATCHES=20, K=3, using 'take_k_from_even_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_longest_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_odd_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 0 |
Nr wins Player 1: 20 |
Win Rate plyer 0 0.0 |

NUM MATCHES=20, K=3, using 'take_k_from_one_andbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 2 |
Nr wins Player 1: 18 |
Win Rate plyer 0 0.1 |

NUM MATCHES=20, K=3, using 'take_k_from_one_xorbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 1 |
Nr wins Player 1: 19 |
Win Rate plyer 0 0.05 |

NUM MATCHES=20, K=3, using 'take_k_from_shortest_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 4 |
Nr wins Player 1: 16 |
Win Rate plyer 0 0.2 |

NUM MATCHES=20, K=3, using 'take_k_from_zero_andbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 3 |
Nr wins Player 1: 17 |
Win Rate plyer 0 0.15 |

NUM MATCHES=20, K=3, using 'take_k_from_zero_xorbitwise_row_strategy' vs 'optimal_strategy':
Nr wins Player 0: 9 |
Nr wins Player 1: 11 |
Win Rate plyer 0 0.45 |
```

