## Lab 03 - Adversarial Search

- [Set Up Environment](#set-up-environment)
- [Run The Script](#run-the-script)
- [Task 1 - hard coded strategies](#task-1---hard-coded-strategies)
- [Task 2 - Evolved Strategy](#task-2---evolved-strategy)

### [Set Up Environment](#set-up-environment)

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

### [Run The Script](#run-the-script)

To run the solution run the files `lab3/lab3_task*.py`.

In general the implementations doesn't accept condition where the upper bound `K` is None

In the `lab3/utils.py` module there are the following classes:

* `Nimply` named tuple to handle row index and the amount of objects to remove from a give row index
* `Statistics` named tuple as a wrapper to log statistics
* `Nim` is the class that represent the board. the attribute `K` is the upper bound to the nr. of rows that can be removed from a row. The method `nimming` is the one that actually does the operation of removing objects from a row as indicated by its `Nimply` argument.
* `BaseStrategies(AbstractClass)` is an interface with the `K` attribute that is the upper bound and with a utility function that return all `strategy` methods that must follow the naming convention of having the `strategy` word at the end of the function name. By default for all strategies if no available row respect the policy of the given strategy, a random row is selected and a k nr. of elements are removed.
* `Player0Strategies(BaseStrategies)` is a wrapper class, subclass of BaseStrategies that has all the hardcoded strategy methods that can be used by the human player.
* `Player1Strategies(BaseStrategies)` is a wrapper class, subclass of BaseStrategies that has all the hardcoded strategy methods that can be used by the AI player.

### [Task 1 - hard coded strategies](#task-1---hard-coded-strategies)

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

### [Task 2 - Evolved Strategy](#task-2---evolved-strategy)

It was use a Genetic Algorithm approach to evolve the strategies. The evolving part involves the probability of choosing a given strategy between the hardcoded ones.

So the Genome of an individual in the population is an array of probability that corresponds to the probability of choosing a given strategy. 

For example:

```python
strategies = [
    'take_k_from_even_row_strategy',
    'take_k_from_longest_row_strategy',
    'take_k_from_odd_row_strategy',
    'take_k_from_one_andbitwise_row_strategy',
    'take_k_from_one_xorbitwise_row_strategy',
    'take_k_from_shortest_row_strategy',
    'take_k_from_zero_andbitwise_row_strategy',
    'take_k_from_zero_xorbitwise_row_strategy'
]

genome = [0.0312, 0.240, 0.136, 0.007, 0.182, 0.063, 0.253, 0.084]
```

In this example we have `3%` of probability of choosing the `take_k_from_even_row_strategy` strategy, the `24%` probability of choosing `take_k_from_longest_row_strategy` strategy ecc...

So when it is time to evaluate the fitness (hence the winning rate), a random strategy is selected based on it probability, for example:

```python
prob = random.random()  # prob = 0.518
```

At this point the values of probability of the genome are accumulated and will be choose the strategy where the value `prob = 0.518` falls:

```python
genome_accum = accumulate(genome, operator.add)

genome_accum == [0.0312, 0.27, 0.40, 0.41, 0.59, 0.66, 0.9, 1.00]
```

In this case `0.518` falls on index 4 so will be picked the strategy with index 4, that is `take_k_from_one_xorbitwise_row_strategy`.

These are the results of playing against the NimSum.

```python
[
    [('take_k_from_even_row_strategy', 0.23585659416833787), ('take_k_from_longest_row_strategy', 0.04787993365142319), ('take_k_from_odd_row_strategy', 0.04739430349054491), ('take_k_from_one_andbitwise_row_strategy', 0.07527270667651094), ('take_k_from_one_xorbitwise_row_strategy', 0.11871579309568914), ('take_k_from_shortest_row_strategy', 0.031163622979142467), ('take_k_from_zero_andbitwise_row_strategy', 0.1945110509413592), ('take_k_from_zero_xorbitwise_row_strategy', 0.24920599499699228)],
    fitness: 0.8,

    strategies  : [('take_k_from_even_row_strategy', 0.2613449613414318), ('take_k_from_longest_row_strategy', 0.04244334809729503), ('take_k_from_odd_row_strategy', 0.026888230020002978), ('take_k_from_one_andbitwise_row_strategy', 0.10425902382443615), ('take_k_from_one_xorbitwise_row_strategy', 0.16443134898671877), ('take_k_from_shortest_row_strategy', 0.014144057006125211), ('take_k_from_zero_andbitwise_row_strategy', 0.11035203587678359), ('take_k_from_zero_xorbitwise_row_strategy', 0.2761369948472065)],
    fitness: 0.75,

    strategies  : [('take_k_from_even_row_strategy', 0.3231683685465943), ('take_k_from_longest_row_strategy', 0.03358956068449545), ('take_k_from_odd_row_strategy', 0.041561091249228736), ('take_k_from_one_andbitwise_row_strategy', 0.12892239614134107), ('take_k_from_one_xorbitwise_row_strategy', 0.13013056280484242), ('take_k_from_shortest_row_strategy', 0.014030313954224468), ('take_k_from_zero_andbitwise_row_strategy', 0.10946461175662837), ('take_k_from_zero_xorbitwise_row_strategy', 0.21913309486264512)],
    fitness: 0.75,

    strategies  : [('take_k_from_even_row_strategy', 0.15944593470600563), ('take_k_from_longest_row_strategy', 0.032368231219619424), ('take_k_from_odd_row_strategy', 0.03203993107098165), ('take_k_from_one_andbitwise_row_strategy', 0.09938778221928939), ('take_k_from_one_xorbitwise_row_strategy', 0.1253989650287567), ('take_k_from_shortest_row_strategy', 0.016854014236086068), ('take_k_from_zero_andbitwise_row_strategy', 0.20546115928351516), ('take_k_from_zero_xorbitwise_row_strategy', 0.329043982235746)],
    fitness: 0.75,

    strategies  : [('take_k_from_even_row_strategy', 0.18849948587875304), ('take_k_from_longest_row_strategy', 0.00028651643185284993), ('take_k_from_odd_row_strategy', 0.03924132662341958), ('take_k_from_one_andbitwise_row_strategy', 0.12172649235118761), ('take_k_from_one_xorbitwise_row_strategy', 0.15358403031612528), ('take_k_from_shortest_row_strategy', 0.013210992267643544), ('take_k_from_zero_andbitwise_row_strategy', 0.16105040310123397), ('take_k_from_zero_xorbitwise_row_strategy', 0.3224007530297841)],
    fitness: 0.75,

    strategies  : [('take_k_from_even_row_strategy', 0.25175515729629544), ('take_k_from_longest_row_strategy', 0.015487309106350874), ('take_k_from_odd_row_strategy', 0.1043641151823068), ('take_k_from_one_andbitwise_row_strategy', 0.24967343460418515), ('take_k_from_one_xorbitwise_row_strategy', 0.054133693367668025), ('take_k_from_shortest_row_strategy', 0.14270883115602964), ('take_k_from_zero_andbitwise_row_strategy', 0.061077805465076275), ('take_k_from_zero_xorbitwise_row_strategy', 0.12079965382208772)],
    fitness: 0.75,

    strategies  : [('take_k_from_even_row_strategy', 0.281730175895327), ('take_k_from_longest_row_strategy', 0.017331292695395902), ('take_k_from_odd_row_strategy', 0.09343211346528113), ('take_k_from_one_andbitwise_row_strategy', 0.17881638055728538), ('take_k_from_one_xorbitwise_row_strategy', 0.06057907658426837), ('take_k_from_shortest_row_strategy', 0.10220821731129512), ('take_k_from_zero_andbitwise_row_strategy', 0.054679986894488025), ('take_k_from_zero_xorbitwise_row_strategy', 0.21122275659665912)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.185782371271217), ('take_k_from_longest_row_strategy', 0.037714644534047395), ('take_k_from_odd_row_strategy', 0.029865694001963723), ('take_k_from_one_andbitwise_row_strategy', 0.1158041306611714), ('take_k_from_one_xorbitwise_row_strategy', 0.11688936250876437), ('take_k_from_shortest_row_strategy', 0.015710296965512942), ('take_k_from_zero_andbitwise_row_strategy', 0.19151851790367125), ('take_k_from_zero_xorbitwise_row_strategy', 0.306714982153652)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.3021891750058105), ('take_k_from_longest_row_strategy', 0.018589876021487223), ('take_k_from_odd_row_strategy', 0.08017365750018231), ('take_k_from_one_andbitwise_row_strategy', 0.19180186980825084), ('take_k_from_one_xorbitwise_row_strategy', 0.06497827617306903), ('take_k_from_shortest_row_strategy', 0.08770439096901064), ('take_k_from_zero_andbitwise_row_strategy', 0.07331349968308427), ('take_k_from_zero_xorbitwise_row_strategy', 0.1812492548391051)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.17484003683846552), ('take_k_from_longest_row_strategy', 0.0002657542712461289), ('take_k_from_odd_row_strategy', 0.03639773848954714), ('take_k_from_one_andbitwise_row_strategy', 0.14113211176467283), ('take_k_from_one_xorbitwise_row_strategy', 0.14245469897750038), ('take_k_from_shortest_row_strategy', 0.01914635837714897), ('take_k_from_zero_andbitwise_row_strategy', 0.1867250345524904), ('take_k_from_zero_xorbitwise_row_strategy', 0.2990382667289287)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.33327798048398505), ('take_k_from_longest_row_strategy', 0.013121521166907013), ('take_k_from_odd_row_strategy', 0.08842181279048691), ('take_k_from_one_andbitwise_row_strategy', 0.13538187622296813), ('take_k_from_one_xorbitwise_row_strategy', 0.07166315159328489), ('take_k_from_shortest_row_strategy', 0.07738183818442257), ('take_k_from_zero_andbitwise_row_strategy', 0.08085589140021907), ('take_k_from_zero_xorbitwise_row_strategy', 0.1998959281577264)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.18120375157029026), ('take_k_from_longest_row_strategy', 0.000275427023560502), ('take_k_from_odd_row_strategy', 0.03772251986582329), ('take_k_from_one_andbitwise_row_strategy', 0.11701515776926401), ('take_k_from_one_xorbitwise_row_strategy', 0.14763967309952217), ('take_k_from_shortest_row_strategy', 0.012699670504156721), ('take_k_from_zero_andbitwise_row_strategy', 0.19352133175460096), ('take_k_from_zero_xorbitwise_row_strategy', 0.3099224684127821)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.4779730150127842), ('take_k_from_longest_row_strategy', 0.018818324044669328), ('take_k_from_odd_row_strategy', 0.10144862349679502), ('take_k_from_one_andbitwise_row_strategy', 0.05089758869268805), ('take_k_from_one_xorbitwise_row_strategy', 0.08222097981419219), ('take_k_from_shortest_row_strategy', 0.029092143513421712), ('take_k_from_zero_andbitwise_row_strategy', 0.09276804699305011), ('take_k_from_zero_xorbitwise_row_strategy', 0.14678127843239952)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.4859531977990506), ('take_k_from_longest_row_strategy', 0.02391564078800879), ('take_k_from_odd_row_strategy', 0.16115999704783557), ('take_k_from_one_andbitwise_row_strategy', 0.06468421126454965), ('take_k_from_one_xorbitwise_row_strategy', 0.04335311513078959), ('take_k_from_shortest_row_strategy', 0.01783070308895383), ('take_k_from_zero_andbitwise_row_strategy', 0.08903377631482787), ('take_k_from_zero_xorbitwise_row_strategy', 0.11406935856598417)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.15441380539122534), ('take_k_from_longest_row_strategy', 0.02006187947028539), ('take_k_from_odd_row_strategy', 0.024822998166821184), ('take_k_from_one_andbitwise_row_strategy', 0.09625109407883066), ('take_k_from_one_xorbitwise_row_strategy', 0.12144136141140623), ('take_k_from_shortest_row_strategy', 0.025503281087788286), ('take_k_from_zero_andbitwise_row_strategy', 0.15918142797962614), ('take_k_from_zero_xorbitwise_row_strategy', 0.39832415241401686)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.3094203336398717), ('take_k_from_longest_row_strategy', 0.0004703143328159451), ('take_k_from_odd_row_strategy', 0.04122515859711051), ('take_k_from_one_andbitwise_row_strategy', 0.15985041741850686), ('take_k_from_one_xorbitwise_row_strategy', 0.12907873515136425), ('take_k_from_shortest_row_strategy', 0.017348586882819872), ('take_k_from_zero_andbitwise_row_strategy', 0.16919225167108748), ('take_k_from_zero_xorbitwise_row_strategy', 0.17341420230642343)],
    fitness: 0.7,

    strategies  : [('take_k_from_even_row_strategy', 0.2525225587315512), ('take_k_from_longest_row_strategy', 0.05126319830144223), ('take_k_from_odd_row_strategy', 0.025980545442936917), ('take_k_from_one_andbitwise_row_strategy', 0.12592434981272085), ('take_k_from_one_xorbitwise_row_strategy', 0.15888052621592355), ('take_k_from_shortest_row_strategy', 0.010963235032753295), ('take_k_from_zero_andbitwise_row_strategy', 0.10691904956407493), ('take_k_from_zero_xorbitwise_row_strategy', 0.2675465368985969)],
    fitness: 0.65,

    strategies  : [('take_k_from_even_row_strategy', 0.08923685472252718), ('take_k_from_longest_row_strategy', 0.22531498022160873), ('take_k_from_odd_row_strategy', 0.1248644555614997), ('take_k_from_one_andbitwise_row_strategy', 0.00404437351928876), ('take_k_from_one_xorbitwise_row_strategy', 0.06791869291663964), ('take_k_from_shortest_row_strategy', 0.17235628511866014), ('take_k_from_zero_andbitwise_row_strategy', 0.15259939303566533), ('take_k_from_zero_xorbitwise_row_strategy', 0.16366496490411048)],
    fitness: 0.65,

    strategies  : [('take_k_from_even_row_strategy', 0.19369602393596097), ('take_k_from_longest_row_strategy', 0.03985108818786997), ('take_k_from_odd_row_strategy', 0.008619204744956916), ('take_k_from_one_andbitwise_row_strategy', 0.21061782019057168), ('take_k_from_one_xorbitwise_row_strategy', 0.22309808309492496), ('take_k_from_shortest_row_strategy', 0.09917331948112086), ('take_k_from_zero_andbitwise_row_strategy', 0.032487385266369975), ('take_k_from_zero_xorbitwise_row_strategy', 0.19245707509822468)],
    fitness: 0.65,

    strategies  : [('take_k_from_even_row_strategy', 0.34889124221903006), ('take_k_from_longest_row_strategy', 0.05666117443241633), ('take_k_from_odd_row_strategy', 0.03589534661238041), ('take_k_from_one_andbitwise_row_strategy', 0.0034554703335234094), ('take_k_from_one_xorbitwise_row_strategy', 0.0725362739574581), ('take_k_from_shortest_row_strategy', 0.23009281721236496), ('take_k_from_zero_andbitwise_row_strategy', 0.16297415194020928), ('take_k_from_zero_xorbitwise_row_strategy', 0.08949352329261759)],
    fitness: 0.65
]
```

