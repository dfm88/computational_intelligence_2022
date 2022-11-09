## Lab 02 - Set Coverage using Genetic Algorithm

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

To run the solution run the file `lab2/lab2.py`.

The problem was solved with the following hyper-parameters (**TODO** tune hyper-parameters depending on problem size):

* `PROBLEM_SIZE = nr. of generated sub-lists`
* `POPULATION_SIZE = 600`
* `OFFSPRING_SIZE = 200`
* `NUM_GENERATIONS = 1000`
* `mutation rate = 80%`
* `mutation_behavior = 1 element if genome is valid (at least one True value) else mutate 30% of genes`
* `crossover_rate = 20%`
* `crossover_behavior = cut in a random place the 2 genomes and switch the trimmed part`

The population is composed by a list of Individuals; an Individual if modelled by an `Individual` class with the following attributes`:

* `genome` a random tuple of boolean values that represent (where gene is True) the element chosen as solution from problem list.
* `fitness` a tuple of 2 integer where first value is the 'completeness' of the solution (how mny single values are present from 0 to N) and the second part is the `weight` hence the sum of lengths of all sub-lists;
in this way, when evaluates fitness, it takes for the same level of completeness, the lower weight

Here the results:


**N=5**
```py
completeness = 5
weight = -5
```


**N=10**
```py
completeness = 10
weight = -10
```

**N=20**
```py
completeness = 20
weight = -24
```

**N=100**
```py
completeness = 100
weight = -207
```

**N=500**
```py
completeness = 500
weight = -1583
```

**N=1000**
```py
completeness = 1000
weight = -3973
```
