## Lab 01

### Setup environment

* Using **Poetry**
From the root execute the following commands
```shell
poetry shell && poetry install
```

* Using **pip**
From the root, after creating and activating the virtual environment, execute the following comand
```shell
pip install .
```

### Run the script

To run the solution run the file `ci_2022/lab1/lab1.py`.

I decided to implement the **A\*** strategy to approach the problem.
The `lab1.py` file contains tow main classes `StateLab1` and `SearchLab1`.

`StateLab1` extends the basic State class of Prof. Squillero, basically changing only the accepted data type (set) and adding a `update_history` method that basically update a State history to keep track of the State that build the final solution.

`SearchLab1` extends the basic Search class of Prof. Squillero defining custuom attributes and behaviours related to the lab.

The **heuristic** used is the difference in terms of length between the length of the final goal and the length of the current state data.

Following the reached results (could not compute in reasonable time for N = 100, 500, 1000)


**N=5**
```py
Found a solution in 3 steps;             weight: 5;             visited 21 states;             bloat=0%
[{4}, {2, 3}, {0, 1}]
```


**N=10**
```py
Found a solution in 4 steps;             weight: 12;             visited 776 states;             bloat=20%
[{0, 9, 4, 5}, {4, 5, 6}, {8, 2, 7}, {0, 1}]
```

**N=20**
```py
Found a solution in 5 steps;             weight: 26;             visited 15312 states;             bloat=30%
[{2, 18, 6, 8, 10, 12, 15}, {8, 4, 7}, {2, 18, 6, 8, 10, 12, 15}, {1, 3, 13, 14}, {0, 16, 17, 5, 11}]
```
