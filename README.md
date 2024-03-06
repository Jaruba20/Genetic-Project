# Genetic Optimization Algorithm (under development)

This library, inspired by Scikit_Learns's machine learning modules, provides a class `Optimization` for solving optimization problems using a genetic algorithm, particularly focused (for the moment) for the Traveling Salesman Problem (TSP).

## Features

- **Optimization Class**: The main class in this library. It takes a numpy array of cities as input.

- **Optimization.optimize()**: This method is used to run the optimization process. It has several parameters that can be adjusted according to the needs of the problem.

- **Grid Class**: Much like GridSearch in Scikit-Learn, this class is designed to iterate over different parameters of the Optimization class and obtain the best results (shortest route), returning also the parameters used in that iteration.
  
- **Grid.iterate()**: Method to apply the grid search, iterating over all the possible iterations of the parameters passed as a dictionary. 


## Usage

```python
from Optimization import Optimization
import numpy as np

# Define cities
cities = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Create Optimization object
opt = Optimization(cities)

# Run optimization
opt.optimize()

###########################
# Grid Search
from Grid import Grid
import numpy as np

# Define parameters to iterate over
dict_of_parameters = {
    'population_size': [20, 50],
    'generations': [100, 200],
    'mutation_rate': [0.01, 0.05]
}

# Create Grid object
grid = Grid(cities, dict_of_parameters)

# Run grid search
best_result, best_parameters = grid.iterate()
```

## Parameters
### **optimize()**
- population_size: The size of the population for each generation (default is 20).
- generations: The number of generations to run the optimization for (default is 100).
- show_distances: If set to True, the distances will be printed (default is False).
- plot_distances: If set to True, the evolution of distances through the generations will be plotted (default is False).
- plot_route: If set to True, the route will be plotted (default is False).
- selection_method: The method used for selection (default is "tournament").
- tournament_size: The size of the tournament for selection (default is 3).
- crossover_method: The method used for crossover (default is "order").
- mutation_method: The method used for mutation (default is "swap").
- mutation_rate: The rate of mutation (default is 0.01).
- mutation_rate_increase: If set to True, the mutation rate

