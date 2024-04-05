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

# Create Optimization object with a 2 column array of cities (x and y coordinates). 
# Optionally, add the shortest distance if known.
opt = Optimization(cities_array=cities, ROUTE_MIN_DISTANCE=22)

# Or add the path to the .tsp file
opt = Optimization(file_path="YOUR_PATH")

# Run optimization
opt.optimize()

###########################
# Grid Search
from Optimization import Optimization
from Grid import Grid
import numpy as np

opt = Optimization(cities)

# Define parameters to iterate over
dict_of_parameters = {
    'population_size': [20, 50],
    'generations': [100, 200],
    'mutation_rate': [0.01, 0.05]
}

# Create Grid object
grid = Grid(opt, dict_of_parameters)

# Run grid search
best_result, best_parameters = grid.iterate()
```

## Parameters
### **optimize()**
- population_size: The size of the population for each generation (default is 20).
- generations: The number of generations to run the optimization for (default is 100).
- show_distances: If set to True, the distances will be printed (default is False).
- save_distances_at: Creates a .txt file with the distances by generation at the given path.
- plot_distances: If set to True, the evolution of distances through the generations will be plotted (default is False).
- save_distances_plot_at: Creates an image file with the evolution of distances through generations at the path given. 
- plot_route: If set to True, the route will be plotted (default is False).
- save_route_plot_at: Creates an image file with the route in 2D, at the given path.
- selection_method: The method used for selection (default is "tournament").
- tournament_size: The size of the tournament for selection (default is 3).
- crossover_method: The method used for crossover (default is "order").
- mutation_method: The method used for mutation (default is "swap").
- mutation_rate: The rate of mutation (default is 0.01).
- mutation_rate_increase: If set to True, the mutation rate will increase when the shortest distance doesn't vary after "number_of_generations_to_increase".
- number_of_generations_to_increase: Number of generations without change in the shortest distance of each generation for the mutation rate to increase.
- mutation_rate_increase_by: Amount that the mutation rate increases.

