import matplotlib.pyplot as plt
import numpy as np
import random

class Optimization:
    def __init__(self, file_path: str = None, cities_array: np.array = None, ROUTE_MIN_DISTANCE: float = 0) -> None:
        if file_path is not None:
            self.cities = self.file_loader(file_path)
        elif cities_array is not None:
            self.cities = cities_array
        else:
            raise ValueError("Either file_path or cities must be provided.")

        self.number_of_cities = self.cities.shape[0]
        self.distance_matrix = self.get_distance_matrix()
        self.ROUTE_MIN_DISTANCE = ROUTE_MIN_DISTANCE

    def optimize(
        self,
        population_size: int = 20,
        generations: int = 100,
        show_distances: bool = False,
        save_distances_at: str = None,
        plot_distances: bool = False,
        save_distances_plot_at: str = None,
        plot_route: bool = False,
        save_route_plot_at: str = None,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        crossover_method: str = "order",
        mutation_method: str = "swap",
        mutation_rate: float = 0.01,
        mutation_rate_increase: bool = False,
        number_of_generations_to_increase: int = 30,
        mutation_rate_increase_by: float = 0.01,
    ) -> list:
        """Optimizes the TSP problem using a genetic algorithm. The algorithm runs for a number of generations and returns the best route found."""
        population = self.generate_population(population_size, self.number_of_cities)
        shortest_distance_by_generation = []
        last_increase_generation = 0
        for i in range(generations):
            print(f"Generation {i+1}/{generations}")

            population = self.evolve_population(
                population,
                selection_method,
                crossover_method,
                tournament_size,
                mutation_method,
                mutation_rate,
            )

            shortest_distance_by_generation.append(self.show_distances(population))

            if shortest_distance_by_generation[-1] < self.ROUTE_MIN_DISTANCE:
                print(f"Optimal route reached at generation {i+1}")
                break

            if show_distances:
                print(f"Distance: {shortest_distance_by_generation[-1]}")

            if mutation_rate_increase:
                if (
                    i - last_increase_generation >= number_of_generations_to_increase
                    and mutation_rate <= 0.2
                ):
                    if (
                        shortest_distance_by_generation[
                            -number_of_generations_to_increase
                        ]
                        == shortest_distance_by_generation[-1]
                    ):
                        mutation_rate += mutation_rate_increase_by
                        last_increase_generation = i
                        print(f"Mutation rate increased to {mutation_rate}")

        if save_distances_at is not None:
            with open(save_distances_at, "a") as file:
                i = 1
                for distance in shortest_distance_by_generation:
                    file.write(f"Generation {i}: {distance}\n")
                    i += 1

        if plot_distances or save_distances_plot_at is not None:
            self.plot_distances(
                shortest_distance_by_generation, plot_distances, save_distances_plot_at
            )

        if plot_route or save_route_plot_at is not None:
            self.plot_route(population[0], plot_route, save_route_plot_at)

        return population[0]

    def evolve_population(
        self,
        population: list,
        selection_method: str,
        crossover_method: str,
        tournament_size: int,
        mutation_method: str,
        mutation_rate: float,
    ) -> list:
        """Evolves a population of routes for the TSP problem. The evolution process consists of selection, crossover and mutation."""
        new_population = []
        for _ in range(len(population)):
            parents = self.selection(population, selection_method, tournament_size)
            offspring = self.crossover(parents, crossover_method)
            offspring = self.mutation(offspring, mutation_method, mutation_rate)
            new_population.append(offspring)

        return new_population

    def selection(self, population, method, tournament_size) -> list:
        """Selects two parents from the population, based on a given algorithm"""
        parents = []

        if method == "roulette":
            parents = self.roulette_selection(population)

        if method == "rank":
            parents = self.rank_selection(population)

        if method == "tournament":
            parents = self.tournament_selection(population, tournament_size)

        return parents

    def roulette_selection(self, population) -> list:
        """Selects two parents from the population using the roulette selection method."""
        parents = []
        offset = 0
        normalized_fitness_sum = sum(
            self.get_route_distance(individual) for individual in population
        )

        sorted_population = sorted(population, key=lambda x: self.get_route_distance(x))
        lowest_fitness = self.get_route_distance(sorted_population[0])
        if lowest_fitness < 0:
            offset = abs(lowest_fitness)
            normalized_fitness_sum += offset * len(population)

        draw = random.uniform(0, 1)

        accumulated = 0
        for _ in range(2):
            for individual in sorted_population:
                fitness = self.get_route_distance(individual) + offset
                accumulated += fitness / normalized_fitness_sum
                if draw <= accumulated:
                    parents.append(individual)
                    break
        return parents

    def rank_selection(self, population) -> list:
        """Selects two parents from the population using the rank selection method."""
        population = self.sort_population(population)
        ranks = np.arange(len(population)) + 1
        probabilities = [rank / sum(ranks) for rank in ranks]
        indices = np.arange(len(population))
        selected_indices = np.random.choice(indices, 2, p=probabilities)
        parents = [population[i] for i in selected_indices]
        return parents

    def tournament_selection(self, population, tournament_size) -> list:
        """Selects two parents from the population using the tournament selection method."""
        parents = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = min(tournament, key=lambda x: self.get_route_distance(x))
            parents.append(winner)
        return parents

    def crossover(self, parents, method) -> list:
        """Crossover the parents to generate the offspring. The crossover method is defined by the method parameter."""
        parent1 = parents[0]
        parent2 = parents[1]

        if method == "order":
            offspring = self.order_crossover(parent1, parent2)
        if method == "pmx":
            offspring = self.pmx_crossover(parent1, parent2)

        return offspring

    def mutation(
        self,
        offspring,
        mutation_method,
        mutation_rate,
    ) -> list:
        """Mutates an offspring for the TSP problem. The mutation rate is defined by the mutation_rate parameter."""
        if mutation_method == "swap":
            if random.random() <= mutation_rate:
                self.swap_mutation(offspring)

        if mutation_method == "inversion":
            if random.random() <= mutation_rate:
                self.inversion_mutation(offspring)

        if mutation_method == "scramble":
            if random.random() <= mutation_rate:
                self.scramble_mutation(offspring)

        return offspring

    def plot_route(self, route, plot_it, save_path) -> None:
        """Plots the route for the TSP problem."""
        x = [self.cities[route[i], 0] for i in range(self.number_of_cities)]
        y = [self.cities[route[i], 1] for i in range(self.number_of_cities)]
        plt.plot(x, y, marker="o", color="blue")
        plt.plot([x[-1], x[0]], [y[-1], y[0]], marker="o", color="blue")
        # Starting point
        plt.plot(x[0], y[0], marker="o", color="red")
        if save_path is not None:
            plt.savefig(save_path)
        if plot_it:
            plt.show()
        

    def plot_distances(self, distances, plot_it, save_path) -> None:
        """Plots the lowest distance found for each generation of the genetic algorithm."""
        plt.plot(distances)
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.title("Evolution of the lowest distance")
        if save_path is not None:
            plt.savefig(save_path)
        if plot_it:
            plt.show()
        

    def sort_population(self, population) -> list:
        """Sorts the population based on the total distance of each route (minimum distance first)."""
        return sorted(population, key=lambda x: (-1) * self.get_route_distance(x))

    def order_crossover(self, parent1, parent2) -> list:
        """Applies the order crossover method to the parents to generate the offspring."""
        offspring = [-1] * self.number_of_cities

        start, end = sorted(random.sample(range(self.number_of_cities), 2))

        offspring[start:end] = parent1[start:end]

        for i in range(self.number_of_cities):
            if offspring[i] == -1:
                for gene in parent2:
                    if gene not in offspring:
                        offspring[i] = gene
                        break

        return offspring

    def pmx_crossover(self, parent1, parent2) -> list:
        """Applies the partially mapped crossover method to the parents to generate the offspring."""
        offspring = [-1] * self.number_of_cities

        start, end = sorted(random.sample(range(self.number_of_cities), 2))

        offspring[start : end + 1] = parent1[start : end + 1]

        # Map the corresponding positions from parent2 to the child
        for i in range(start, end + 1):
            if parent2[i] not in offspring:
                j = i
                while offspring[j] != -1:
                    j = parent2.index(parent1[j])
                offspring[j] = parent2[i]

        # Fill in the remaining positions from parent2
        for i in range(self.number_of_cities):
            if offspring[i] == -1:
                offspring[i] = parent2[i]

        return offspring

    def scramble_mutation(self, route) -> None:
        """Applies the scramble mutation method to a route for the TSP problem."""
        start, end = sorted(random.sample(range(self.number_of_cities), 2))
        scrambled = route[start:end]
        random.shuffle(scrambled)
        route[start:end] = scrambled

    def inversion_mutation(self, route) -> None:
        """Applies the inversion mutation method to a route for the TSP problem."""
        start, end = sorted(random.sample(range(self.number_of_cities), 2))
        route[start:end] = route[start:end][::-1]

    def swap_mutation(self, route) -> None:
        """Applies the swap mutation method to a route for the TSP problem."""
        indexes = random.sample(range(self.number_of_cities), 2)
        route[indexes[0]], route[indexes[1]] = route[indexes[1]], route[indexes[0]]

    def show_distances(self, population) -> float:
        """Prints the total distance of the shortest route in the population."""
        distances = []
        for individual in population:
            distances.append(self.get_route_distance(individual))
        return np.min(distances)

    def get_route_distance(self, route) -> float:
        """Calculates the total distance of a route for the TSP problem."""
        distance = 0
        for i in range(self.number_of_cities - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        distance += self.distance_matrix[route[-1], route[0]]

        return distance

    def generate_population(self, population_size: int, number_of_cities: int) -> list:
        """Generates a population of routes for the TSP problem. Each route is a permutation of the cities."""
        population = []

        for _ in range(population_size):
            route = np.random.permutation(number_of_cities)
            population.append(route.tolist())

        return population

    def get_distance_matrix(self) -> np.ndarray:
        """Generates a distance matrix for the cities. The distance matrix is a symmetric matrix with the distances between each pair of cities."""
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))

        for i in range(self.number_of_cities):
            for j in range(i + 1, self.number_of_cities):
                distance = np.linalg.norm(self.cities[i] - self.cities[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def file_loader(self, file) -> np.array:
        """Loads the cities from a file and returns an array with the coordinates of the cities.
        The file format is standarized as follows (example with Berlin52.tsp):
        NAME: berlin52
        TYPE: TSP
        COMMENT: 52 locations in Berlin (Groetschel)
        DIMENSION: 52
        EDGE_WEIGHT_TYPE: EUC_2D
        NODE_COORD_SECTION
        1 565.0 575.0
        2 25.0 185.0
        3 345.0 750.0
        4 945.0 685.0
        5 845.0 655.0
        ...
        """
        cities = []
        with open(file, "r") as file:
            for linea in file:
                if linea[0].isdigit():
                    cities.append([float(x) for x in linea.split()[1:]])

        return np.array(cities)
