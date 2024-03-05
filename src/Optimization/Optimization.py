import numpy as np
import random
import matplotlib.pyplot as plt


class Optimization:
    def __init__(self, cities):
        # self.generations = generations
        # self.population_size = population_size
        self.cities = cities
        self.number_of_cities = cities.shape[0]
        # self.population = self.generate_population(population_size, self.number_of_cities)
        self.distance_matrix = self.get_distance_matrix()

    def optimize(
        self,
        population_size: int = 20,
        generations: int = 100,
        show_distances: bool = False,
        plot_distances: bool = False,
        plot_route: bool = False,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        crossover_method: str = "order",
        mutation_method: str = "swap",
        mutation_rate: float = 0.01,
        mutation_rate_increase: bool = False,
        number_of_generations_to_increase: int = 30,
        mutation_rate_increase_by: float = 0.01,
    ):
        """Optimizes the TSP problem using a genetic algorithm. The algorithm runs for a number of generations and returns the best route found."""
        population = self.generate_population(population_size, self.number_of_cities)
        lowest_distance_by_generation = []
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

            lowest_distance_by_generation.append(self.show_distances(population))

            if lowest_distance_by_generation[-1] < 7543:
                break

            if show_distances:
                print(f"Distance: {lowest_distance_by_generation[-1]}")

            if mutation_rate_increase:
                if (
                    i - last_increase_generation >= number_of_generations_to_increase
                    and mutation_rate <= 0.2
                ):
                    if (
                        lowest_distance_by_generation[
                            -number_of_generations_to_increase
                        ]
                        == lowest_distance_by_generation[-1]
                    ):
                        mutation_rate += mutation_rate_increase_by
                        last_increase_generation = i
                        print(f"Mutation rate increased to {mutation_rate}")

        if plot_distances:
            self.plot_distances(lowest_distance_by_generation)

        if plot_route:
            self.plot_route(population[0])

        return population[0]

    def evolve_population(
        self,
        population,
        selection_method,
        crossover_method,
        tournament_size,
        mutation_method,
        mutation_rate,
    ):
        """Evolves a population of routes for the TSP problem. The evolution process consists of selection, crossover and mutation."""
        new_population = []
        for _ in range(len(population)):
            parents = self.selection(population, selection_method, tournament_size)
            offspring = self.crossover(parents, crossover_method)
            offspring = self.mutation(offspring, mutation_method, mutation_rate)
            new_population.append(offspring)

        return new_population

    def selection(self, population, method, tournament_size):
        """Selects two parents from the population, based on a given algorithm"""
        parents = []

        if method == "roulette":
            parents = self.roulette_selection(population)

        if method == "rank":
            parents = self.rank_selection(population)

        if method == "tournament":
            parents = self.tournament_selection(population, tournament_size)

        return parents

    def roulette_selection(self, population):
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

    def rank_selection(self, population):
        """Selects two parents from the population using the rank selection method."""
        population = self.sort_population(population)
        ranks = np.arange(len(population)) + 1
        probabilities = [rank / sum(ranks) for rank in ranks]
        indices = np.arange(len(population))
        selected_indices = np.random.choice(indices, 2, p=probabilities)
        parents = [population[i] for i in selected_indices]
        return parents

    def tournament_selection(self, population, tournament_size):
        """Selects two parents from the population using the tournament selection method."""
        parents = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = min(tournament, key=lambda x: self.get_route_distance(x))
            parents.append(winner)
        return parents

    def crossover(self, parents, method):
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
    ):
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

    def plot_route(self, route):
        """Plots the route for the TSP problem."""
        x = [self.cities[route[i], 0] for i in range(self.number_of_cities)]
        y = [self.cities[route[i], 1] for i in range(self.number_of_cities)]
        plt.plot(x, y, marker="o", color="blue")
        plt.plot([x[-1], x[0]], [y[-1], y[0]], marker="o", color="blue")
        # Starting point
        plt.plot(x[0], y[0], marker="o", color="red")
        plt.show()

    def plot_distances(self, distances):
        """Plots the lowest distance found for each generation of the genetic algorithm."""
        plt.plot(distances)
        plt.xlabel("Generation")
        plt.ylabel("Distance")
        plt.title("Evolution of the lowest distance")
        plt.show()

    def sort_population(self, population):
        """Sorts the population based on the total distance of each route (minimum distance first)."""
        return sorted(population, key=lambda x: (-1) * self.get_route_distance(x))

    def order_crossover(self, parent1, parent2):
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

    def pmx_crossover(self, parent1, parent2):
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

    def scramble_mutation(self, route):
        """Applies the scramble mutation method to a route for the TSP problem."""
        start, end = sorted(random.sample(range(self.number_of_cities), 2))
        scrambled = route[start:end]
        random.shuffle(scrambled)
        route[start:end] = scrambled

    def inversion_mutation(self, route):
        """Applies the inversion mutation method to a route for the TSP problem."""
        start, end = sorted(random.sample(range(self.number_of_cities), 2))
        route[start:end] = route[start:end][::-1]

    def swap_mutation(self, route):
        """Applies the swap mutation method to a route for the TSP problem."""
        indexes = random.sample(range(self.number_of_cities), 2)
        route[indexes[0]], route[indexes[1]] = route[indexes[1]], route[indexes[0]]

    def show_distances(self, population):
        """Prints the total distance of the shortest route in the population."""
        distances = []
        for individual in population:
            distances.append(self.get_route_distance(individual))
        return np.min(distances)

    def get_route_distance(self, route):
        """Calculates the total distance of a route for the TSP problem."""
        distance = 0
        for i in range(self.number_of_cities - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        distance += self.distance_matrix[route[-1], route[0]]

        return distance

    def generate_population(self, population_size, number_of_cities):
        """Generates a population of routes for the TSP problem. Each route is a permutation of the cities."""
        population = []

        for _ in range(population_size):
            route = np.random.permutation(number_of_cities)
            population.append(route.tolist())

        return population

    def get_distance_matrix(self):
        """Generates a distance matrix for the cities. The distance matrix is a symmetric matrix with the distances between each pair of cities."""
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))

        for i in range(self.number_of_cities):
            for j in range(i + 1, self.number_of_cities):
                distance = np.linalg.norm(self.cities[i] - self.cities[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix
