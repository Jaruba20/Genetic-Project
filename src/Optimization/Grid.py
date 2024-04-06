import itertools
from Optimization import Optimization

class Grid:
    """Class to iterate over different parameters of the Optimization class and obtain the best results."""

    def __init__(self, optimization: Optimization, dict_of_parameters):
        """Constructor of the class.
        :param dict_of_parameters: Dictionary with the parameters to iterate over.
        """
        self.cities = optimization.cities
        self.dict_of_parameters = dict_of_parameters
        self.best_result = None
        self.best_parameters = None

    def iterate(self):
        """Method to iterate over the parameters and obtain the best result.
        :return: Tuple with the best result and the best parameters.
        """
        parameter_combinations = list(
            itertools.product(*self.dict_of_parameters.values())
        )
        for combination in parameter_combinations:
            self.dict_of_parameters = dict(
                zip(self.dict_of_parameters.keys(), combination)
            )
            miOptimization = Optimization(self.cities)
            result = miOptimization.optimize(
                population_size=self.dict_of_parameters["population_size"],
                generations=self.dict_of_parameters["generations"],
                selection_method=self.dict_of_parameters["selection_method"],
                tournament_size=self.dict_of_parameters["tournament_size"],
                crossover_method=self.dict_of_parameters["crossover_method"],
                mutation_method=self.dict_of_parameters["mutation_method"],
                mutation_rate_increase=self.dict_of_parameters[
                    "mutation_rate_increase"
                ],
                number_of_generations_to_increase=self.dict_of_parameters[
                    "number_of_generations_to_increase"
                ],
                mutation_rate_increase_by=self.dict_of_parameters[
                    "mutation_rate_increase_by"
                ],
                mutation_rate=self.dict_of_parameters["mutation_rate"],
                show_distances=self.dict_of_parameters["show_distances"],
                plot_distances=self.dict_of_parameters["plot_distances"],
                plot_route=self.dict_of_parameters["plot_route"],
            )
            if self.best_result is None or result[0] < self.best_result[0]:
                self.best_result = result
                self.best_parameters = self.dict_of_parameters
        return self.best_result, self.best_parameters
