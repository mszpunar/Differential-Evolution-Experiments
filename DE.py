import numpy as np
from numpy.random import random
from collections import namedtuple

from MinorClasses import Point
from MinorClasses import EndConditions
from Consts import *


Result = namedtuple(
    'Result', ['best_point', 'end_reason', 'iteration_num'])


class DifferentialEvolution:

    DEFAULT_CR = 0.5
    DEFAULT_F = 0.8

    def __init__(self, objective_fun, max_iter=MAX_ITER, tolerance=MAX_TOL, f=DEFAULT_F, cr=DEFAULT_CR, lbd=None, crossover_method='BIN'):
        self.objective_fun = objective_fun
        self.population = None
        self.point_dim = self.objective_fun.dim

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.end_reason = None  # STRING

        # ALGORITHM PARAMS
        self.cr = cr  # FLOAT - parameter for crossover
        self.f = f  # FLOAT
        self.lbd = None

        # ALGORITHM METHODS
        if crossover_method == 'EXP':
            self.crossover = self._exp_crossover
        else:
            self.crossover = self._bin_crossover

    # ------- PRIVATE METHODS ------- #

    def _sel_best(self):
        # create an array of objective function values
        val_array = np.array([p.value for p in self.population])

        # Get the indices of minimum element
        min_index = np.where(val_array == np.amin(val_array))
        # minindex[0][0], because the value returned by np.where is a bit weird
        return self.population[min_index[0][0]]

    def _check_end_cond(self, prev_best):
        end_conditions = EndConditions.check_end_conditions(iteration=self.iterations,
                                                            vec1=prev_best, vec2=self._sel_best(),
                                                            obj_fun=self.objective_fun,
                                                            max_iter=self.max_iter,
                                                            tol=self.tolerance)

        if self.iterations != 0 and any(end_conditions.values()):
            reason = next((i for i, j in enumerate(
                end_conditions.values()) if j), None)
            self.end_reason = list(end_conditions.keys())[reason]
            return True
        else:
            return False

    def _is_initialized(self, valid_parameters=None):
        if valid_parameters:
            parameter_list = [p for key, p in enumerate(
                self.__dict__.values()) if key in valid_parameters]
        else:
            parameter_list = self.__dict__.values()

        for val in parameter_list:
            if val is None:
                return False
        return True

    def _gen_random_population(self, size=50, scaler=1):
        generated = scaler * \
            np.random.uniform(size=(size, np.random.randint(
                low=self.point_dim, high=self.point_dim+1)))
        raw_population = np.array([Point(
            coordinates=generated[i], objective_fun=self.objective_fun) for i in range(len(generated))])

        if self.objective_fun.bounds:
            self.population = self.objective_fun.repair_points(raw_population)
        else:
            self.population = raw_population

    # ------- ALGORITHM METHODS ------- #

    def _bin_crossover(self, x, y):
        z = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            a = random()
            if a < self.cr:
                z[i] = y[i]
            else:
                z[i] = x[i]
        return z

    def _exp_crossover(self, x, y):
        z = np.empty(x.shape[0])
        # for i in range(x.shape[0]):
        i, n = 0, x.shape[0]
        while(i < n):
            a = random()
            if a < self.cr:
                z[i] = y[i]
                i += 1
            else:
                i += 1
                break

        while(i < n):
            z[i] = x[i]
            i += 1
        return z

    def tournament(self, x, y):
        res = self.objective_fun.compare(x, y)
        if res > 0:
            return y
        elif res < 0:
            return x
        else:
            if random() >= 0.5:
                return x
            else:
                return y

    def single_iteration(self):
        next_population = self.population
        for i in range(0, np.size(self.population, axis=0)):
            # SELECTION - generates a random distinct 3 indexes from <0,mi> and picks corresponding points
            indexes = np.random.choice(
                np.arange(self.population.shape[0]), 3, replace=False)
            r = self.population[indexes[0]]
            d_e = np.array([self.population[indexes[1]],
                            self.population[indexes[2]]])

            # MUTATION and CROSSOVER
            M = r.coordinates + self.f * \
                (d_e[1].coordinates - d_e[0].coordinates)
            O = Point(self.crossover(self.population[i].coordinates, M))
            if self.objective_fun.bounds:
                O = self.objective_fun.repair_point(O)
            O.update(self.objective_fun)

            next_population[i] = self.tournament(self.population[i], O)

        prev_best = self._sel_best()
        self.population = next_population
        self.iterations += 1
        return prev_best

    def run(self):
        self.iterations = 0
        if not self._is_initialized() and self.lbd:
            self._gen_random_population(size=self.lbd)
        elif not self._is_initialized():
            self._gen_random_population()
        prev_best = self._sel_best()

        # MAIN LOOP
        while not self._check_end_cond(prev_best):
            prev_best = self.single_iteration()

        return Result(best_point=self._sel_best(),
                      end_reason=self.end_reason,
                      iteration_num=self.iterations)
