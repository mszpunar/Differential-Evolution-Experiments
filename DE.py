from typing import Optional

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

    def __init__(self, objective_fun, lbd: int, max_iter: int = MAX_ITER, tolerance: float = MAX_TOL,
                 f: float = DEFAULT_F, cr: float = DEFAULT_CR,
                 crossover_method: str = 'BIN', selection_method: str = 'RAND'):
        self.objective_fun = objective_fun
        self.population = None
        self.point_dim = self.objective_fun.dim

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.end_reason: Optional[str] = None
        self.iterations = 0

        # ALGORITHM PARAMS
        self.cr = cr
        self.f = f
        self.lbd = lbd

        # ALGORITHM METHODS
        if crossover_method == 'EXP':
            self.crossover = self._exp_crossover
        else:
            self.crossover = self._bin_crossover

        if selection_method == 'RAND':
            self.selection_method = self._rand_selection
        elif selection_method == 'BEST':
            self.selection_method = self._best_selection
        elif selection_method == 'MEAN':
            self.selection_method = self._mean_selection
        elif selection_method == 'MEAN_VEC':
            self.selection_method = self._mean_vector_base_selection
        else:
            self.selection_method = self._rand_mean_selection

    # ------- PRIVATE METHODS ------- #

    def _get_best(self):
        # create an array of objective function values
        val_array = np.array([p.value for p in self.population])

        # Get the indices of minimum element
        min_index = np.where(val_array == np.amin(val_array))
        # minindex[0][0], because the value returned by np.where is a bit weird
        return self.population[min_index[0][0]]

    def _get_mean(self):
        # arrange population into matrix, where rows are points
        population_matrix = np.asmatrix(np.array([p.coordinates for p in self.population]))
        mean_point = np.asarray(population_matrix.mean(axis=0))[0]
        return Point(coordinates=mean_point, objective_fun=self.objective_fun)

    def _check_end_cond(self, prev_best):
        end_conditions = EndConditions.check_end_conditions(iteration=self.iterations,
                                                            vec1=prev_best, vec2=self._get_best(),
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

    def _gen_random_population(self, size=None, scaler=1):
        if not size:
            size = self.lbd
        generated = scaler * np.random.uniform(
            size=(size, np.random.randint(low=self.point_dim, high=self.point_dim + 1)))
        raw_population = np.array([Point(
            coordinates=generated[i], objective_fun=self.objective_fun) for i in range(len(generated))])

        if self.objective_fun.bounds:
            self.population = self.objective_fun.repair_points(raw_population)
        else:
            self.population = raw_population

    # ------- CROSSOVER METHODS ------- #

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
        i, n = 0, x.shape[0]
        while i < n:
            a = random()
            if a < self.cr:
                z[i] = y[i]
                i += 1
            else:
                i += 1
                break

        while i < n:
            z[i] = x[i]
            i += 1
        return z

    # --------- SELECTION METHODS --------- #
    def _rand_selection(self):
        # generates a random distinct 3 points
        points = np.random.choice(self.population, size=3, replace=False)
        r = points[0]
        d_e = points[1:]
        return r, d_e

    def _best_selection(self):
        # picks a best point and then picks two different random points
        r = self._get_best()
        d_e = np.random.choice(self.population, size=2, replace=False)
        while d_e[0] == r or d_e[1] == r:
            d_e = np.random.choice(self.population, size=2, replace=False)
        return r, d_e

    def _mean_selection(self):
        # picks a mean point and then picks two different random points
        r = self._get_mean()
        d_e = np.random.choice(self.population, size=2, replace=False)
        while d_e[0] == r or d_e[1] == r:
            d_e = np.random.choice(self.population, size=2, replace=False)
        return r, d_e

    def _rand_mean_selection(self):
        # picks two random points and avg of a third random point and a mean point
        r, d_e = self._rand_selection()
        mean = self._get_mean()
        new_coords = np.mean(np.array([r.coordinates, mean.coordinates]), axis=0)
        return Point(coordinates=new_coords, objective_fun=self.objective_fun), d_e

    def _mean_vector_base_selection(self):
        # picks two random points and a mean point
        r, d_e = self._rand_selection()
        d_e[0] = self._get_mean()
        return r, d_e

    # --------- GENERAL METHODS --------- #
    def _tournament(self, x, y):
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

    def _single_iteration(self):
        next_population = self.population
        for i in range(0, np.size(self.population, axis=0)):
            # SELECTION
            r, d_e = self.selection_method()

            # MUTATION and CROSSOVER
            m = r.coordinates + self.f * \
                (d_e[1].coordinates - d_e[0].coordinates)
            o = Point(self.crossover(self.population[i].coordinates, m))
            if self.objective_fun.bounds:
                o = self.objective_fun.repair_point(o)
            o.update(self.objective_fun)

            # SUCCESSION
            next_population[i] = self._tournament(self.population[i], o)

        prev_best = self._get_best()
        self.population = next_population
        self.iterations += 1
        return prev_best

    def run(self):
        # INITIALIZE
        self.iterations = 0
        self._gen_random_population(size=self.lbd)
        prev_best = self._get_best()

        # MAIN LOOP
        while not self._check_end_cond(prev_best):
            prev_best = self._single_iteration()
            print(prev_best.value)

        return Result(best_point=self._get_best(),
                      end_reason=self.end_reason,
                      iteration_num=self.iterations)
