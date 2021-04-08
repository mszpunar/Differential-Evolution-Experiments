import numpy as np
import time
import math


class EndConditions:

    MAX_ITER = 10e6
    TOLERANCE = 0
    TOL_FUN = 0.0000001

    @staticmethod
    def max_iter(i, max_iter):
        return i >= max_iter

    @staticmethod
    def tol_fun(vec1, vec2, objective_fun):
        return False

    @staticmethod
    def tol(vec, tolerance):
        return vec.value <= tolerance

    @staticmethod
    def check_end_conditions(iteration, vec1, vec2, obj_fun, max_iter, tol):
        return {'max_iter': EndConditions.max_iter(iteration, max_iter),
                'tolfun': EndConditions.tol_fun(vec1=vec1, vec2=vec2, objective_fun=obj_fun),
                'tolerance': EndConditions.tol(vec=vec2, tolerance=tol)}


class Point:

    # during the creation of object, there's an evaluation of objective function hence it's done once
    def __init__(self, coordinates, objective_fun=None):
        self.coordinates = coordinates
        if objective_fun:
            self.defined = True
            self.value = objective_fun.eval(self.coordinates)
        else:
            self.defined = False

    def update(self, objective_fun):
        if not self.defined:
            self.value = objective_fun.eval(self.coordinates)
