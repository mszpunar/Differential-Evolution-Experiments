import math


def sphere(point):
    s = 0.0
    for x in point:
        s += x**2
    return s


def elliptic(point):
    s = 0.0
    for i, x in enumerate(point):
        s += 10**((i-1)/(len(point)-1)) * x**2
    return s


def rastrigin(point):
    s = 0.0
    for x in point:
        s += x**2 - 10*math.cos(2*math.pi*x) + 10
    return s


def ackley(point):
    D = len(point)
    sum_1 = sum([x**2 for x in point])
    sum_2 = sum([math.cos(2*math.pi*x) for x in point])
    return -20 * math.exp(-0.2 * math.sqrt(1/D * sum_1)) - math.exp(1/D * sum_2) + 20 + math.e


def schwefel(point):
    return sum([sum(point[:i])**2 for i in range(len(point))])


def rosenbrock(point):
    return sum([100*(point[i]**2 - point[i+1])**2 + (point[i] - 1)**2 for i in range(len(point)-1)])
