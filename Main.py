
from ObjectveFunction import ObjectiveFunction
from DE import DifferentialEvolution
import TargetFunctions


tf = TargetFunctions.sphere
dim = 2
c = [(-100, 100) for _ in range(dim)]
of = ObjectiveFunction(fun=tf, bounds=c, dim=dim, repair_method='projection')
de = DifferentialEvolution(
    objective_fun=of, crossover_method='EXP', tolerance=1e-50)
res = de.run()
print(res.best_point.__dict__)
