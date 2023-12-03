import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.core.repair import NoRepair

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=-10,
                         xu=10)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = (x[0]**2 + x[1]**2)
        f2 = ((x[0]-1)**2 + x[1]**2)

        # g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        # out["G"] = [g1, g2]
        out["G"] = []


problem = MyProblem()
alg = DES(visuals=True)

res = minimize(problem, alg, save_history=False)

X = res.X
print(X)
F = res.F