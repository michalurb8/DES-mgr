from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.core.problem import Problem
from pymoo.indicators.gd import GD
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import numpy as np

p = get_problem("zdt1")
n = NSGA2()
a = DES()

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                        n_obj=2,
                        n_ieq_constr=0,
                        xl=-10,
                        xu=10)

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[0]
        x2 = x[1]

        f1 = 1 - np.exp(-(x1+2)**2 - x2**2)
        f2 = 2 - np.exp(- x1**2 -  (x2-2)**2) - np.exp(-x1**2 - (x2+2)**2)

        out["F"] = [f1, f2]
        out["G"] = []



mp = MyProblem()

res = minimize(mp, n, termination=get_termination("n_iter", 1))

resolution = 10

pf = np.array([np.array([0]*(mp.n_obj-1) + [i/resolution])  for i in range(-resolution, resolution+1)])


ind = GD(pf)
result = ind(res.pop.get('F'))

print(result)