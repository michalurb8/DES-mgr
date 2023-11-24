from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.core.problem import Problem

p = get_problem("dtlz1", n_var=3, n_obj=2)
n = NSGA2()
a = DES()
res = minimize(p, a)
print(res.X)