from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import numpy as np

p = get_problem("zdt1")
# p = OmniTest(n_var=2)
pf = p.pareto_front()

v = False
nsga = NSGA2(visuals=v)
des = DES(visuals=v)

res = minimize(p, des, termination=get_termination("n_eval", 10000))

ind = GD(pf)
result = ind(res.opt.get('F'))

print(result)