from pymooDes import DES
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import numpy as np
import matplotlib.pyplot as plt

p = get_problem("zdt1")
p = OmniTest(n_var=5)
p = get_problem("dtlz1")

v = False
nsga = NSGA2(visuals=v)
des4 = DES(visuals=v, pop_size=4)
des20 = DES(visuals=v, pop_size=20)
des100 = DES(visuals=v, pop_size=100)

num = 2
n_eval = 500

alld = []
for i in range(num):
    resd = minimize(p, des20, termination=get_termination("n_eval", n_eval))
    values = np.array(resd.history)
    alld.append(values)
alld = np.mean(np.array(alld), axis=0)

alln = []
for i in range(num):
    resn = minimize(p, nsga, termination=get_termination("n_eval", n_eval))
    values = np.array(resn.history)
    alln.append(values)
alln = np.mean(np.array(alln), axis=0)

dx = [i[0] for i in alld]
dgd = [i[1] for i in alld]
digd = [i[2] for i in alld]

nx = [i[0] for i in alln]
ngd = [i[1] for i in alln]
nigd = [i[2] for i in alln]

fig, (axgd, axigd) = plt.subplots(1, 2)

axgd.set_title("Porównanie algorytmów względem metryki GD")
axgd.plot(dx, dgd, c='green', label='alg1')
axgd.plot(nx, ngd, c='blue', label='alg2')
axgd.set_xlabel("Liczba ocenionych punktów")
axgd.set_ylabel("Wartość metryki GD")
axgd.set_yscale('log')
axgd.grid()

axigd.set_title("Porównanie algorytmów względem metryki IGD")
axigd.plot(dx, digd, c='green')
axigd.plot(nx, nigd, c='blue')
axigd.set_xlabel("Liczba ocenionych punktów")
axigd.set_ylabel("Wartość metryki IGD")
axigd.set_yscale('log')
axigd.grid()

fig.suptitle(f"Uśrednione wartości metryk dla populacji zwracanej przez algorytmy.")
fig.legend()
plt.show()