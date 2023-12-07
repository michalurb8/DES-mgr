import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

# The pareto front of a scaled zdt1 problem
pf = get_problem("zdt1").pareto_front()

# The result found by an algorithm
A = pf[::10] * 1.1

# plot the result
# Scatter(legend=True).add(pf, label="Pareto-front").add(A, label="Result").show()

from pymoo.indicators.gd import GD

ind = GD(pf)
print("GD", ind(A))


print(len(A))
print(len(pf))